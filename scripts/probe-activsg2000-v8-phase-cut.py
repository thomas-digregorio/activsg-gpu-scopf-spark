#!/usr/bin/env python3
"""Exercise v8 logic using the preserved first-v7 commitment as diagnostic input."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from activsg_scopf.commitment_cuts import generate_commitment_cut_repairs
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.lagrangian import (
    RegionMasks,
    evaluate_lagrangian_bound,
    optimize_lagrangian_bound_cupy,
)
from activsg_scopf.lagrangian_experiment import (
    PrimalCandidatePolicy,
    RegionAttemptRejected,
    _solve_fixed_commitment_cost_projection,
    _solve_fixed_commitment_feasibility,
)
from activsg_scopf.matpower import PMAX, PMIN, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import reconstruct_full_values, security_pair_from_record
from activsg_scopf.screening import ContingencyScreener


def main() -> None:
    root = Path("/workspace")
    config = load_config(root / "configs" / "activsg2000-gpu-lagrangian-v8.json")
    evidence = json.loads(
        (
            root
            / "results"
            / "experiments"
            / "activsg2000-gpu-lagrangian-v7-dgx-spark.json"
        ).read_text(encoding="utf-8")
    )
    case = read_matpower_case(
        config.case_path,
        expected_sha256=config.raw["raw_inputs"]["case_sha256"],
    )
    table = read_contingency_table(
        config.contingency_path,
        expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
    )
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    root_region = next(
        record for record in evidence["solved_region_history"] if record["region_id"] == "r"
    )
    pairs = tuple(
        security_pair_from_record(
            record,
            catalog,
            lodf_absolute_tolerance=float(
                config.model["serialized_lodf_replay_tolerance"]
            ),
        )
        for record in root_region["security_pairs"]
    )
    source_rows = np.flatnonzero(case.gen[:, 7] > 0.0).astype(np.int64)
    first_attempt = evidence["primal_repairs"][0]
    committed_rows = {
        int(row) - 1 for row in first_attempt["committed_generator_source_rows"]
    }
    commitment = np.asarray(
        [int(int(row) in committed_rows) for row in source_rows], dtype=np.int8
    )
    root_on_values = np.asarray(
        [
            record["on_value"]
            for record in root_region["lagrangian_certificate"]["generator_subproblems"]
        ],
        dtype=np.float64,
    )
    screener = ContingencyScreener(
        network,
        catalog,
        backend="cupy",
        chunk_columns=int(config.model["screen_chunk_columns"]),
    )
    started = time.perf_counter()
    output: dict[str, object] = {
        "source_attempt_commitment_sha256": first_attempt["commitment_sha256"],
        "source_attempt_commitment_count": int(np.count_nonzero(commitment)),
    }
    try:
        _solve_fixed_commitment_feasibility(
            region_id="v7_p1_phase_cut_source",
            commitment=commitment,
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(180.0, 10.0, 2.0),
            initial_pairs=pairs,
            screener=screener,
            checkpoint=lambda: None,
            progress=None,
            policy=PrimalCandidatePolicy.from_config(config),
        )
    except RegionAttemptRejected as exc:
        output.update(
            {
                "source_status": "rejected",
                "source_reason": exc.reason,
                "source_wall_time_seconds": time.perf_counter() - started,
            }
        )
        cut = exc.commitment_feasibility_cut
        if cut is None:
            output["cut_status"] = "unavailable"
        else:
            output["cut_status"] = "replay_certified"
            output["cut"] = exc.commitment_feasibility_cut_record
            repairs = generate_commitment_cut_repairs(
                cut=cut,
                commitment=commitment,
                fixed_off=np.zeros(commitment.size, dtype=bool),
                fixed_on=np.zeros(commitment.size, dtype=bool),
                pmin_mw=case.gen[source_rows, PMIN],
                pmax_mw=case.gen[source_rows, PMAX],
                demand_mw=exc.master.operator.total_demand_mw,
                economic_on_values=root_on_values,
                maximum_repairs=3,
            )
            repair_records: list[dict[str, object]] = []
            secure_repair: tuple[np.ndarray, object] | None = None
            for position, (repair, audit) in enumerate(repairs, start=1):
                repair_started = time.perf_counter()
                try:
                    result = _solve_fixed_commitment_feasibility(
                        region_id=f"v7_p1_phase_cut_repair_{position}",
                        commitment=repair,
                        case=case,
                        network=network,
                        catalog=catalog,
                        config=config,
                        deadline=Deadline(180.0, 10.0, 2.0),
                        initial_pairs=pairs,
                        screener=screener,
                        checkpoint=lambda: None,
                        progress=None,
                        policy=PrimalCandidatePolicy.from_config(config),
                    )
                except RegionAttemptRejected as repair_exc:
                    repair_records.append(
                        {
                            "position": position,
                            "status": "rejected",
                            "reason": repair_exc.reason,
                            "wall_time_seconds": time.perf_counter() - repair_started,
                            "audit": audit,
                            "next_cut_available": (
                                repair_exc.commitment_feasibility_cut is not None
                            ),
                            "next_cut_source_violation_pu": (
                                None
                                if repair_exc.commitment_feasibility_cut is None
                                else (
                                    repair_exc.commitment_feasibility_cut
                                    .conservative_source_violation_pu
                                )
                            ),
                        }
                    )
                else:
                    repair_records.append(
                        {
                            "position": position,
                            "status": "secure",
                            "wall_time_seconds": time.perf_counter() - repair_started,
                            "audit": audit,
                            "round_count": len(result.rounds),
                            "final_screen": result.final_screen,
                            "source_residual_pu": (
                                result.master.canonical.max_row_violation(
                                    result.source_values
                                )
                                / float(case.base_mva)
                            ),
                        }
                    )
                    secure_repair = (repair.copy(), result)
                    break
            output["repair_results"] = repair_records
            if secure_repair is not None:
                repair_commitment, repair_result = secure_repair
                cost_started = time.perf_counter()
                cost_result = _solve_fixed_commitment_cost_projection(
                    region_id="v8_phase_cut_repair_cost_probe",
                    commitment=repair_commitment,
                    case=case,
                    network=network,
                    catalog=catalog,
                    config=config,
                    deadline=Deadline(180.0, 10.0, 2.0),
                    prepared_master=repair_result.master,
                    initial_source_values=repair_result.source_values,
                    initial_pairs=repair_result.security_pairs,
                    screener=screener,
                    checkpoint=lambda: None,
                    progress=None,
                    policy=PrimalCandidatePolicy.from_config(config),
                )
                cost_full, cost_full_values = reconstruct_full_values(
                    case,
                    network,
                    cost_result.master,
                    cost_result.source_values,
                    exact_commitment=repair_commitment,
                )
                output["cost_polish"] = {
                    "status": cost_result.projected_solve.status,
                    "objective": float(
                        np.asarray(cost_full.canonical.objective) @ cost_full_values
                    ),
                    "wall_time_seconds": time.perf_counter() - cost_started,
                    "pricing_certified": cost_result.pricing_certified,
                    "pricing_audit": cost_result.pricing_audit,
                    "final_screen": cost_result.final_screen,
                    "source_residual_pu": (
                        cost_result.master.canonical.max_row_violation(
                            cost_result.source_values
                        )
                        / float(case.base_mva)
                    ),
                }
                full, full_values = reconstruct_full_values(
                    case,
                    network,
                    repair_result.master,
                    repair_result.source_values,
                    exact_commitment=repair_commitment,
                )
                feasible_cost = float(np.asarray(full.canonical.objective) @ full_values)
                root_dual_by_name = {
                    str(record["row_name"]): float(record["canonical_row_dual"])
                    for record in root_region["lagrangian_certificate"][
                        "coupling_row_duals"
                    ]
                }
                root_row_dual = np.zeros(exc.master.canonical.num_rows)
                for coupling in exc.master.coupling_rows:
                    root_row_dual[coupling.row_index] = root_dual_by_name.get(
                        coupling.row_name, 0.0
                    )
                polished, bound_audit = optimize_lagrangian_bound_cupy(
                    exc.master,
                    root_row_dual,
                    RegionMasks.root(commitment.size),
                    relaxation_primal_objective=feasible_cost,
                    iterations=8192,
                    polyak_fraction=float(
                        config.raw["platforms"]["dgx_spark"][
                            "lagrangian_polyak_fraction"
                        ]
                    ),
                    commitment_cuts=(cut,),
                    initial_commitment_cut_dual=np.zeros(1),
                )
                best_cut_dual = np.asarray(
                    bound_audit["best_commitment_cut_dual"], dtype=np.float64
                )
                replayed_bound = evaluate_lagrangian_bound(
                    exc.master,
                    polished,
                    RegionMasks.root(commitment.size),
                    safety_margin_dollars=float(
                        config.raw["benchmark"]["certificate_safety_margin_dollars"]
                    ),
                    commitment_cuts=(cut,),
                    commitment_cut_dual=best_cut_dual,
                )
                output["cut_lagrangian_bound"] = {
                    "feasible_gpu_repair_cost_target": feasible_cost,
                    "initial_raw_lower_bound": bound_audit[
                        "initial_raw_lower_bound"
                    ],
                    "best_raw_lower_bound": bound_audit["best_raw_lower_bound"],
                    "improvement_dollars": bound_audit["improvement_dollars"],
                    "best_cut_dual": best_cut_dual.tolist(),
                    "independent_cpu_replay_conservative_lower_bound": (
                        replayed_bound.conservative_lower_bound
                    ),
                    "minimizing_commitment_count": int(
                        np.count_nonzero(replayed_bound.minimizing_commitment)
                    ),
                    "iterations": 8192,
                }
    else:
        output["source_status"] = "unexpectedly_secure"
    output.update(
        {
            "total_wall_time_seconds": time.perf_counter() - started,
            "gpu_only": True,
            "cpu_solution_data_read": False,
            "exact_source_pmin_pmax": True,
            "diagnostic_only": True,
        }
    )
    print("PHASE_CUT_DIAGNOSTIC=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
