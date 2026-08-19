#!/usr/bin/env python3
"""Test a GPU root-certificate minimizer with exact Phase I and one global cut.

This is a bounded development probe, not a registered experiment.  It reads
only the preserved GPU v12 certificate, rebuilds the mathematical model from
the immutable raw inputs, and never reads a CPU objective, bound, commitment,
or dispatch.  The purpose is to determine whether a replay-certified Phase-I
cut against the current Lagrangian minimizer can materially strengthen the
GPU-resident lower bound before a full replacement run is authorized.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from activsg_scopf.commitment_cuts import commitment_upper_cut_from_record
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.lagrangian import (
    RegionMasks,
    evaluate_lagrangian_bound,
    optimize_commitment_cut_duals_coordinate_cupy,
    replay_lagrangian_certificate,
)
from activsg_scopf.lagrangian_experiment import (
    PrimalCandidatePolicy,
    RegionAttemptRejected,
    _certificate_dual_arrays,
    _commitment_upper_cut_record,
    _prepare_region_master,
    _solve_fixed_commitment_feasibility,
    _validate_prepared_region_master,
)
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import security_pair_from_record
from activsg_scopf.screening import ContingencyScreener


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _commitment_sha256(commitment: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(commitment, dtype=np.int8).tobytes()).hexdigest()


def main() -> None:
    started = time.perf_counter()
    workspace = Path("/workspace")
    config = load_config(workspace / "configs" / "activsg2000-gpu-lagrangian-v12.json")
    prior = _read_json(
        workspace
        / "results"
        / "experiments"
        / "activsg2000-gpu-lagrangian-v12-dgx-spark.json"
    )
    if prior["frozen_identity"]["tag"] != "experiment-2000-gpu-lagrangian-v12":
        raise RuntimeError("The preserved v12 GPU fixture identity changed")

    case = read_matpower_case(
        config.case_path,
        expected_sha256=config.raw["raw_inputs"]["case_sha256"],
    )
    table = read_contingency_table(
        config.contingency_path,
        expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
    )
    network = build_network(case)
    catalog = build_contingency_catalog(
        case,
        network,
        table,
        validation_columns=int(config.model["lodf_validation_columns"]),
        validation_tolerance_pu=float(config.model["lodf_validation_tolerance_pu"]),
        chunk_columns=int(config.model["lodf_build_chunk_columns"]),
    )
    source_rows = np.flatnonzero(case.gen[:, GEN_STATUS] > 0).astype(np.int64)
    root_record = next(
        record
        for record in prior["solved_region_history"]
        if str(record["region_id"]) == "r"
    )
    root_pairs = tuple(
        security_pair_from_record(
            record,
            catalog,
            lodf_absolute_tolerance=float(config.model["serialized_lodf_replay_tolerance"]),
        )
        for record in root_record["security_pairs"]
    )
    existing_cuts = tuple(
        commitment_upper_cut_from_record(record, source_rows + 1)
        for record in root_record["commitment_upper_cuts"]
    )
    masks = RegionMasks.root(source_rows.size)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=root_pairs,
        commitment_cuts=existing_cuts,
    )
    _validate_prepared_region_master(master, masks, root_pairs, existing_cuts)
    replayed_root = replay_lagrangian_certificate(
        master,
        root_record["lagrangian_certificate"],
        masks,
        commitment_cuts_by_id={cut.cut_id: cut for cut in existing_cuts},
    )
    root_row_dual, root_cut_dual = _certificate_dual_arrays(
        master, replayed_root, existing_cuts
    )
    minimizer = np.asarray(replayed_root.minimizing_commitment, dtype=np.int8)
    screener = ContingencyScreener(
        network,
        catalog,
        backend="cupy",
        chunk_columns=int(config.model["screen_chunk_columns"]),
    )
    policy = PrimalCandidatePolicy(
        total_seconds=180.0,
        maximum_round_seconds=60.0,
        minimum_round_seconds=1.0,
        stagnation_window_rounds=3,
        minimum_relative_residual_improvement=0.001,
        dual_divergence_multiple=1_000_000.0,
        cold_restart_attempts=1,
    )
    output: dict[str, Any] = {
        "passed": False,
        "development_component_only": True,
        "gpu_only_problem_solution_data": True,
        "cpu_problem_solution_data_used": False,
        "exact_source_pmin_pmax": True,
        "root_certificate_replay_bound": replayed_root.conservative_lower_bound,
        "root_certificate_recorded_bound": float(
            root_record["lagrangian_certificate"]["conservative_lower_bound"]
        ),
        "root_certificate_replay_difference_dollars": abs(
            replayed_root.conservative_lower_bound
            - float(root_record["lagrangian_certificate"]["conservative_lower_bound"])
        ),
        "existing_global_cut_count": len(existing_cuts),
        "root_minimizer_commitment_count": int(np.count_nonzero(minimizer)),
        "root_minimizer_commitment_sha256": _commitment_sha256(minimizer),
        "phase_one_policy": policy.as_dict(),
    }
    phase_started = time.perf_counter()
    try:
        feasible = _solve_fixed_commitment_feasibility(
            region_id="v14_root_lagrangian_minimizer",
            commitment=minimizer,
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(240.0, 15.0, 5.0),
            initial_pairs=root_pairs,
            screener=screener,
            checkpoint=lambda: None,
            progress=None,
            policy=policy,
        )
    except RegionAttemptRejected as exc:
        output["root_minimizer_phase_one"] = {
            "status": "rejected",
            "reason": exc.reason,
            "wall_time_seconds": time.perf_counter() - phase_started,
            "rounds": exc.rounds,
            "replay_certified_global_cut_available": (
                exc.commitment_feasibility_cut is not None
            ),
        }
        new_cut = exc.commitment_feasibility_cut
        if new_cut is None:
            output["passed"] = True
            output["component_gate"] = "no_replay_certified_global_cut_available"
        else:
            cut_record = _commitment_upper_cut_record(new_cut, source_rows)
            output["root_minimizer_phase_one"]["global_cut"] = cut_record
            combined_cuts = tuple(sorted(existing_cuts + (new_cut,), key=lambda cut: cut.cut_id))
            strengthened_master = _prepare_region_master(
                case=case,
                network=network,
                config=config,
                masks=masks,
                initial_pairs=tuple(sorted(exc.security_pairs)),
                commitment_cuts=combined_cuts,
            )
            _validate_prepared_region_master(
                strengthened_master,
                masks,
                tuple(sorted(exc.security_pairs)),
                combined_cuts,
            )
            inherited_row_dual, inherited_cut_dual = _certificate_dual_arrays(
                strengthened_master, replayed_root, combined_cuts
            )
            inherited = evaluate_lagrangian_bound(
                strengthened_master,
                inherited_row_dual,
                masks,
                safety_margin_dollars=float(replayed_root.safety_margin_dollars),
                commitment_cuts=combined_cuts,
                commitment_cut_dual=inherited_cut_dual,
            )
            coordinate_started = time.perf_counter()
            optimized_cut_dual, audit = optimize_commitment_cut_duals_coordinate_cupy(
                strengthened_master,
                inherited_row_dual,
                masks,
                commitment_cuts=combined_cuts,
                initial_commitment_cut_dual=inherited_cut_dual,
                cycles=64,
            )
            coordinate_wall = time.perf_counter() - coordinate_started
            strengthened = evaluate_lagrangian_bound(
                strengthened_master,
                inherited_row_dual,
                masks,
                safety_margin_dollars=float(replayed_root.safety_margin_dollars),
                commitment_cuts=combined_cuts,
                commitment_cut_dual=optimized_cut_dual,
            )
            replay_difference = abs(
                float(audit["best_raw_lower_bound"]) - strengthened.raw_lower_bound
            )
            if replay_difference > float(
                config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]
            ):
                raise RuntimeError(
                    "GPU cut coordinate certificate failed exact replay"
                ) from exc
            if strengthened.conservative_lower_bound + 1e-6 < inherited.conservative_lower_bound:
                raise RuntimeError("GPU cut coordinate ascent weakened the certificate") from exc
            required_bound = float(prior["objective"]) * (
                1.0 - float(config.model["mip_relative_gap_tolerance"])
            )
            required_lift = max(
                0.0, required_bound - inherited.conservative_lower_bound
            )
            achieved_lift = max(
                0.0,
                strengthened.conservative_lower_bound
                - inherited.conservative_lower_bound,
            )
            output["cut_coordinate_ascent"] = {
                "cycles": 64,
                "wall_time_seconds": coordinate_wall,
                "inherited_bound": inherited.conservative_lower_bound,
                "strengthened_bound": strengthened.conservative_lower_bound,
                "improvement_dollars": achieved_lift,
                "required_bound_for_gpu_incumbent_at_requested_gap": required_bound,
                "required_lift_dollars": required_lift,
                "achieved_fraction_of_required_lift": (
                    1.0 if required_lift == 0.0 else achieved_lift / required_lift
                ),
                "nonzero_cut_dual_count": int(np.count_nonzero(optimized_cut_dual)),
                "new_cut_dual": float(
                    optimized_cut_dual[
                        next(
                            index
                            for index, cut in enumerate(combined_cuts)
                            if cut.cut_id == new_cut.cut_id
                        )
                    ]
                ),
                "new_minimizer_commitment_count": int(
                    np.count_nonzero(strengthened.minimizing_commitment)
                ),
                "new_minimizer_commitment_sha256": _commitment_sha256(
                    strengthened.minimizing_commitment
                ),
                "minimizer_changed": bool(
                    np.any(strengthened.minimizing_commitment != minimizer)
                ),
                "cpu_replay_difference_dollars": replay_difference,
                "host_transfer_during_coordinates": audit[
                    "host_transfer_during_coordinates"
                ],
            }
            output["passed"] = True
            output["component_gate"] = (
                "material" if required_lift == 0.0 or achieved_lift >= 0.01 * required_lift
                else "insufficient_bound_lift"
            )
    else:
        output["root_minimizer_phase_one"] = {
            "status": "secure",
            "wall_time_seconds": time.perf_counter() - phase_started,
            "round_count": len(feasible.rounds),
            "final_screen": feasible.final_screen,
            "canonical_model_residual_pu": (
                feasible.master.canonical.max_row_violation(feasible.source_values)
                / float(case.base_mva)
            ),
        }
        output["passed"] = True
        output["component_gate"] = "root_lagrangian_minimizer_is_secure"

    output["total_wall_time_seconds"] = time.perf_counter() - started
    print("V14_ROOT_MINIMIZER_CUT_PROBE=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
