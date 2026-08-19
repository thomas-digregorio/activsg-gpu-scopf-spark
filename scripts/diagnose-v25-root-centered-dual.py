"""Search the v25 root Lagrangian dual without CPU solution data.

The weakest v25 leaf contains only hard-cardinality branches.  Removing those
branches recovers the globally valid root relaxation.  This diagnostic solves
conditioned, centered GPU proposal LPs with every currently violated coupling
row available, then accepts only exactly replayed improvements of that root
certificate.  The proposal LP objective is never reported as a bound.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from activsg_scopf.commitment_cuts import (
    CommitmentCardinalityCut,
    commitment_upper_cut_from_record,
)
from activsg_scopf.config import load_config
from activsg_scopf.lagrangian import (
    RegionMasks,
    build_lagrangian_multiplier_delta_search_model,
    evaluate_lagrangian_bound,
    expand_lagrangian_multiplier_delta_candidate,
    replay_lagrangian_certificate,
)
from activsg_scopf.lagrangian_experiment import (
    _masks_from_record,
    _security_row_raw_violation_envelope_mw,
    _solve_summary,
    validate_lagrangian_experiment_config,
)
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    security_pair_from_record,
)
from activsg_scopf.solvers.cuopt_lp import solve_cuopt_continuous_pdlp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seconds-per-pass", type=float, default=10.0)
    parser.add_argument("--maximum-passes", type=int, default=3)
    args = parser.parse_args()
    if args.seconds_per_pass <= 0.0 or args.maximum_passes < 1:
        raise ValueError("Diagnostic budgets must be positive")

    started = time.perf_counter()
    config = load_config(args.config)
    validate_lagrangian_experiment_config(config)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    if payload.get("benchmark_id") != config.benchmark_id:
        raise RuntimeError("Diagnostic result/config benchmark identity differs")
    if payload.get("bound_status") != "independently_replayed_current_frontier":
        raise RuntimeError("Diagnostic requires an independently replayed frontier")

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
    source_rows = np.flatnonzero(case.gen[:, GEN_STATUS] > 0).astype(np.int64) + 1
    record = min(
        payload["frontier_regions"],
        key=lambda item: (
            float(item["lagrangian_certificate"]["conservative_lower_bound"]),
            str(item["region_id"]),
        ),
    )
    leaf_masks = _masks_from_record(record, source_rows)
    if np.any(leaf_masks.fixed_off) or np.any(leaf_masks.fixed_on):
        raise RuntimeError("Weakest v25 leaf unexpectedly contains binary fixings")
    pairs = tuple(
        security_pair_from_record(
            pair_record,
            catalog,
            lodf_absolute_tolerance=float(
                config.model.get("serialized_lodf_replay_tolerance", 0.0)
            ),
        )
        for pair_record in record["security_pairs"]
    )
    master = build_reduced_master(
        case,
        network,
        segments=int(config.model["pwl_segments"]),
        coefficient_zero_tolerance=float(
            config.model.get("reduced_coefficient_zero_tolerance", 1e-14)
        ),
        security_row_maximum_raw_violation_envelope_mw=(
            _security_row_raw_violation_envelope_mw(
                config,
                base_mva=float(case.base_mva),
            )
        ),
    )
    add_reduced_security_pairs(master, network, pairs)
    cuts = tuple(
        commitment_upper_cut_from_record(cut_record, source_rows)
        for cut_record in record.get("commitment_upper_cuts", [])
    )
    cut_by_id = {cut.cut_id: cut for cut in cuts}
    replayed_leaf = replay_lagrangian_certificate(
        master,
        record["lagrangian_certificate"],
        leaf_masks,
        commitment_cuts_by_id=cut_by_id,
    )
    row_by_name = {row.row_name: row.row_index for row in master.coupling_rows}
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    for row_name, value in replayed_leaf.coupling_duals:
        row_dual[row_by_name[row_name]] = value
    supplied_by_id = dict(replayed_leaf.commitment_cut_duals)
    nonhard_cuts = tuple(
        cut for cut in cuts if not isinstance(cut, CommitmentCardinalityCut)
    )
    cut_dual = np.asarray(
        [supplied_by_id.get(cut.cut_id, 0.0) for cut in nonhard_cuts],
        dtype=np.float64,
    )
    root_masks = RegionMasks.root(source_rows.size)
    current = evaluate_lagrangian_bound(
        master,
        row_dual,
        root_masks,
        safety_margin_dollars=float(
            config.raw["benchmark"]["certificate_safety_margin_dollars"]
        ),
        commitment_cuts=nonhard_cuts,
        commitment_cut_dual=cut_dual,
    )
    initial_root_bound = current.conservative_lower_bound
    schedule = (
        (100.0, 10_000.0),
        (1_000.0, 100_000.0),
        (10_000.0, 1_000_000.0),
    )[: args.maximum_passes]
    passes: list[dict[str, object]] = []
    profile = config.raw["platforms"]["dgx_spark"]
    for pass_number, (coupling_radius, cut_radius) in enumerate(schedule, start=1):
        search_started = time.perf_counter()
        search = build_lagrangian_multiplier_delta_search_model(
            master,
            row_dual,
            root_masks,
            commitment_cuts=nonhard_cuts,
            commitment_cut_dual=cut_dual,
            maximum_new_violated_coupling_rows=master.canonical.num_rows,
            coupling_trust_radius=coupling_radius,
            commitment_cut_trust_radius=cut_radius,
            search_coefficient_zero_tolerance=float(
                config.runtime["centered_dual_search_coefficient_zero_tolerance"]
            ),
            search_objective_zero_tolerance=float(
                config.runtime.get(
                    "centered_dual_search_objective_zero_tolerance", 0.0
                )
            ),
            include_all_coupling_rows=True,
        )
        build_wall = time.perf_counter() - search_started
        solve_started = time.perf_counter()
        solve = solve_cuopt_continuous_pdlp(
            search.canonical,
            time_limit_seconds=float(args.seconds_per_pass),
            optimality_tolerance=float(
                config.runtime["centered_dual_optimality_tolerance"]
            ),
            primal_feasibility_tolerance=float(
                config.runtime["centered_dual_primal_feasibility_tolerance"]
            ),
            certificate_residual_tolerance=float(
                config.runtime["centered_dual_certificate_residual_tolerance"]
            ),
            native_scaling_mode="none",
            native_base_mva=float(case.base_mva),
            log_to_console=True,
            per_constraint_residual=True,
            presolve=0,
            initial_native_primal=search.initial_values,
            pdlp_solver_mode=int(profile.get("pdlp_solver_mode_native", 1)),
        )
        solve_wall = time.perf_counter() - solve_started
        pass_record: dict[str, object] = {
            "pass": pass_number,
            "coupling_radius": coupling_radius,
            "cut_radius": cut_radius,
            "build_wall_time_seconds": build_wall,
            "solve_wall_time_seconds": solve_wall,
            "search_model": search.audit,
            "search_solve": _solve_summary(solve),
            "center_exact_bound": current.conservative_lower_bound,
            "search_lp_solution_used_as_bound": False,
        }
        if solve.values is None or not np.all(np.isfinite(solve.values)):
            pass_record["status"] = "no_finite_search_vector"
            passes.append(pass_record)
            continue
        candidate_row_dual, candidate_cut_dual, reconstruction = (
            expand_lagrangian_multiplier_delta_candidate(master, search, solve.values)
        )
        candidate = evaluate_lagrangian_bound(
            master,
            candidate_row_dual,
            root_masks,
            safety_margin_dollars=float(
                config.raw["benchmark"]["certificate_safety_margin_dollars"]
            ),
            commitment_cuts=nonhard_cuts,
            commitment_cut_dual=candidate_cut_dual,
        )
        accepted = candidate.conservative_lower_bound > current.conservative_lower_bound
        pass_record.update(
            {
                "status": (
                    "exact_candidate_accepted"
                    if accepted
                    else "exact_candidate_rejected_monotone"
                ),
                "candidate_exact_bound": candidate.conservative_lower_bound,
                "exact_lift_over_center_dollars": (
                    candidate.conservative_lower_bound
                    - current.conservative_lower_bound
                ),
                "candidate_reconstruction": reconstruction,
            }
        )
        if accepted:
            current = candidate
            row_dual = candidate_row_dual
            cut_dual = candidate_cut_dual
        passes.append(pass_record)

    print(
        "V25_ROOT_CENTERED_DUAL="
        + json.dumps(
            {
                "benchmark_id": config.benchmark_id,
                "weakest_region_id": record["region_id"],
                "weakest_leaf_exact_bound": replayed_leaf.conservative_lower_bound,
                "initial_root_exact_bound": initial_root_bound,
                "final_root_exact_bound": current.conservative_lower_bound,
                "exact_bound_lift_dollars": (
                    current.conservative_lower_bound - initial_root_bound
                ),
                "passes": passes,
                "total_wall_time_seconds": time.perf_counter() - started,
                "cpu_solution_data_used": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
