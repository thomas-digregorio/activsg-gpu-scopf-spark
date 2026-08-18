#!/usr/bin/env python3
"""One bounded ACTIVSg2000 probe of the exact v6 GPU primal pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from activsg_scopf.config import load_config
from activsg_scopf.matpower import read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    commitment_vector,
    reconstruct_full_values,
    security_pair_from_record,
)
from activsg_scopf.screening import ContingencyScreener, add_security_pairs
from activsg_scopf.solvers.cuopt import (
    FULL_MIP_START,
    native_scaling_vectors,
    solve_cuopt,
    validate_full_mip_start_feasibility,
)
from activsg_scopf.solvers.cuopt_lp import derive_rate_a_angle_bounds


def trace_summary(result: object) -> dict[str, object]:
    trace = result.statistics["incumbent_commitment_trace"]
    return {
        key: trace.get(key)
        for key in (
            "callback_count",
            "commitment_transition_count",
            "unique_commitment_count",
            "last_commitment_change_elapsed_seconds",
            "stabilization_window_seconds_at_solver_return",
            "final_commitment_fingerprint_sha256",
            "complete",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/configs/activsg2000-gpu-lagrangian-v6.json"),
    )
    parser.add_argument(
        "--gpu-evidence",
        type=Path,
        default=Path(
            "/workspace/results/experiments/"
            "activsg2000-gpu-lagrangian-v5-dgx-spark.json"
        ),
    )
    parser.add_argument("--seed-seconds", type=float, default=75.0)
    parser.add_argument("--full-seconds", type=float, default=20.0)
    args = parser.parse_args()

    config = load_config(args.config)
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
    evidence = json.loads(args.gpu_evidence.read_text(encoding="utf-8"))
    root_record = next(
        record
        for record in evidence["solved_region_history"]
        if record["region_id"] == "r"
    )
    pairs = tuple(
        security_pair_from_record(
            record,
            catalog,
            lodf_absolute_tolerance=float(
                config.model["serialized_lodf_replay_tolerance"]
            ),
        )
        for record in root_record["security_pairs"]
    )
    reduced = build_reduced_master(
        case,
        network,
        segments=int(config.model["pwl_segments"]),
        coefficient_zero_tolerance=float(
            config.model["reduced_coefficient_zero_tolerance"]
        ),
    )
    add_reduced_security_pairs(reduced, network, pairs)
    profile = config.raw["platforms"]["dgx_spark"]
    seed_started = time.perf_counter()
    seed = solve_cuopt(
        reduced.canonical,
        time_limit_seconds=float(args.seed_seconds),
        mip_relative_gap=float(config.model["mip_relative_gap_tolerance"]),
        native_scaling_mode=str(profile["native_scaling_mode"]),
        native_base_mva=float(case.base_mva),
        log_to_console=True,
        track_incumbent_commitments=True,
        mip_heuristics_only=True,
    )
    seed_wall = time.perf_counter() - seed_started
    if seed.values is None:
        raise RuntimeError("Reduced GPU heuristic returned no incumbent")
    seed_values = np.asarray(seed.values, dtype=np.float64)
    commitment_values = commitment_vector(reduced, seed_values)
    commitment = np.rint(commitment_values)
    if float(np.max(np.abs(commitment_values - commitment))) > 1e-5:
        raise RuntimeError("Reduced GPU heuristic commitment is not integral")

    full, full_start = reconstruct_full_values(
        case,
        network,
        reduced,
        seed_values,
        exact_commitment=commitment,
    )
    add_security_pairs(full.canonical, full.index, network, pairs)
    bounds = derive_rate_a_angle_bounds(
        network,
        full.index.theta_by_bus,
        total_columns=full.canonical.num_columns,
    )
    _, original_lower, original_upper, _ = full.canonical.column_arrays()
    tightened_lower = np.maximum(original_lower, bounds.lower)
    tightened_upper = np.minimum(original_upper, bounds.upper)
    tightened = np.flatnonzero(
        (tightened_lower > original_lower) | (tightened_upper < original_upper)
    )
    for column in tightened:
        position = int(column)
        full.canonical.column_lower[position] = float(tightened_lower[position])
        full.canonical.column_upper[position] = float(tightened_upper[position])
    column_scale, row_scale = native_scaling_vectors(
        full.canonical,
        mode=str(profile["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    start_audit = validate_full_mip_start_feasibility(
        full.canonical,
        full_start,
        column_scale=column_scale,
        row_scale=row_scale,
        tolerance=float(config.model["model_residual_tolerance_pu"]),
    )
    full_started = time.perf_counter()
    improved = solve_cuopt(
        full.canonical,
        time_limit_seconds=float(args.full_seconds),
        mip_relative_gap=float(config.model["mip_relative_gap_tolerance"]),
        mip_start_values=full_start,
        mip_start_mode=FULL_MIP_START,
        native_scaling_mode=str(profile["native_scaling_mode"]),
        native_base_mva=float(case.base_mva),
        log_to_console=True,
        track_incumbent_commitments=True,
        mip_heuristics_only=True,
    )
    full_wall = time.perf_counter() - full_started
    if improved.values is None:
        raise RuntimeError("Sparse full GPU heuristic lost its feasible MIP start")
    improved_values = np.asarray(improved.values, dtype=np.float64)
    screened = ContingencyScreener(
        network,
        catalog,
        backend="cupy",
        chunk_columns=int(config.model["screen_chunk_columns"]),
    ).screen(
        improved_values[full.index.flow_by_active_branch],
        tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
        already_added={pair.pair_id for pair in pairs},
    )
    print(
        "V6_PRIMAL_PIPELINE_PROBE="
        + json.dumps(
            {
                "seed": {
                    "status": seed.status,
                    "objective": seed.objective,
                    "wall_time_seconds": seed_wall,
                    "canonical_residual_pu": (
                        reduced.canonical.max_row_violation(seed_values)
                        / float(case.base_mva)
                    ),
                    "trace": trace_summary(seed),
                },
                "full": {
                    "status": improved.status,
                    "objective": improved.objective,
                    "wall_time_seconds": full_wall,
                    "canonical_residual_pu": (
                        full.canonical.max_row_violation(improved_values)
                        / float(case.base_mva)
                    ),
                    "mip_start_contract": improved.statistics[
                        "mip_start_native_contract"
                    ],
                    "trace": trace_summary(improved),
                },
                "start_audit": start_audit,
                "commitment_count": int(np.count_nonzero(commitment)),
                "security_pair_count": len(pairs),
                "new_security_pairs": len(screened.violations),
                "maximum_exhaustive_security_violation_pu": (
                    screened.maximum_violation_pu
                ),
                "cpu_solution_data_read": False,
                "diagnostic_only": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
