#!/usr/bin/env python3
"""Reproduce the v6 fixed-commitment PDLP continuation under the v7 adapter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.lagrangian import RegionMasks
from activsg_scopf.lagrangian_experiment import (
    PrimalCandidatePolicy,
    _prepare_region_master,
    _solve_fixed_commitment_cost_projection,
)
from activsg_scopf.matpower import read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import security_pair_from_record
from activsg_scopf.screening import ContingencyScreener
from activsg_scopf.solvers.cuopt import native_scaling_vectors
from activsg_scopf.solvers.cuopt_lp import solve_cuopt_continuous_pdlp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/configs/activsg2000-gpu-lagrangian-v7.json"),
    )
    parser.add_argument(
        "--gpu-evidence",
        type=Path,
        default=Path(
            "/workspace/results/experiments/"
            "activsg2000-gpu-lagrangian-v6-dgx-spark.json"
        ),
    )
    parser.add_argument("--slice-seconds", type=float, default=2.0)
    parser.add_argument("--projected-cost", action="store_true")
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
    checkpoint = evidence["secure_incumbent_checkpoint"]
    if checkpoint["status"] != "independently_verified":
        raise RuntimeError("The GPU diagnostic seed was not independently verified")
    root = next(
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
        for record in root["security_pairs"]
    )
    generator_records = {
        int(record["source_row"]) - 1: record
        for record in checkpoint["solution"]["generators"]
        if int(record["source_status"]) == 1
    }
    source_rows = np.asarray(sorted(generator_records), dtype=np.int64)
    commitment = np.asarray(
        [int(round(float(generator_records[int(row)]["commitment"]))) for row in source_rows],
        dtype=np.int8,
    )
    masks = RegionMasks(commitment == 0, commitment == 1)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=pairs,
    )
    if not np.array_equal(master.index.generator_source_rows, source_rows):
        raise RuntimeError("GPU checkpoint generator source identities changed")

    source_values = np.zeros(master.canonical.num_columns, dtype=np.float64)
    for row in source_rows:
        source_row = int(row)
        record = generator_records[source_row]
        source_values[master.index.commitment_by_generator[source_row]] = float(
            record["commitment"]
        )
        source_values[master.index.dispatch_by_generator[source_row]] = float(
            record["dispatch_mw"]
        )
        for column, segment_dispatch in zip(
            master.index.segments_by_generator[source_row],
            record["segment_dispatch_mw"],
            strict=True,
        ):
            if column is not None:
                source_values[column] = float(segment_dispatch)
    residual_pu = master.canonical.max_row_violation(source_values) / float(case.base_mva)
    if residual_pu > float(config.model["model_residual_tolerance_pu"]):
        raise RuntimeError(f"GPU checkpoint no longer satisfies the reduced model: {residual_pu}")

    if args.projected_cost:
        probe_started = time.perf_counter()
        result = _solve_fixed_commitment_cost_projection(
            region_id="v7_projected_cost_probe",
            commitment=commitment,
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(45.0, 5.0, 1.0),
            prepared_master=master,
            initial_source_values=source_values,
            initial_pairs=pairs,
            screener=ContingencyScreener(
                network,
                catalog,
                backend="cupy",
                chunk_columns=int(config.model["screen_chunk_columns"]),
            ),
            checkpoint=lambda: None,
            progress=None,
            policy=PrimalCandidatePolicy(
                total_seconds=30.0,
                maximum_round_seconds=float(args.slice_seconds),
                minimum_round_seconds=1.0,
                stagnation_window_rounds=2,
                minimum_relative_residual_improvement=0.01,
                dual_divergence_multiple=1e6,
                cold_restart_attempts=0,
            ),
        )
        print(
            "V7_PROJECTED_COST_PROBE="
            + json.dumps(
                {
                    "status": result.projected_solve.status,
                    "wall_time_seconds": time.perf_counter() - probe_started,
                    "source_objective": float(
                        np.asarray(result.master.canonical.objective)
                        @ result.source_values
                    ),
                    "source_residual_pu": (
                        result.master.canonical.max_row_violation(result.source_values)
                        / float(case.base_mva)
                    ),
                    "projection": result.rounds[-1]["projection"],
                    "solve_attempts": result.rounds[-1]["solve_attempts"],
                    "final_screen": result.final_screen,
                    "gpu_evidence_only": True,
                    "cpu_solution_data_read": False,
                    "diagnostic_only": True,
                },
                sort_keys=True,
            )
        )
        return

    profile = config.raw["platforms"]["dgx_spark"]
    column_scale, _ = native_scaling_vectors(
        master.canonical,
        mode=str(profile["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    common = {
        "time_limit_seconds": float(args.slice_seconds),
        "optimality_tolerance": float(profile["pdlp_optimality_tolerance"]),
        "primal_feasibility_tolerance": float(
            config.model["model_residual_tolerance_pu"]
        ),
        "certificate_residual_tolerance": float(
            profile["dual_certificate_residual_tolerance"]
        ),
        "native_scaling_mode": str(profile["native_scaling_mode"]),
        "native_base_mva": float(case.base_mva),
        "log_to_console": True,
        "per_constraint_residual": bool(profile["per_constraint_residual"]),
        "presolve": int(profile["presolve"]),
        "pdlp_solver_mode": int(profile["pdlp_solver_mode_native"]),
    }
    first = solve_cuopt_continuous_pdlp(
        master.canonical,
        **common,
        initial_native_primal=source_values / column_scale,
    )
    if first.native_primal is None or first.native_row_dual is None:
        raise RuntimeError("First bounded PDLP slice returned no raw continuation vectors")
    if first.pdlp_warm_start_data is None:
        second = solve_cuopt_continuous_pdlp(
            master.canonical,
            **common,
            initial_native_primal=first.native_primal,
            initial_native_row_dual=first.native_row_dual,
        )
        continuation = "raw_primal_dual_after_full_state_audit_rejection"
    else:
        second = solve_cuopt_continuous_pdlp(
            master.canonical,
            **common,
            initial_pdlp_warm_start_data=first.pdlp_warm_start_data,
        )
        continuation = "complete_full_stable2_state"
    if second.native_primal is None or second.native_row_dual is None:
        raise RuntimeError("Second bounded PDLP slice returned no continuation vectors")
    print(
        "V7_COST_CONTINUATION_PROBE="
        + json.dumps(
            {
                "continuation": continuation,
                "first_status": first.status,
                "first_residual_pu": (
                    master.canonical.max_row_violation(first.values) / float(case.base_mva)
                    if first.values is not None
                    else None
                ),
                "first_returned_state_audit": first.statistics[
                    "returned_pdlp_warm_start_state"
                ],
                "second_status": second.status,
                "second_residual_pu": (
                    master.canonical.max_row_violation(second.values) / float(case.base_mva)
                    if second.values is not None
                    else None
                ),
                "second_warm_start": second.statistics["warm_start"],
                "gpu_evidence_only": True,
                "cpu_solution_data_read": False,
                "diagnostic_only": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
