#!/usr/bin/env python3
"""One bounded DGX component probe of cuOpt GPU-only MIP heuristics.

The probe reuses only the GPU-generated security-pair identities from the
preserved v5 result.  It never reads a CPU solution, objective, or bound, and
its output is diagnostic rather than benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from activsg_scopf.config import load_config
from activsg_scopf.matpower import read_contingency_table, read_matpower_case
from activsg_scopf.model import build_master
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    commitment_vector,
    reduced_dispatch,
    security_pair_from_record,
)
from activsg_scopf.screening import ContingencyScreener, add_security_pairs
from activsg_scopf.solvers.cuopt import solve_cuopt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/configs/activsg2000-gpu-lagrangian-v5.json"),
    )
    parser.add_argument(
        "--gpu-evidence",
        type=Path,
        default=Path(
            "/workspace/results/experiments/"
            "activsg2000-gpu-lagrangian-v5-dgx-spark.json"
        ),
    )
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--formulation", choices=("reduced", "full"), default="full")
    parser.add_argument("--gpu-root-rounding-start", action="store_true")
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
    root = next(
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
        for record in root["security_pairs"]
    )
    if args.formulation == "reduced":
        master = build_reduced_master(
            case,
            network,
            segments=int(config.model["pwl_segments"]),
            coefficient_zero_tolerance=float(
                config.model["reduced_coefficient_zero_tolerance"]
            ),
        )
        add_reduced_security_pairs(master, network, pairs)
    else:
        master = build_master(case, network, segments=int(config.model["pwl_segments"]))
        add_security_pairs(master.canonical, master.index, network, pairs)
    profile = config.raw["platforms"]["dgx_spark"]
    mip_start_values = None
    if args.gpu_root_rounding_start:
        candidate = next(
            record
            for record in evidence["primal_candidate_queue"]
            if record["origin"] == "root_pdlp_rounding"
        )
        committed_rows = {
            int(row) for row in candidate["committed_generator_source_rows"]
        }
        mip_start_values = np.zeros(master.canonical.num_columns, dtype=np.float64)
        for generator in master.index.generator_source_rows:
            source_row = int(generator) + 1
            mip_start_values[
                master.index.commitment_by_generator[int(generator)]
            ] = float(source_row in committed_rows)
    result = solve_cuopt(
        master.canonical,
        time_limit_seconds=float(args.seconds),
        mip_relative_gap=float(config.model["mip_relative_gap_tolerance"]),
        native_scaling_mode=str(profile["native_scaling_mode"]),
        native_base_mva=float(case.base_mva),
        log_to_console=True,
        track_incumbent_commitments=True,
        mip_heuristics_only=True,
        mip_start_values=mip_start_values,
    )
    trace = result.statistics["incumbent_commitment_trace"]
    payload: dict[str, object] = {
        "status": result.status,
        "has_incumbent": bool(result.has_incumbent),
        "objective": result.objective,
        "native_solve_seconds": result.solve_time_seconds,
        "security_pairs_before_solve": len(pairs),
        "mip_heuristics_only_requested": result.statistics[
            "mip_heuristics_only_requested"
        ],
        "mip_heuristics_only_readback": result.statistics[
            "mip_heuristics_only_readback"
        ],
        "native_log_audit": result.statistics["native_log_audit"],
        "incumbent_trace_summary": {
            key: trace.get(key)
            for key in (
                "enabled",
                "callback_count",
                "commitment_transition_count",
                "unique_commitment_count",
                "last_commitment_change_elapsed_seconds",
                "stabilization_window_seconds_at_solver_return",
                "final_commitment_fingerprint_sha256",
                "complete",
            )
        },
        "cpu_solution_data_read": False,
        "diagnostic_only": True,
        "formulation": args.formulation,
        "gpu_root_rounding_start_submitted": bool(args.gpu_root_rounding_start),
        "mip_start_contract": result.statistics["mip_start_native_contract"],
    }
    if result.values is not None:
        if args.formulation == "reduced":
            commitments = commitment_vector(master, result.values)
            dispatch = reduced_dispatch(master, result.values)
            flow = master.operator.flows(dispatch)
        else:
            commitments = np.asarray(
                [
                    result.values[
                        master.index.commitment_by_generator[int(generator)]
                    ]
                    for generator in master.index.generator_source_rows
                ],
                dtype=np.float64,
            )
            flow = np.asarray(
                result.values[master.index.flow_by_active_branch], dtype=np.float64
            )
        screened = ContingencyScreener(
            network,
            catalog,
            backend="cupy",
            chunk_columns=int(config.model["screen_chunk_columns"]),
        ).screen(
            flow,
            tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
            already_added={pair.pair_id for pair in pairs},
        )
        payload.update(
            {
                "commitment_count": int(np.rint(commitments).sum()),
                "maximum_integrality_residual": float(
                    np.max(np.abs(commitments - np.rint(commitments)))
                ),
                "canonical_maximum_residual_pu": float(
                    master.canonical.max_row_violation(result.values) / case.base_mva
                ),
                "new_security_pairs": len(screened.violations),
                "maximum_exhaustive_security_violation_pu": (
                    screened.maximum_violation_pu
                ),
            }
        )
    print("GPU_HEURISTICS_PRIMAL_PROBE=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
