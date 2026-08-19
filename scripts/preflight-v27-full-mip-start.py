"""Exercise the production v27 polish on the preserved v26 GPU incumbent."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from activsg_scopf.config import load_config
from activsg_scopf.lagrangian_experiment import _polish_secure_gpu_full_mip_start
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--polish-seconds", type=float, default=20.0)
    args = parser.parse_args()

    config = load_config(args.config)
    result = json.loads(args.result.read_text(encoding="utf-8"))
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
    pairs = tuple(
        security_pair_from_record(
            record,
            catalog,
            lodf_absolute_tolerance=float(
                config.model["serialized_lodf_replay_tolerance"]
            ),
        )
        for record in result["shared_master_evidence"]["security_pairs"]
    )
    exact = build_reduced_master(
        case,
        network,
        segments=int(config.model["pwl_segments"]),
        coefficient_zero_tolerance=0.0,
        security_row_maximum_raw_violation_envelope_mw=None,
    )
    add_reduced_security_pairs(exact, network, pairs)
    records = result["secure_incumbent_checkpoint"]["solution"]["generators"]
    by_row = {int(record["source_row"]): record for record in records}
    source = np.zeros(exact.canonical.num_columns, dtype=np.float64)
    commitment = np.zeros(exact.index.generator_source_rows.size, dtype=np.int8)
    for position, generator in enumerate(exact.index.generator_source_rows):
        source_index = int(generator)
        record = by_row[source_index + 1]
        if case.gen[source_index, GEN_STATUS] <= 0:
            raise RuntimeError("Exact source master unexpectedly retained an offline generator")
        commitment[position] = int(round(float(record["commitment"])))
        source[exact.index.commitment_by_generator[source_index]] = commitment[position]
        source[exact.index.dispatch_by_generator[source_index]] = float(
            record["dispatch_mw"]
        )
        for column, value in zip(
            exact.index.segments_by_generator[source_index],
            record["segment_dispatch_mw"],
            strict=True,
        ):
            if column is not None:
                source[column] = float(value)
    screener = ContingencyScreener(
        network,
        catalog,
        backend="cupy",
        chunk_columns=int(config.model["screen_chunk_columns"]),
    )
    profile = config.raw["platforms"]["dgx_spark"]
    started = time.perf_counter()
    polished = _polish_secure_gpu_full_mip_start(
        case=case,
        network=network,
        config=config,
        commitment=commitment,
        source_master=exact,
        source_values=source,
        security_pairs=pairs,
        screener=screener,
        time_limit_seconds=float(args.polish_seconds),
    )
    full, full_values = reconstruct_full_values(
        case,
        network,
        polished.source_master,
        polished.source_values,
        exact_commitment=commitment,
    )
    add_security_pairs(full.canonical, full.index, network, pairs)
    redundant = derive_rate_a_angle_bounds(
        network,
        full.index.theta_by_bus,
        total_columns=full.canonical.num_columns,
    )
    _, lower, upper, _ = full.canonical.column_arrays()
    for column in np.flatnonzero(
        (redundant.lower > lower) | (redundant.upper < upper)
    ):
        position = int(column)
        full.canonical.column_lower[position] = max(
            float(lower[position]), float(redundant.lower[position])
        )
        full.canonical.column_upper[position] = min(
            float(upper[position]), float(redundant.upper[position])
        )
    full_column_scale, full_row_scale = native_scaling_vectors(
        full.canonical,
        mode=str(profile["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    contract = validate_full_mip_start_feasibility(
        full.canonical,
        full_values,
        column_scale=full_column_scale,
        row_scale=full_row_scale,
        tolerance=float(config.model["model_residual_tolerance_pu"]),
    )
    mip_started = time.perf_counter()
    mip = solve_cuopt(
        full.canonical,
        time_limit_seconds=1.0,
        mip_relative_gap=float(config.model["mip_relative_gap_tolerance"]),
        mip_start_values=full_values,
        mip_start_mode=FULL_MIP_START,
        native_scaling_mode=str(profile["native_scaling_mode"]),
        native_base_mva=float(case.base_mva),
        log_to_console=True,
        track_incumbent_commitments=True,
        mip_certificate_residual_tolerance=float(
            config.model["model_residual_tolerance_pu"]
        ),
        mip_heuristics_only=True,
    )
    print(
        "V27_FULL_MIP_START_PREFLIGHT="
        + json.dumps(
            {
                "passed": bool(
                    contract["passed"]
                    and mip.statistics["mip_start_native_contract"][
                        "contract_passed"
                    ]
                    and mip.statistics["native_log_audit"][
                        "mip_start_rejection_count"
                    ]
                    == 0
                ),
                "polish_wall_time_seconds": time.perf_counter() - started,
                "production_polish_audit": polished.audit,
                "full_start_contract": contract,
                "mip_status": mip.status,
                "mip_adapter_wall_time_seconds": time.perf_counter() - mip_started,
                "mip_native_solve_time_seconds": mip.solve_time_seconds,
                "mip_start_native_contract": mip.statistics[
                    "mip_start_native_contract"
                ],
                "native_log_audit": mip.statistics["native_log_audit"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
