#!/usr/bin/env python3
"""Exercise the exact saved ACTIVSg2000 round-2 master with its prior GPU start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from activsg_scopf.config import load_config
from activsg_scopf.errors import ProvenanceError
from activsg_scopf.matpower import read_contingency_table, read_matpower_case
from activsg_scopf.model import build_master
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.screening import add_security_pairs
from activsg_scopf.seeded_diagnostic import (
    canonical_feasibility_audit,
    deserialize_solution_values,
    security_pairs_from_ids,
)
from activsg_scopf.solvers.common import (
    HIGHS_FIXED_COMMITMENT_PRECHECK,
    RebuildingSolverSession,
)
from activsg_scopf.solvers.cuopt import fixed_or_unused_columns

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v8.json"
EXPECTED_BENCHMARK_ID = "activsg2000-gpu-gap-v8-1e-3"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProvenanceError(f"Expected a JSON object in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--time-limit-seconds", type=float, default=15.0)
    arguments = parser.parse_args()

    if arguments.time_limit_seconds <= 0.0 or arguments.time_limit_seconds > 60.0:
        raise ProvenanceError("Diagnostic solve time must be in (0, 60] seconds")
    config = load_config(arguments.config)
    checkpoint = _read_json(arguments.checkpoint)
    if checkpoint.get("benchmark_id") != EXPECTED_BENCHMARK_ID:
        raise ProvenanceError("Checkpoint is not the registered failed v8 run")
    rounds = checkpoint.get("constraint_generation_rounds")
    if not isinstance(rounds, list) or len(rounds) != 1:
        raise ProvenanceError("Expected exactly one completed v8 constraint-generation round")
    pair_ids = list(checkpoint.get("added_security_pair_ids", []))
    if not pair_ids or pair_ids != sorted(pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ProvenanceError("Saved v8 security-pair IDs are absent or noncanonical")
    if rounds[0].get("added_pair_ids") != pair_ids:
        raise ProvenanceError("Checkpoint round-1 additions do not match the saved pair set")

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
    master = build_master(case, network, segments=int(config.model["pwl_segments"]))
    pairs = security_pairs_from_ids(pair_ids, network, catalog)
    add_security_pairs(master.canonical, master.index, network, pairs)
    prior_values = deserialize_solution_values(
        checkpoint["solution"], case, network, master
    )
    _, column_lower, column_upper, _ = master.canonical.column_arrays()
    row_lower, row_upper = master.canonical.row_bound_arrays()
    equality_rows = int(
        np.count_nonzero(np.isfinite(row_lower) & (row_lower == row_upper))
    )
    expected_free_columns = int(
        np.count_nonzero(np.isneginf(column_lower) & np.isposinf(column_upper))
    )
    expected_eliminated_columns = int(
        fixed_or_unused_columns(master.canonical).size
    )
    expected_native_columns = (
        master.canonical.num_columns
        + expected_free_columns
        - expected_eliminated_columns
    )
    profile = config.raw["platforms"]["dgx_spark"]
    session = RebuildingSolverSession(
        model=master.canonical,
        solver="cuopt",
        mip_relative_gap=float(config.model["mip_relative_gap_tolerance"]),
        threads=int(profile["solver_threads"]),
        native_scaling_mode=str(profile["native_scaling_mode"]),
        native_base_mva=float(case.base_mva),
        log_to_console=True,
        cuopt_pdlp_profile=dict(profile["cuopt_pdlp_profile"]),
        mip_acceptance_policy=str(profile["mip_acceptance_policy"]),
        mip_certificate_residual_tolerance=float(
            profile["mip_certificate_residual_tolerance"]
        ),
        mip_start_precheck=HIGHS_FIXED_COMMITMENT_PRECHECK,
        mip_start_precheck_time_limit_seconds=10.0,
    )
    session.previous_values = prior_values.copy()
    result = session.solve(
        time_limit_seconds=float(arguments.time_limit_seconds)
    )
    statistics = result.statistics
    start_contract = statistics["mip_start_native_contract"]
    native_log_audit = statistics["native_log_audit"]
    precheck = statistics["mip_start_feasibility_precheck"]
    passed = bool(
        precheck.get("prior_commitment_extendable") is False
        and precheck.get("decision") == "solve_cold"
        and start_contract.get("submitted") is False
        and native_log_audit.get("mip_start_rejection_count") == 0
    )
    payload = {
        "schema_version": "1.0.0",
        "diagnostic": "exact_v8_round2_prior_gpu_integer_start",
        "checkpoint": str(arguments.checkpoint),
        "canonical_dimensions": {
            "columns": master.canonical.num_columns,
            "rows": master.canonical.num_rows,
            "nonzeros": int(master.canonical.matrix_csr().nnz),
            "equality_rows": equality_rows,
            "free_columns_requiring_native_split": expected_free_columns,
            "fixed_or_unused_columns_requiring_native_elimination": (
                expected_eliminated_columns
            ),
            "native_columns_expected_if_start_submitted": (
                expected_native_columns
            ),
        },
        "prior_round_solution_on_round2_master": canonical_feasibility_audit(
            master.canonical,
            prior_values,
            expected_objective=float(checkpoint["objective"]),
        ),
        "solve": {
            "status": result.status,
            "objective": result.objective,
            "bound": result.bound,
            "mip_gap": result.mip_gap,
            "solve_time_seconds": result.solve_time_seconds,
            "has_incumbent": result.has_incumbent,
            "mip_start_feasibility_precheck": precheck,
            "mip_start_native_contract": start_contract,
            "native_log_audit": native_log_audit,
            "native_columns_translated": statistics[
                "native_columns_translated"
            ],
            "native_free_variable_split_columns": statistics[
                "native_free_variable_split_columns"
            ],
            "native_fixed_or_unused_columns_eliminated": statistics[
                "native_fixed_or_unused_columns_eliminated"
            ],
            "native_explicit_free_variable_split": statistics[
                "native_explicit_free_variable_split"
            ],
        },
        "passed": passed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("Exact round-2 prior-GPU MIP-start diagnostic failed")


if __name__ == "__main__":
    main()
