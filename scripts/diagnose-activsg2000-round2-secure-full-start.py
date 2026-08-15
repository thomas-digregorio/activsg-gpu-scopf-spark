#!/usr/bin/env python3
"""Validate one known-secure full start on the exact saved v8 round-2 master."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from activsg_scopf.config import load_config
from activsg_scopf.errors import ProvenanceError
from activsg_scopf.matpower import (
    read_contingency_table,
    read_matpower_case,
    sha256_file,
)
from activsg_scopf.model import build_master
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.screening import add_security_pairs
from activsg_scopf.seeded_diagnostic import (
    canonical_feasibility_audit,
    deserialize_solution_values,
    security_pairs_from_ids,
)
from activsg_scopf.solvers.cuopt import (
    FULL_MIP_START,
    fixed_or_unused_columns,
    solve_cuopt,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v8.json"
SECURE_RESULT_SHA256 = (
    "5573425a8e625c0c964b2c61d33ca80666de74a350e9431a96e5ce9b90e02e3f"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProvenanceError(f"Expected a JSON object in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--secure-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--time-limit-seconds", type=float, default=15.0)
    arguments = parser.parse_args()
    if arguments.time_limit_seconds <= 0.0 or arguments.time_limit_seconds > 60.0:
        raise ProvenanceError("Diagnostic solve time must be in (0, 60] seconds")
    if sha256_file(arguments.secure_result) != SECURE_RESULT_SHA256:
        raise ProvenanceError("Secure CPU result hash changed")

    config = load_config(arguments.config)
    checkpoint = _read_json(arguments.checkpoint)
    secure = _read_json(arguments.secure_result)
    pair_ids = list(checkpoint.get("added_security_pair_ids", []))
    if (
        checkpoint.get("benchmark_id") != "activsg2000-gpu-gap-v8-1e-3"
        or len(pair_ids) != 173
        or pair_ids != sorted(pair_ids)
    ):
        raise ProvenanceError("Checkpoint is not the exact saved v8 round-2 boundary")

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
    values = deserialize_solution_values(secure["solution"], case, network, master)
    feasibility = canonical_feasibility_audit(
        master.canonical, values, expected_objective=float(secure["objective"])
    )
    if (
        feasibility["maximum_row_violation"] > 1e-6
        or feasibility["maximum_column_bound_violation"] > 1e-6
        or feasibility["maximum_integrality_violation"] > 1e-6
    ):
        raise ProvenanceError("Registered secure solution is not feasible on round 2")

    profile = config.raw["platforms"]["dgx_spark"]
    result = solve_cuopt(
        master.canonical,
        time_limit_seconds=float(arguments.time_limit_seconds),
        mip_relative_gap=float(config.model["mip_relative_gap_tolerance"]),
        threads=int(profile["solver_threads"]),
        mip_start_values=values,
        mip_start_mode=FULL_MIP_START,
        clip_mip_start_to_bounds=True,
        native_scaling_mode=str(profile["native_scaling_mode"]),
        native_base_mva=float(case.base_mva),
        log_to_console=True,
        cuopt_pdlp_profile=dict(profile["cuopt_pdlp_profile"]),
        mip_acceptance_policy=str(profile["mip_acceptance_policy"]),
        mip_certificate_residual_tolerance=float(
            profile["mip_certificate_residual_tolerance"]
        ),
    )
    statistics = result.statistics
    contract = statistics["mip_start_native_contract"]
    native_log = statistics["native_log_audit"]
    _, lower, upper, _ = master.canonical.column_arrays()
    free_columns = int(
        np.count_nonzero(np.isneginf(lower) & np.isposinf(upper))
    )
    eliminated_columns = int(fixed_or_unused_columns(master.canonical).size)
    expected_native_columns = (
        master.canonical.num_columns + free_columns - eliminated_columns
    )
    passed = bool(
        contract.get("contract_passed")
        and contract.get("native_log_contract_passed")
        and native_log.get("mip_start_rejection_count") == 0
        and statistics.get("native_columns_translated")
        == expected_native_columns
        and statistics.get("native_free_variable_split_columns") == free_columns
        and statistics.get("native_fixed_or_unused_columns_eliminated")
        == eliminated_columns
    )
    payload = {
        "schema_version": "1.0.0",
        "diagnostic": "exact_v8_round2_known_secure_full_start",
        "secure_result_sha256": SECURE_RESULT_SHA256,
        "canonical_feasibility": feasibility,
        "expected_native_columns": expected_native_columns,
        "solve": {
            "status": result.status,
            "objective": result.objective,
            "bound": result.bound,
            "mip_gap": result.mip_gap,
            "solve_time_seconds": result.solve_time_seconds,
            "has_incumbent": result.has_incumbent,
            "mip_start_native_contract": contract,
            "native_log_audit": native_log,
            "native_columns_translated": statistics["native_columns_translated"],
        },
        "passed": passed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("Exact round-2 secure full-start diagnostic failed")


if __name__ == "__main__":
    main()
