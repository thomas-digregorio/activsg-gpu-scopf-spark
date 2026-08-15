"""One-time cuOpt full-MIP-start translation smoke test for the Spark image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.solvers import solve_canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    model = CanonicalMILP()
    first = model.add_variable(
        "u_g0000", objective=1.0, lower=0.0, upper=1.0, integer=True
    )
    second = model.add_variable(
        "u_g0001", objective=2.0, lower=0.0, upper=1.0, integer=True
    )
    free = model.add_variable("free_state", lower=-np.inf, upper=np.inf)
    model.add_variable("fixed_state", lower=3.0, upper=3.0)
    model.add_variable("unused_binary", lower=0.0, upper=1.0, integer=True)
    model.add_row(
        "at_least_one_unit", {first: 1.0, second: 1.0}, lower=1.0
    )
    model.add_row(
        "free_state_link", {first: 1.0, free: 1.0}, lower=1.0, upper=1.0
    )
    canonical_start = np.asarray([1.0, 0.0, 0.0, 3.0, 0.0])
    result = solve_canonical(
        model,
        solver="cuopt",
        time_limit_seconds=5.0,
        mip_relative_gap=1e-6,
        mip_start_values=canonical_start,
        mip_start_mode="all_columns",
    )
    contract = result.statistics["mip_start_native_contract"]
    native_log = result.statistics["native_log_audit"]
    if not result.has_incumbent:
        raise RuntimeError("MIP-start smoke test returned no incumbent")
    if contract["contract_passed"] is not True:
        raise RuntimeError(f"MIP-start readback failed: {contract}")
    if native_log["mip_start_rejection_count"] != 0:
        raise RuntimeError(f"cuOpt rejected the MIP start: {native_log}")

    payload = {
        "schema_version": "1.0.0",
        "status": "passed",
        "solver_status": result.status,
        "objective": result.objective,
        "canonical_start": canonical_start.tolist(),
        "returned_values": result.values.tolist() if result.values is not None else None,
        "canonical_columns": result.statistics["canonical_columns_translated"],
        "native_columns": result.statistics["native_columns_translated"],
        "free_columns_split": result.statistics[
            "native_free_variable_split_columns"
        ],
        "fixed_or_unused_columns_eliminated": result.statistics[
            "native_fixed_or_unused_columns_eliminated"
        ],
        "mip_start_native_contract": contract,
        "native_log_audit": native_log,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
