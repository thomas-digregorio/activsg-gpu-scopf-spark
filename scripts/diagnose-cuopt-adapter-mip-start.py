"""Run a two-round tiny cuOpt adapter probe with a prior integer start."""

from __future__ import annotations

import json

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.solvers.cuopt import (
    FINITE_BOUND_GAP_CERTIFICATE,
    solve_cuopt,
)


def main() -> None:
    model = CanonicalMILP()
    cheap = model.add_variable(
        "u_cheap", objective=10.0, lower=0.0, upper=1.0, integer=True
    )
    expensive = model.add_variable(
        "u_expensive", objective=20.0, lower=0.0, upper=1.0, integer=True
    )
    dispatch = model.add_variable(
        "pg", objective=1.0, lower=0.0, upper=6.0
    )
    angle = model.add_variable("theta", lower=-float("inf"), upper=float("inf"))
    model.add_variable("unused_commitment", lower=0.0, upper=1.0, integer=True)
    model.add_variable(
        "fixed_offset", objective=3.0, lower=2.0, upper=2.0
    )
    model.add_row("balance", {dispatch: 1.0}, lower=4.0, upper=4.0)
    model.add_row(
        "angle_link", {angle: 1.0, dispatch: -0.1}, lower=0.0, upper=0.0
    )
    model.add_row(
        "base_capacity",
        {dispatch: 1.0, cheap: -4.0, expensive: -6.0},
        upper=0.0,
    )
    common = {
        "time_limit_seconds": 10.0,
        "mip_relative_gap": 0.0,
        "log_to_console": True,
        "mip_acceptance_policy": FINITE_BOUND_GAP_CERTIFICATE,
    }
    first = solve_cuopt(model, **common)
    if first.values is None:
        raise RuntimeError("Tiny round 1 did not return an incumbent")
    model.add_row(
        "security_capacity",
        {dispatch: 1.0, cheap: -3.0, expensive: -6.0},
        upper=0.0,
    )
    second = solve_cuopt(model, mip_start_values=first.values, **common)
    if second.statistics["native_free_variable_split_columns"] != 1:
        raise RuntimeError("Tiny adapter probe did not explicitly split its free column")
    if second.statistics["native_fixed_or_unused_columns_eliminated"] != 2:
        raise RuntimeError("Tiny adapter probe did not explicitly eliminate two columns")
    if not second.statistics["mip_start_native_contract"].get("contract_passed"):
        raise RuntimeError("Tiny adapter probe did not pass the native MIP-start contract")
    print(
        json.dumps(
            {
                "round_1": {
                    "status": first.status,
                    "objective": first.objective,
                    "values": first.values.tolist(),
                },
                "round_2": {
                    "status": second.status,
                    "objective": second.objective,
                    "values": None if second.values is None else second.values.tolist(),
                    "mip_start_native_contract": second.statistics[
                        "mip_start_native_contract"
                    ],
                    "native_log_audit": second.statistics["native_log_audit"],
                    "native_free_variable_split_columns": second.statistics[
                        "native_free_variable_split_columns"
                    ],
                    "native_fixed_or_unused_columns_eliminated": second.statistics[
                        "native_fixed_or_unused_columns_eliminated"
                    ],
                    "native_columns_translated": second.statistics[
                        "native_columns_translated"
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
