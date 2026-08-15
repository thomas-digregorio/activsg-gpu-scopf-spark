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
    model.add_row("balance", {dispatch: 1.0}, lower=4.0, upper=4.0)
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
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
