#!/usr/bin/env python3
"""Exercise the exact v6 cuOpt adapter contracts on tiny GPU problems."""

from __future__ import annotations

import json

import numpy as np

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.errors import ScopfError
from activsg_scopf.solvers.cuopt import (
    FULL_MIP_START,
    INTEGER_ONLY_MIP_START,
    solve_cuopt,
)
from activsg_scopf.solvers.cuopt_lp import solve_cuopt_continuous_pdlp


def continuous_probe() -> dict[str, object]:
    model = CanonicalMILP()
    x = model.add_variable("x", objective=1.0, lower=0.0, upper=10.0)
    y = model.add_variable("y", objective=2.0, lower=0.0, upper=10.0)
    model.add_row("balance", {x: 1.0, y: 1.0}, lower=5.0)
    arguments = {
        "time_limit_seconds": 2.0,
        "optimality_tolerance": 1e-8,
        "primal_feasibility_tolerance": 1e-6,
        "certificate_residual_tolerance": 1e-7,
        "native_scaling_mode": "none",
        "native_base_mva": 100.0,
        "log_to_console": True,
        "presolve": 0,
        "pdlp_solver_mode": 1,
        "save_best_primal_solution": True,
    }
    first = solve_cuopt_continuous_pdlp(model, **arguments)
    if first.pdlp_warm_start_data is None:
        raise RuntimeError("Stable2 adapter did not return full PDLP state")
    second = solve_cuopt_continuous_pdlp(
        model,
        **arguments,
        initial_pdlp_warm_start_data=first.pdlp_warm_start_data,
    )
    return {
        "first_status": first.status,
        "second_status": second.status,
        "first_state_returned": first.pdlp_warm_start_data is not None,
        "second_state_returned": second.pdlp_warm_start_data is not None,
        "second_full_state_submitted": second.statistics["warm_start"][
            "full_pdlp_state_submitted"
        ],
        "second_primal_objective": second.primal_objective,
        "second_certificate_passed": second.statistics["dual_certificate"][
            "passed"
        ],
    }


def mip_probe() -> dict[str, object]:
    model = CanonicalMILP()
    commitment = model.add_variable(
        "u_g0001", objective=1.0, lower=0.0, upper=1.0, integer=True
    )
    dispatch = model.add_variable("pg_g0001", objective=1.0, lower=0.0, upper=10.0)
    model.add_row("dispatch_on", {dispatch: 1.0, commitment: -10.0}, upper=0.0)
    model.add_row("demand", {dispatch: 1.0}, lower=1.0)
    cold = solve_cuopt(
        model,
        time_limit_seconds=2.0,
        mip_relative_gap=1e-3,
        log_to_console=True,
        track_incumbent_commitments=True,
        mip_heuristics_only=True,
    )
    full = solve_cuopt(
        model,
        time_limit_seconds=2.0,
        mip_relative_gap=1e-3,
        mip_start_values=np.asarray([1.0, 1.0]),
        mip_start_mode=FULL_MIP_START,
        log_to_console=True,
        mip_heuristics_only=True,
    )
    partial_rejection = None
    try:
        solve_cuopt(
            model,
            time_limit_seconds=2.0,
            mip_relative_gap=1e-3,
            mip_start_values=np.asarray([1.0, 1.0]),
            mip_start_mode=INTEGER_ONLY_MIP_START,
            mip_heuristics_only=True,
        )
    except ScopfError as exc:
        partial_rejection = str(exc)
    if partial_rejection is None:
        raise RuntimeError("Partial cuOpt MIP start was not rejected before translation")
    return {
        "cold_status": cold.status,
        "cold_has_incumbent": cold.has_incumbent,
        "heuristics_only_readback": cold.statistics[
            "mip_heuristics_only_readback"
        ],
        "full_start_status": full.status,
        "full_start_has_incumbent": full.has_incumbent,
        "full_start_contract": full.statistics["mip_start_native_contract"],
        "partial_start_rejection": partial_rejection,
    }


def main() -> None:
    print(
        json.dumps(
            {
                "continuous": continuous_probe(),
                "mip": mip_probe(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
