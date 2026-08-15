"""Tiny DGX smoke test for cuOpt incumbent commitment callbacks."""

from __future__ import annotations

import json

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.solvers import solve_canonical


def main() -> None:
    model = CanonicalMILP()
    first = model.add_variable(
        "u_g0000", objective=1.0, lower=0.0, upper=1.0, integer=True
    )
    second = model.add_variable(
        "u_g0001", objective=2.0, lower=0.0, upper=1.0, integer=True
    )
    model.add_row(
        "at_least_one_unit", {first: 1.0, second: 1.0}, lower=1.0
    )
    result = solve_canonical(
        model,
        solver="cuopt",
        time_limit_seconds=5.0,
        mip_relative_gap=1e-6,
        track_incumbent_commitments=True,
    )
    trace = result.statistics["incumbent_commitment_trace"]
    if not result.has_incumbent:
        raise RuntimeError("Tiny cuOpt callback diagnostic has no incumbent")
    if trace["callback_count"] < 1 or not trace["complete"]:
        raise RuntimeError(f"Tiny cuOpt callback trace failed: {trace}")
    print(json.dumps(trace, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
