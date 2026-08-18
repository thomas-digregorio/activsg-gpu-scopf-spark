#!/usr/bin/env python3
"""Read-only DGX probe for cuOpt's GPU-only MIP-heuristics setting."""

from __future__ import annotations

import inspect
import json

from cuopt.linear_programming.problem import INTEGER, MINIMIZE, Problem
from cuopt.linear_programming.solver_settings import SolverSettings


def main() -> None:
    settings = SolverSettings()
    default = settings.get_parameter("mip_heuristics_only")
    settings.set_parameter("mip_heuristics_only", True)
    readback = settings.get_parameter("mip_heuristics_only")
    settings.set_parameter("time_limit", 1.0)
    settings.set_parameter("log_to_console", True)

    problem = Problem("gpu_only_mip_heuristics_probe")
    x = problem.addVariable(lb=0.0, ub=1.0, vtype=INTEGER, name="x")
    y = problem.addVariable(lb=0.0, ub=1.0, vtype=INTEGER, name="y")
    problem.addConstraint(x + y >= 1.0, name="cover")
    problem.setObjective(x + 2.0 * y, sense=MINIMIZE)
    problem.solve(settings)
    print(
        json.dumps(
            {
                "default": bool(default),
                "requested": True,
                "readback": bool(readback),
                "status": problem.Status.name,
                "objective": float(problem.ObjValue),
                "x": float(x.getValue()),
                "y": float(y.getValue()),
                "solve_seconds": float(problem.SolveTime),
                "problem_add_mip_start_available": hasattr(Problem, "addMIPStart"),
                "problem_add_mip_start_signature": (
                    str(inspect.signature(Problem.addMIPStart))
                    if hasattr(Problem, "addMIPStart")
                    else None
                ),
                "variable_set_mip_start_signature": str(
                    inspect.signature(type(x).setMIPStart)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
