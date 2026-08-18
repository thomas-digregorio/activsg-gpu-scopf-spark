#!/usr/bin/env python3
"""Read-only DGX probe for cuOpt 26.6 PDLP warm-start-context support."""

from __future__ import annotations

import json

import numpy as np
from cuopt import linear_programming
from cuopt.linear_programming.problem import CONTINUOUS, MINIMIZE, Problem
from cuopt.linear_programming.solver_settings import (
    PDLPSolverMode,
    SolverMethod,
    SolverSettings,
)


def build_problem(rhs: float) -> Problem:
    problem = Problem("activsg_pdlp_warmstart_probe")
    x = problem.addVariable(lb=0.0, ub=10.0, vtype=CONTINUOUS, name="x")
    y = problem.addVariable(lb=0.0, ub=10.0, vtype=CONTINUOUS, name="y")
    problem.addConstraint(x + y >= rhs, name="balance")
    problem.setObjective(x + 2.0 * y, sense=MINIMIZE)
    return problem


def warmstart_payload_summary(warm: object) -> dict[str, object]:
    """Describe every public PDLP state field without assuming it is populated."""

    summary: dict[str, object] = {}
    for name in sorted(item for item in dir(warm) if not item.startswith("_")):
        value = getattr(warm, name)
        if value is None:
            summary[name] = {"is_none": True}
            continue
        array = np.asarray(value)
        summary[name] = {
            "is_none": False,
            "shape": list(array.shape),
            "finite": bool(np.all(np.isfinite(array))),
            "type": type(value).__name__,
        }
    return summary


def solve_with_warmstart(warm: object, rhs: float) -> dict[str, object]:
    settings = SolverSettings()
    settings.set_parameter("method", SolverMethod.PDLP)
    settings.set_parameter("pdlp_solver_mode", PDLPSolverMode.Stable2)
    settings.set_parameter("presolve", 0)
    settings.set_parameter("log_to_console", True)
    settings.set_pdlp_warm_start_data(warm)
    problem = build_problem(rhs)
    problem.solve(settings)
    return {
        "rhs": rhs,
        "status": problem.Status.name,
        "solve_seconds": problem.SolveTime,
        "public_attributes": sorted(
            name for name in dir(problem) if not name.startswith("_")
        ),
    }


def main() -> None:
    settings = SolverSettings()
    settings.set_parameter("method", SolverMethod.PDLP)
    settings.set_parameter("pdlp_solver_mode", PDLPSolverMode.Stable2)
    settings.set_parameter("presolve", 0)
    settings.set_parameter("log_to_console", False)
    first = build_problem(5.0)
    first.solve(settings)
    warm = first.getWarmstartData()
    low_level_problem = build_problem(5.0)
    low_level_problem._to_data_model()
    low_level_solution = linear_programming.Solve(low_level_problem.model, settings)
    print(
        json.dumps(
            {
                "first_status": first.Status.name,
                "first_solve_seconds": first.SolveTime,
                "warmstart_solves": [
                    solve_with_warmstart(warm, 5.0),
                    solve_with_warmstart(warm, 6.0),
                ],
                "warmstart_type": type(warm).__name__,
                "low_level_solution_type": type(low_level_solution).__name__,
                "low_level_solution_attributes": sorted(
                    name for name in dir(low_level_solution) if not name.startswith("_")
                ),
                "warmstart_attributes": sorted(
                    name for name in dir(warm) if not name.startswith("_")
                ),
                "warmstart_payload": warmstart_payload_summary(warm),
                "primal_count": len(warm.current_primal_solution),
                "dual_count": len(warm.current_dual_solution),
                "solver_modes": {
                    name: int(value.value) for name, value in PDLPSolverMode.__members__.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
