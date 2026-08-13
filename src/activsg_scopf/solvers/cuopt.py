"""NVIDIA cuOpt adapter using its supported Python expression API."""

from __future__ import annotations

from math import isfinite

import numpy as np

from ..canonical import CanonicalMILP
from ..errors import ScopfError
from .common import SolveResult


def _native(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def solve_cuopt(
    model: CanonicalMILP,
    *,
    time_limit_seconds: float,
    mip_relative_gap: float,
    threads: int = 0,
    mip_start_values: np.ndarray | None = None,
) -> SolveResult:
    try:
        import cuopt
        from cuopt.linear_programming.problem import (
            CONTINUOUS,
            INTEGER,
            MINIMIZE,
            LinearExpression,
            Problem,
        )
        from cuopt.linear_programming.solver_settings import SolverSettings
    except ImportError as exc:
        raise ScopfError("The cuOpt adapter requires NVIDIA cuOpt in the Spark runtime") from exc

    objective, lower, upper, integrality = model.column_arrays()
    problem = Problem("activsg_preventive_scuc")
    variables = [
        problem.addVariable(
            lb=float(lower[index]),
            ub=float(upper[index]),
            obj=float(objective[index]),
            vtype=INTEGER if integrality[index] else CONTINUOUS,
            name=name,
        )
        for index, name in enumerate(model.variable_names)
    ]
    mip_start_columns = np.empty(0, dtype=np.int64)
    if mip_start_values is not None:
        candidate = np.asarray(mip_start_values, dtype=np.float64)
        if candidate.shape != objective.shape:
            raise ScopfError(
                "cuOpt partial MIP start has the wrong vector shape: "
                f"expected {objective.shape}, observed {candidate.shape}"
            )
        mip_start_columns = np.flatnonzero(integrality)
        for index in mip_start_columns:
            value = float(np.rint(candidate[index]))
            if not np.isfinite(value):
                raise ScopfError("cuOpt partial MIP start contains a nonfinite value")
            variables[int(index)].setMIPStart(value)
    objective_columns = np.flatnonzero(objective)
    objective_expression = LinearExpression(
        [variables[int(index)] for index in objective_columns],
        objective[objective_columns].tolist(),
        0.0,
    )
    problem.setObjective(objective_expression, sense=MINIMIZE)
    row_lower, row_upper = model.row_bound_arrays()
    for row, name in enumerate(model.row_names):
        indices, coefficients = model.row_entries(row)
        expression = LinearExpression(
            [variables[index] for index in indices], coefficients, 0.0
        )
        lo = float(row_lower[row])
        hi = float(row_upper[row])
        if isfinite(lo) and isfinite(hi) and lo == hi:
            problem.addConstraint(expression == lo, name=name)
        else:
            if isfinite(lo):
                problem.addConstraint(expression >= lo, name=f"{name}__lower")
            if isfinite(hi):
                problem.addConstraint(expression <= hi, name=f"{name}__upper")
    settings = SolverSettings()
    settings.set_parameter("time_limit", float(time_limit_seconds))
    settings.set_parameter("mip_relative_gap", float(mip_relative_gap))
    settings.set_parameter("random_seed", 0)
    settings.set_parameter("log_to_console", False)
    if threads > 0:
        settings.set_parameter("num_cpu_threads", int(threads))
    problem.solve(settings)
    status = problem.Status.name
    stats = problem.SolutionStats
    has_incumbent = (
        status in {"Optimal", "FeasibleFound", "TimeLimit"}
        and np.isfinite(problem.ObjValue)
        and all(np.isfinite(variable.getValue()) for variable in variables)
    )
    values = (
        np.asarray([variable.getValue() for variable in variables], dtype=np.float64)
        if has_incumbent
        else None
    )
    objective_value = float(problem.ObjValue) if has_incumbent else None
    raw_bound = getattr(stats, "solution_bound", None)
    raw_gap = getattr(stats, "mip_gap", None)
    reported_gap = (
        float(raw_gap) if raw_gap is not None and np.isfinite(raw_gap) else None
    )
    meets_requested_gap = (
        reported_gap is not None
        and reported_gap <= float(mip_relative_gap) * (1.0 + 1e-9) + 1e-12
    )
    accepted_optimal = status == "Optimal" and meets_requested_gap
    reported_status = (
        "OptimalGapMismatch" if status == "Optimal" and not meets_requested_gap else status
    )
    return SolveResult(
        solver="cuopt",
        solver_version=str(getattr(cuopt, "__version__", "unknown")),
        status=reported_status,
        optimal=accepted_optimal,
        has_incumbent=has_incumbent,
        objective=objective_value,
        bound=float(raw_bound) if raw_bound is not None and np.isfinite(raw_bound) else None,
        mip_gap=reported_gap,
        solve_time_seconds=float(problem.SolveTime),
        values=values,
        statistics={
            **{
                key: value
                for key in (
                    "max_constraint_violation",
                    "max_int_violation",
                    "max_variable_bound_violation",
                    "num_nodes",
                    "num_simplex_iterations",
                    "presolve_time",
                )
                if (value := _native(getattr(stats, key, None))) is not None
            },
            "native_status": status,
            "requested_mip_relative_gap": float(mip_relative_gap),
            "reported_gap_meets_request": meets_requested_gap,
            "partial_integer_mip_start_columns": int(mip_start_columns.size),
        },
    )
