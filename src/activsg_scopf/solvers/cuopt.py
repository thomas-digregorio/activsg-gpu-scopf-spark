"""NVIDIA cuOpt adapter using its supported Python expression API."""

from __future__ import annotations

from math import isfinite

import numpy as np

from ..canonical import CanonicalMILP
from ..errors import ScopfError
from .common import SolveResult

NATIVE_OPTIMAL_ONLY = "native_optimal_only"
FINITE_BOUND_GAP_CERTIFICATE = (
    "finite_incumbent_bound_gap_and_native_residuals_v1"
)
SUPPORTED_CERTIFICATE_STATUSES = frozenset(
    {"Optimal", "FeasibleFound", "TimeLimit"}
)


def _native(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def evaluate_mip_gap_certificate(
    *,
    native_status: str,
    objective: float | None,
    bound: float | None,
    reported_gap: float | None,
    requested_gap: float,
    native_residuals: dict[str, float | None],
    residual_tolerance: float,
) -> dict[str, object]:
    """Evaluate a fail-closed minimization gap certificate from native evidence."""

    if requested_gap < 0 or residual_tolerance < 0:
        raise ScopfError("MIP certificate tolerances must be nonnegative")
    finite_objective = objective is not None and np.isfinite(objective)
    finite_bound = bound is not None and np.isfinite(bound)
    finite_reported_gap = reported_gap is not None and np.isfinite(reported_gap)
    calculated_gap: float | None = None
    bound_is_valid_for_minimization = False
    if finite_objective and finite_bound:
        objective_value = float(objective)
        bound_value = float(bound)
        if objective_value == 0.0:
            calculated_gap = 0.0 if bound_value == 0.0 else float("inf")
        else:
            calculated_gap = abs(objective_value - bound_value) / abs(
                objective_value
            )
        bound_order_tolerance = 1e-9 + 1e-12 * max(1.0, abs(objective_value))
        bound_is_valid_for_minimization = (
            bound_value <= objective_value + bound_order_tolerance
        )
    gap_limit = float(requested_gap) * (1.0 + 1e-9) + 1e-12
    reported_gap_meets_request = (
        finite_reported_gap and float(reported_gap) <= gap_limit
    )
    calculated_gap_meets_request = (
        calculated_gap is not None
        and np.isfinite(calculated_gap)
        and calculated_gap <= gap_limit
    )
    required_residuals = (
        "max_constraint_violation",
        "max_int_violation",
        "max_variable_bound_violation",
    )
    residuals_available = all(
        native_residuals.get(name) is not None
        and np.isfinite(native_residuals[name])
        for name in required_residuals
    )
    residuals_meet_tolerance = residuals_available and all(
        abs(float(native_residuals[name])) <= residual_tolerance
        for name in required_residuals
    )
    status_supports_certificate = native_status in SUPPORTED_CERTIFICATE_STATUSES
    passed = bool(
        status_supports_certificate
        and finite_objective
        and finite_bound
        and bound_is_valid_for_minimization
        and reported_gap_meets_request
        and calculated_gap_meets_request
        and residuals_meet_tolerance
    )
    return {
        "passed": passed,
        "native_status_supports_certificate": status_supports_certificate,
        "finite_objective": finite_objective,
        "finite_bound": finite_bound,
        "finite_reported_gap": finite_reported_gap,
        "bound_is_valid_for_minimization": bound_is_valid_for_minimization,
        "reported_gap_meets_request": reported_gap_meets_request,
        "calculated_gap_meets_request": bool(calculated_gap_meets_request),
        "calculated_mip_relative_gap": calculated_gap,
        "requested_mip_relative_gap": float(requested_gap),
        "native_residuals_available": residuals_available,
        "native_residuals_meet_tolerance": bool(residuals_meet_tolerance),
        "native_residual_tolerance": float(residual_tolerance),
    }


def solve_cuopt(
    model: CanonicalMILP,
    *,
    time_limit_seconds: float,
    mip_relative_gap: float,
    threads: int = 0,
    mip_start_values: np.ndarray | None = None,
    mip_acceptance_policy: str = NATIVE_OPTIMAL_ONLY,
    mip_certificate_residual_tolerance: float = 1e-6,
) -> SolveResult:
    if mip_acceptance_policy not in {
        NATIVE_OPTIMAL_ONLY,
        FINITE_BOUND_GAP_CERTIFICATE,
    }:
        raise ScopfError(
            f"Unknown cuOpt MIP acceptance policy: {mip_acceptance_policy!r}"
        )
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
    bound_value = (
        float(raw_bound)
        if raw_bound is not None and np.isfinite(raw_bound)
        else None
    )
    reported_gap = (
        float(raw_gap) if raw_gap is not None and np.isfinite(raw_gap) else None
    )
    native_residuals = {
        name: (
            float(value)
            if (value := _native(getattr(stats, name, None))) is not None
            and np.isfinite(value)
            else None
        )
        for name in (
            "max_constraint_violation",
            "max_int_violation",
            "max_variable_bound_violation",
        )
    }
    gap_certificate = evaluate_mip_gap_certificate(
        native_status=status,
        objective=objective_value,
        bound=bound_value,
        reported_gap=reported_gap,
        requested_gap=float(mip_relative_gap),
        native_residuals=native_residuals,
        residual_tolerance=float(mip_certificate_residual_tolerance),
    )
    native_optimal = status == "Optimal" and bool(gap_certificate["passed"])
    requested_gap_certified = (
        native_optimal
        if mip_acceptance_policy == NATIVE_OPTIMAL_ONLY
        else bool(gap_certificate["passed"])
    )
    reported_status = (
        "OptimalGapMismatch"
        if status == "Optimal" and not gap_certificate["reported_gap_meets_request"]
        else status
    )
    return SolveResult(
        solver="cuopt",
        solver_version=str(getattr(cuopt, "__version__", "unknown")),
        status=reported_status,
        optimal=native_optimal,
        requested_gap_certified=requested_gap_certified,
        has_incumbent=has_incumbent,
        objective=objective_value,
        bound=bound_value,
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
            "reported_gap_meets_request": gap_certificate[
                "reported_gap_meets_request"
            ],
            "mip_acceptance_policy": mip_acceptance_policy,
            "mip_gap_certificate": gap_certificate,
            "requested_gap_certified": requested_gap_certified,
            "partial_integer_mip_start_columns": int(mip_start_columns.size),
        },
    )
