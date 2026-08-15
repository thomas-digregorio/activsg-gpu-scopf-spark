"""Pure-continuous cuOpt PDLP adapter and numerical dual-certificate checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from scipy import sparse

from ..canonical import CanonicalMILP
from ..errors import ScopfError
from .cuopt import audit_cuopt_native_log, native_scaling_vectors


@dataclass(frozen=True)
class ContinuousSolveResult:
    """One continuous LP solve in canonical units."""

    status: str
    optimal: bool
    primal_objective: float | None
    dual_objective: float | None
    values: np.ndarray | None
    solve_time_seconds: float
    statistics: dict[str, Any]


def _row_types(values: np.ndarray) -> np.ndarray:
    normalized: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            normalized.append(value.decode("ascii"))
        else:
            normalized.append(str(value))
    return np.asarray(normalized, dtype="U1")


def validate_numeric_lp_certificate(
    *,
    matrix: sparse.csr_matrix,
    objective: np.ndarray,
    objective_offset: float,
    row_types: np.ndarray,
    row_rhs: np.ndarray,
    column_lower: np.ndarray,
    column_upper: np.ndarray,
    primal: np.ndarray,
    row_dual: np.ndarray,
    reduced_cost: np.ndarray,
    reported_primal_objective: float,
    reported_dual_objective: float,
    native_primal_residual: float,
    native_dual_residual: float,
    native_gap: float,
    residual_tolerance: float,
) -> dict[str, Any]:
    """Independently check cuOpt's FP64 LP primal/dual certificate.

    This is a numerical certificate at ``residual_tolerance`` rather than an
    exact rational certificate.  The returned dual objective is accepted only
    when row-multiplier signs, stationarity, bound compatibility, objectives,
    and cuOpt's own residuals all pass.
    """

    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    objective = np.asarray(objective, dtype=np.float64)
    row_types = _row_types(np.asarray(row_types))
    row_rhs = np.asarray(row_rhs, dtype=np.float64)
    column_lower = np.asarray(column_lower, dtype=np.float64)
    column_upper = np.asarray(column_upper, dtype=np.float64)
    primal = np.asarray(primal, dtype=np.float64)
    row_dual = np.asarray(row_dual, dtype=np.float64)
    reduced_cost = np.asarray(reduced_cost, dtype=np.float64)
    if matrix.shape != (row_rhs.size, objective.size):
        raise ScopfError("LP certificate matrix dimensions are inconsistent")
    if primal.shape != objective.shape or reduced_cost.shape != objective.shape:
        raise ScopfError("LP certificate column-vector dimensions are inconsistent")
    if row_dual.shape != row_rhs.shape or row_types.shape != row_rhs.shape:
        raise ScopfError("LP certificate row-vector dimensions are inconsistent")
    if residual_tolerance < 0:
        raise ScopfError("LP certificate residual tolerance must be nonnegative")
    if not all(
        np.all(np.isfinite(values))
        for values in (objective, row_rhs, primal, row_dual, reduced_cost)
    ):
        raise ScopfError("LP certificate contains a nonfinite dense vector")
    if np.any(~np.isin(row_types, np.asarray(["E", "G", "L"]))):
        raise ScopfError("LP certificate contains an unsupported row type")

    activity = np.asarray(matrix @ primal, dtype=np.float64)
    equality = row_types == "E"
    lower = row_types == "G"
    upper = row_types == "L"
    row_violation = np.zeros(row_rhs.size, dtype=np.float64)
    row_violation[equality] = np.abs(activity[equality] - row_rhs[equality])
    row_violation[lower] = np.maximum(row_rhs[lower] - activity[lower], 0.0)
    row_violation[upper] = np.maximum(activity[upper] - row_rhs[upper], 0.0)
    maximum_primal_row_violation = float(np.max(row_violation)) if row_violation.size else 0.0
    bound_violation = np.maximum(column_lower - primal, 0.0)
    bound_violation = np.maximum(bound_violation, primal - column_upper)
    maximum_primal_bound_violation = float(np.max(bound_violation)) if bound_violation.size else 0.0

    sign_violation = np.zeros(row_rhs.size, dtype=np.float64)
    sign_violation[lower] = np.maximum(-row_dual[lower], 0.0)
    sign_violation[upper] = np.maximum(row_dual[upper], 0.0)
    maximum_row_dual_sign_violation = float(np.max(sign_violation)) if sign_violation.size else 0.0
    stationarity = objective - np.asarray(matrix.T @ row_dual).ravel() - reduced_cost
    maximum_stationarity_residual = (
        float(np.max(np.abs(stationarity))) if stationarity.size else 0.0
    )

    positive_reduced_cost = reduced_cost > residual_tolerance
    negative_reduced_cost = reduced_cost < -residual_tolerance
    missing_lower = positive_reduced_cost & ~np.isfinite(column_lower)
    missing_upper = negative_reduced_cost & ~np.isfinite(column_upper)
    incompatible_bound_columns = np.flatnonzero(missing_lower | missing_upper)
    bound_term = np.zeros(objective.size, dtype=np.float64)
    usable_positive = positive_reduced_cost & np.isfinite(column_lower)
    usable_negative = negative_reduced_cost & np.isfinite(column_upper)
    bound_term[usable_positive] = column_lower[usable_positive] * reduced_cost[usable_positive]
    bound_term[usable_negative] = column_upper[usable_negative] * reduced_cost[usable_negative]
    reconstructed_primal = float(objective @ primal + objective_offset)
    reconstructed_dual = (
        None
        if incompatible_bound_columns.size
        else float(row_rhs @ row_dual + np.sum(bound_term) + objective_offset)
    )
    objective_scale = max(
        1.0,
        abs(float(reported_primal_objective)),
        abs(float(reported_dual_objective)),
    )
    objective_tolerance = max(1e-5, residual_tolerance * objective_scale)
    primal_objective_error = abs(reconstructed_primal - float(reported_primal_objective))
    dual_objective_error = (
        None
        if reconstructed_dual is None
        else abs(reconstructed_dual - float(reported_dual_objective))
    )
    dual_not_above_primal = bool(
        float(reported_dual_objective) <= float(reported_primal_objective) + objective_tolerance
    )
    native_metrics_finite = all(
        isfinite(float(value))
        for value in (
            native_primal_residual,
            native_dual_residual,
            native_gap,
        )
    )
    passed = bool(
        native_metrics_finite
        and maximum_primal_row_violation <= residual_tolerance
        and maximum_primal_bound_violation <= residual_tolerance
        and maximum_row_dual_sign_violation <= residual_tolerance
        and maximum_stationarity_residual <= residual_tolerance
        and incompatible_bound_columns.size == 0
        and primal_objective_error <= objective_tolerance
        and dual_objective_error is not None
        and dual_objective_error <= objective_tolerance
        and abs(float(native_primal_residual)) <= residual_tolerance
        and abs(float(native_dual_residual)) <= residual_tolerance
        and abs(float(native_gap)) <= objective_tolerance
        and dual_not_above_primal
    )
    return {
        "passed": passed,
        "certificate_kind": "independently_reconstructed_fp64_numerical_lp_dual_v1",
        "formal_exact_rational_certificate": False,
        "residual_tolerance": float(residual_tolerance),
        "objective_consistency_tolerance": objective_tolerance,
        "reconstructed_primal_objective": reconstructed_primal,
        "reconstructed_dual_objective": reconstructed_dual,
        "reported_primal_objective": float(reported_primal_objective),
        "reported_dual_objective": float(reported_dual_objective),
        "primal_objective_error": primal_objective_error,
        "dual_objective_error": dual_objective_error,
        "maximum_primal_row_violation": maximum_primal_row_violation,
        "maximum_primal_bound_violation": maximum_primal_bound_violation,
        "maximum_row_dual_sign_violation": maximum_row_dual_sign_violation,
        "maximum_stationarity_residual": maximum_stationarity_residual,
        "incompatible_bound_column_count": int(incompatible_bound_columns.size),
        "incompatible_bound_columns": [int(value) for value in incompatible_bound_columns[:100]],
        "native_primal_residual": float(native_primal_residual),
        "native_dual_residual": float(native_dual_residual),
        "native_gap": float(native_gap),
        "dual_not_above_primal": dual_not_above_primal,
    }


def solve_cuopt_continuous_pdlp(
    model: CanonicalMILP,
    *,
    time_limit_seconds: float,
    optimality_tolerance: float,
    certificate_residual_tolerance: float,
    native_scaling_mode: str,
    native_base_mva: float,
    log_to_console: bool,
) -> ContinuousSolveResult:
    """Relax every integer column and solve the resulting LP using PDLP only."""

    if time_limit_seconds <= 0:
        raise ScopfError("cuOpt continuous solve requires a positive time limit")
    try:
        import cuopt
        from cuopt import linear_programming
        from cuopt.linear_programming.problem import (
            CONTINUOUS,
            MINIMIZE,
            LinearExpression,
            Problem,
        )
        from cuopt.linear_programming.solver_settings import SolverSettings
    except ImportError as exc:
        raise ScopfError("The cuOpt LP adapter requires the DGX Spark runtime") from exc

    objective, lower, upper, original_integrality = model.column_arrays()
    column_scale, row_scale = native_scaling_vectors(
        model,
        mode=native_scaling_mode,
        base_mva=float(native_base_mva),
    )
    native_objective = objective * column_scale
    native_lower = lower / column_scale
    native_upper = upper / column_scale
    problem = Problem("activsg_preventive_scuc_continuous_relaxation")
    variables = [
        problem.addVariable(
            lb=float(native_lower[index]),
            ub=float(native_upper[index]),
            obj=float(native_objective[index]),
            vtype=CONTINUOUS,
            name=name,
        )
        for index, name in enumerate(model.variable_names)
    ]
    objective_columns = np.flatnonzero(native_objective)
    problem.setObjective(
        LinearExpression(
            [variables[int(index)] for index in objective_columns],
            [float(native_objective[int(index)]) for index in objective_columns],
            0.0,
        ),
        sense=MINIMIZE,
    )
    row_lower, row_upper = model.row_bound_arrays()
    native_constraint_count = 0
    for row, name in enumerate(model.row_names):
        indices, coefficients = model.row_entries(row)
        expression = LinearExpression(
            [variables[index] for index in indices],
            [
                float(coefficient) * column_scale[index] * row_scale[row]
                for index, coefficient in zip(indices, coefficients, strict=True)
            ],
            0.0,
        )
        lo = float(row_lower[row] * row_scale[row])
        hi = float(row_upper[row] * row_scale[row])
        if isfinite(lo) and isfinite(hi) and lo == hi:
            problem.addConstraint(expression == lo, name=name)
            native_constraint_count += 1
        else:
            if isfinite(lo):
                problem.addConstraint(expression >= lo, name=f"{name}__lower")
                native_constraint_count += 1
            if isfinite(hi):
                problem.addConstraint(expression <= hi, name=f"{name}__upper")
                native_constraint_count += 1

    settings = SolverSettings()
    settings.set_parameter("time_limit", float(time_limit_seconds))
    settings.set_parameter("method", 1)
    settings.set_parameter("pdlp_solver_mode", 4)
    settings.set_parameter("pdlp_precision", 1)
    settings.set_parameter("log_to_console", bool(log_to_console))
    settings.set_optimality_tolerance(float(optimality_tolerance))
    requested_parameters = {
        "method": 1,
        "pdlp_solver_mode": 4,
        "pdlp_precision": 1,
    }
    readback = {name: int(settings.get_parameter(name)) for name in requested_parameters}
    if readback != requested_parameters:
        raise ScopfError(
            "cuOpt continuous PDLP parameter readback mismatch: "
            f"requested={requested_parameters}, observed={readback}"
        )

    with NamedTemporaryFile(
        mode="w", prefix="activsg-cuopt-pdlp-lp-", suffix=".log", delete=False
    ) as native_log_stream:
        native_log_path = Path(native_log_stream.name)
    settings.set_parameter("log_file", str(native_log_path))
    native_log = ""
    try:
        problem._to_data_model()
        data_model = problem.model
        solution = linear_programming.Solve(data_model, settings)
        native_log = native_log_path.read_text(encoding="utf-8", errors="replace")
    finally:
        native_log_path.unlink(missing_ok=True)
    native_log_audit = audit_cuopt_native_log(native_log)
    status_value = solution.get_termination_status()
    status = str(getattr(status_value, "name", status_value))
    error_status_value = solution.get_error_status()
    error_status = str(getattr(error_status_value, "name", error_status_value))
    solved_by_value = solution.get_solved_by()
    solved_by = str(getattr(solved_by_value, "name", solved_by_value))
    optimal = status == "Optimal" and error_status == "Success"
    primal = None
    canonical_values = None
    primal_objective = None
    dual_objective = None
    dual_certificate: dict[str, Any] = {
        "passed": False,
        "reason": "LP solve did not return Optimal/Success",
    }
    lp_stats = {key: float(value) for key, value in solution.get_lp_stats().items()}
    if optimal:
        primal = np.asarray(solution.get_primal_solution(), dtype=np.float64)
        row_dual = np.asarray(solution.get_dual_solution(), dtype=np.float64)
        reduced_cost = np.asarray(solution.get_reduced_cost(), dtype=np.float64)
        primal_objective = float(solution.get_primal_objective())
        dual_objective = float(solution.get_dual_objective())
        canonical_values = primal * column_scale
        offsets = np.asarray(data_model.get_constraint_matrix_offsets(), dtype=np.int64)
        indices = np.asarray(data_model.get_constraint_matrix_indices(), dtype=np.int32)
        matrix_values = np.asarray(data_model.get_constraint_matrix_values(), dtype=np.float64)
        native_matrix = sparse.csr_matrix(
            (matrix_values, indices, offsets),
            shape=(native_constraint_count, model.num_columns),
        )
        row_rhs = np.asarray(data_model.get_constraint_bounds(), dtype=np.float64)
        row_types = _row_types(np.asarray(data_model.get_row_types()))
        dual_certificate = validate_numeric_lp_certificate(
            matrix=native_matrix,
            objective=np.asarray(data_model.get_objective_coefficients(), dtype=np.float64),
            objective_offset=float(data_model.get_objective_offset()),
            row_types=row_types,
            row_rhs=row_rhs,
            column_lower=np.asarray(data_model.get_variable_lower_bounds(), dtype=np.float64),
            column_upper=np.asarray(data_model.get_variable_upper_bounds(), dtype=np.float64),
            primal=primal,
            row_dual=row_dual,
            reduced_cost=reduced_cost,
            reported_primal_objective=primal_objective,
            reported_dual_objective=dual_objective,
            native_primal_residual=lp_stats["primal_residual"],
            native_dual_residual=lp_stats["dual_residual"],
            native_gap=lp_stats["gap"],
            residual_tolerance=float(certificate_residual_tolerance),
        )
        dual_certificate.update(
            {
                "row_dual_sha256": hashlib.sha256(row_dual.tobytes()).hexdigest(),
                "reduced_cost_sha256": hashlib.sha256(reduced_cost.tobytes()).hexdigest(),
                "row_dual_count": int(row_dual.size),
                "reduced_cost_count": int(reduced_cost.size),
            }
        )

    variable_types = _row_types(np.asarray(problem.model.get_variable_types()))
    branch_text = native_log.casefold()
    no_branch_and_bound = not any(
        marker in branch_text
        for marker in ("branch-and-bound", "branch and bound", "mip node", "b&b")
    )
    return ContinuousSolveResult(
        status=status,
        optimal=optimal,
        primal_objective=primal_objective,
        dual_objective=dual_objective,
        values=canonical_values,
        solve_time_seconds=float(solution.get_solve_time()),
        statistics={
            "cuopt_version": str(getattr(cuopt, "__version__", "unknown")),
            "error_status": error_status,
            "error_message": str(solution.get_error_message()),
            "termination_reason": str(solution.get_termination_reason()),
            "solved_by": solved_by,
            "solved_by_pdlp": bool(solution.get_solved_by_pdlp()),
            "lp_stats": lp_stats,
            "dual_certificate": dual_certificate,
            "original_integer_columns": int(np.count_nonzero(original_integrality)),
            "native_integer_columns": int(np.count_nonzero(variable_types == "I")),
            "native_continuous_columns": int(np.count_nonzero(variable_types == "C")),
            "canonical_columns_translated": model.num_columns,
            "canonical_rows_translated": model.num_rows,
            "native_constraints_translated": native_constraint_count,
            "native_scaling_mode": native_scaling_mode,
            "method_parameters_requested": requested_parameters,
            "method_parameters_readback": readback,
            "optimality_tolerance": float(optimality_tolerance),
            "certificate_residual_tolerance": float(certificate_residual_tolerance),
            "native_log_audit": native_log_audit,
            "native_log_branch_and_bound_markers_absent": no_branch_and_bound,
            "native_log_sha256": hashlib.sha256(native_log.encode("utf-8")).hexdigest(),
            "native_log_bytes": len(native_log.encode("utf-8")),
        },
    )
