"""Pure-continuous cuOpt PDLP adapter and numerical dual-certificate checks."""

from __future__ import annotations

import hashlib
import heapq
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from scipy import sparse

from ..canonical import CanonicalMILP
from ..errors import ScopfError
from ..network import NetworkData
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


@dataclass(frozen=True)
class RedundantColumnBounds:
    """Finite bounds rigorously implied by the unchanged canonical model."""

    lower: np.ndarray
    upper: np.ndarray
    audit: dict[str, Any]


def derive_rate_a_angle_bounds(
    network: NetworkData,
    theta_columns: np.ndarray,
    *,
    total_columns: int,
) -> RedundantColumnBounds:
    """Bound every bus angle using finite RATE_A paths to the reference bus.

    For branch ``e``, the existing DC equation and ``RATE_A`` constraint imply
    ``|theta_i - theta_j| <= |shift_e| + RATE_A_e / |baseMVA * b_e|``.
    Summing those inequalities along any path to the fixed reference angle is
    valid; shortest paths provide the tightest bounds available from this
    symmetric construction.  These bounds are redundant, not a model relaxation
    or restriction.
    """

    columns = np.asarray(theta_columns, dtype=np.int64)
    bus_count = int(network.bus_ids.size)
    if columns.shape != (bus_count,):
        raise ScopfError("Theta-column identities do not match the network buses")
    if np.any(columns < 0) or np.any(columns >= total_columns):
        raise ScopfError("Theta-column identities contain an invalid canonical column")
    if np.unique(columns).size != columns.size:
        raise ScopfError("Theta-column identities are not unique")
    if not (0 <= int(network.reference_bus_index) < bus_count):
        raise ScopfError("The network reference-bus index is invalid")

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(bus_count)]
    finite_edge_count = 0
    for active_index in range(network.active_branch_source_rows.size):
        rate = float(network.rate_a_mw[active_index])
        susceptance = float(network.susceptance_pu[active_index])
        shift = float(network.phase_shift_rad[active_index])
        if not (
            isfinite(rate)
            and rate > 0.0
            and isfinite(susceptance)
            and susceptance != 0.0
            and isfinite(shift)
        ):
            continue
        coefficient = float(network.base_mva) * susceptance
        weight = abs(shift) + rate / abs(coefficient)
        if not isfinite(weight) or weight <= 0.0:
            raise ScopfError("A RATE_A-implied angle-difference bound is invalid")
        source = int(network.from_bus_index[active_index])
        target = int(network.to_bus_index[active_index])
        adjacency[source].append((target, weight))
        adjacency[target].append((source, weight))
        finite_edge_count += 1

    distances = np.full(bus_count, np.inf, dtype=np.float64)
    reference = int(network.reference_bus_index)
    distances[reference] = 0.0
    queue: list[tuple[float, int]] = [(0.0, reference)]
    while queue:
        distance, bus = heapq.heappop(queue)
        if distance != distances[bus]:
            continue
        for neighbor, weight in adjacency[bus]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    if not np.all(np.isfinite(distances)):
        unreachable = network.bus_ids[~np.isfinite(distances)]
        raise ScopfError(
            "Finite-RATE_A branches do not connect every bus to the reference; "
            f"unreachable bus count={unreachable.size}"
        )

    lower = np.full(total_columns, -np.inf, dtype=np.float64)
    upper = np.full(total_columns, np.inf, dtype=np.float64)
    lower[columns] = -distances
    upper[columns] = distances
    digest = hashlib.sha256(distances.tobytes()).hexdigest()
    return RedundantColumnBounds(
        lower=lower,
        upper=upper,
        audit={
            "policy": "rate_a_dc_shortest_path_v1",
            "proof": (
                "existing_dc_flow_equation_and_rate_a_imply_each_edge_angle_bound; "
                "shortest_path_sums_to_fixed_reference_bound_each_bus_angle"
            ),
            "reference_bus_id": int(network.bus_ids[reference]),
            "bounded_theta_column_count": int(columns.size),
            "finite_rate_a_edge_count": finite_edge_count,
            "maximum_absolute_angle_bound_rad": float(np.max(distances)),
            "angle_bound_vector_sha256": digest,
            "canonical_physical_feasible_set_changed": False,
        },
    )


def _pdlp_log_metrics(native_log: str) -> dict[str, float | None]:
    """Extract cuOpt's own final absolute/relative PDLP residual telemetry."""

    metrics: dict[str, float | None] = {}
    patterns = {
        "gap": r"Duality gap \(abs/rel\):\s+([+\-0-9.eE]+)\s*/\s*([+\-0-9.eE]+)",
        "primal": r"Primal infeasibility \(abs/rel\):\s+([+\-0-9.eE]+)\s*/\s*([+\-0-9.eE]+)",
        "dual": r"Dual infeasibility \(abs/rel\):\s+([+\-0-9.eE]+)\s*/\s*([+\-0-9.eE]+)",
    }
    for name, pattern in patterns.items():
        matches = re.findall(pattern, native_log)
        if not matches:
            metrics[f"{name}_absolute"] = None
            metrics[f"{name}_relative"] = None
            continue
        absolute, relative = matches[-1]
        metrics[f"{name}_absolute"] = float(absolute)
        metrics[f"{name}_relative"] = float(relative)
    return metrics


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
    optimality_tolerance: float,
    primal_feasibility_tolerance: float,
    residual_tolerance: float,
) -> dict[str, Any]:
    """Independently check cuOpt's FP64 LP primal/dual certificate.

    This is a numerical certificate rather than an exact rational certificate.
    The returned dual objective is accepted only when row-multiplier signs,
    stationarity, bound compatibility, objectives, and cuOpt's own scaled
    residuals all pass.
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
    if residual_tolerance < 0 or primal_feasibility_tolerance < 0:
        raise ScopfError("LP certificate tolerances must be nonnegative")
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
    dual_feasible_row_dual = row_dual.copy()
    dual_feasible_row_dual[lower] = np.maximum(dual_feasible_row_dual[lower], 0.0)
    dual_feasible_row_dual[upper] = np.minimum(dual_feasible_row_dual[upper], 0.0)
    implied_reduced_cost = objective - np.asarray(matrix.T @ dual_feasible_row_dual).ravel()
    returned_reduced_cost_mismatch = reduced_cost - implied_reduced_cost
    maximum_returned_reduced_cost_mismatch = (
        float(np.max(np.abs(returned_reduced_cost_mismatch)))
        if returned_reduced_cost_mismatch.size
        else 0.0
    )
    reported_gap_threshold = optimality_tolerance * (
        1.0 + abs(float(reported_primal_objective)) + abs(float(reported_dual_objective))
    )
    effective_reduced_cost = implied_reduced_cost.copy()
    needs_missing_lower = (effective_reduced_cost > 0.0) & ~np.isfinite(column_lower)
    needs_missing_upper = (effective_reduced_cost < 0.0) & ~np.isfinite(column_upper)
    numerically_zero_unbounded = (needs_missing_lower | needs_missing_upper) & (
        np.abs(effective_reduced_cost) <= residual_tolerance
    )
    effective_reduced_cost[numerically_zero_unbounded] = 0.0
    stationarity_adjustment = implied_reduced_cost - effective_reduced_cost
    maximum_stationarity_residual = (
        float(np.max(np.abs(stationarity_adjustment))) if stationarity_adjustment.size else 0.0
    )
    positive_reduced_cost = effective_reduced_cost > 0.0
    negative_reduced_cost = effective_reduced_cost < 0.0
    missing_lower = positive_reduced_cost & ~np.isfinite(column_lower)
    missing_upper = negative_reduced_cost & ~np.isfinite(column_upper)
    incompatible_bound_columns = np.flatnonzero(missing_lower | missing_upper)
    bound_term = np.zeros(objective.size, dtype=np.float64)
    usable_positive = positive_reduced_cost & np.isfinite(column_lower)
    usable_negative = negative_reduced_cost & np.isfinite(column_upper)
    bound_term[usable_positive] = (
        column_lower[usable_positive] * effective_reduced_cost[usable_positive]
    )
    bound_term[usable_negative] = (
        column_upper[usable_negative] * effective_reduced_cost[usable_negative]
    )
    reconstructed_primal = float(objective @ primal + objective_offset)
    reconstructed_dual = (
        None
        if incompatible_bound_columns.size
        else float(row_rhs @ dual_feasible_row_dual + np.sum(bound_term) + objective_offset)
    )
    objective_scale = max(
        1.0,
        abs(float(reported_primal_objective)),
        abs(float(reported_dual_objective)),
    )
    objective_tolerance = max(
        1e-5,
        residual_tolerance * objective_scale,
    )
    primal_objective_error = abs(reconstructed_primal - float(reported_primal_objective))
    dual_objective_error = (
        None
        if reconstructed_dual is None
        else abs(reconstructed_dual - float(reported_dual_objective))
    )
    dual_not_above_primal = bool(
        float(reported_dual_objective) <= float(reported_primal_objective) + objective_tolerance
    )
    conservative_numerical_lower_bound = (
        None
        if reconstructed_dual is None
        else min(float(reported_dual_objective), reconstructed_dual) - objective_tolerance
    )
    reconstructed_bound_not_above_primal = bool(
        conservative_numerical_lower_bound is not None
        and conservative_numerical_lower_bound <= reconstructed_primal + objective_tolerance
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
        and maximum_primal_row_violation <= primal_feasibility_tolerance
        and maximum_primal_bound_violation <= primal_feasibility_tolerance
        and maximum_row_dual_sign_violation <= residual_tolerance
        and maximum_stationarity_residual <= residual_tolerance
        and incompatible_bound_columns.size == 0
        and primal_objective_error <= objective_tolerance
        and reconstructed_dual is not None
        and reconstructed_bound_not_above_primal
    )
    return {
        "passed": passed,
        "certificate_kind": "independently_reconstructed_fp64_numerical_lp_dual_v1",
        "formal_exact_rational_certificate": False,
        "primal_feasibility_tolerance": float(primal_feasibility_tolerance),
        "residual_tolerance": float(residual_tolerance),
        "objective_consistency_tolerance": objective_tolerance,
        "reconstructed_primal_objective": reconstructed_primal,
        "reconstructed_dual_objective": reconstructed_dual,
        "reported_primal_objective": float(reported_primal_objective),
        "reported_dual_objective": float(reported_dual_objective),
        "conservative_numerical_lower_bound": conservative_numerical_lower_bound,
        "primal_objective_error": primal_objective_error,
        "dual_objective_error": dual_objective_error,
        "maximum_primal_row_violation": maximum_primal_row_violation,
        "maximum_primal_bound_violation": maximum_primal_bound_violation,
        "maximum_row_dual_sign_violation": maximum_row_dual_sign_violation,
        "maximum_stationarity_residual": maximum_stationarity_residual,
        "stationarity_policy": (
            "derive_c_minus_ATy; only numerically zero residuals on columns "
            "without the required finite bound"
        ),
        "maximum_returned_reduced_cost_mismatch": (maximum_returned_reduced_cost_mismatch),
        "returned_reduced_cost_used_for_certificate": False,
        "implied_reduced_cost_policy": "objective_minus_matrix_transpose_times_row_dual",
        "incompatible_bound_column_count": int(incompatible_bound_columns.size),
        "incompatible_bound_columns": [int(value) for value in incompatible_bound_columns[:100]],
        "native_primal_residual": float(native_primal_residual),
        "native_dual_residual": float(native_dual_residual),
        "native_gap": float(native_gap),
        "native_metrics_used_as_telemetry_only": True,
        "optimality_tolerance": float(optimality_tolerance),
        "reported_gap_threshold": reported_gap_threshold,
        "dual_not_above_primal": dual_not_above_primal,
        "reported_dual_objective_consistent": bool(
            dual_objective_error is not None and dual_objective_error <= objective_tolerance
        ),
        "reconstructed_bound_not_above_primal": reconstructed_bound_not_above_primal,
    }


def solve_cuopt_continuous_pdlp(
    model: CanonicalMILP,
    *,
    time_limit_seconds: float,
    optimality_tolerance: float,
    primal_feasibility_tolerance: float,
    certificate_residual_tolerance: float,
    native_scaling_mode: str,
    native_base_mva: float,
    log_to_console: bool,
    per_constraint_residual: bool = False,
    redundant_bounds: RedundantColumnBounds | None = None,
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
    redundant_bounds_audit: dict[str, Any] | None = None
    if redundant_bounds is not None:
        candidate_lower = np.asarray(redundant_bounds.lower, dtype=np.float64)
        candidate_upper = np.asarray(redundant_bounds.upper, dtype=np.float64)
        if candidate_lower.shape != lower.shape or candidate_upper.shape != upper.shape:
            raise ScopfError("Redundant LP bounds do not match the canonical columns")
        effective_lower = np.maximum(lower, candidate_lower)
        effective_upper = np.minimum(upper, candidate_upper)
        if np.any(effective_lower > effective_upper):
            raise ScopfError("Redundant LP bounds conflict with canonical bounds")
        tightened = np.flatnonzero((effective_lower > lower) | (effective_upper < upper))
        if any(not model.variable_names[int(index)].startswith("theta_") for index in tightened):
            raise ScopfError("Only provably redundant theta bounds are accepted by this adapter")
        lower = effective_lower
        upper = effective_upper
        redundant_bounds_audit = dict(redundant_bounds.audit)
        redundant_bounds_audit["tightened_native_column_count"] = int(tightened.size)
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
    settings.set_parameter("per_constraint_residual", bool(per_constraint_residual))
    settings.set_parameter("log_to_console", bool(log_to_console))
    settings.set_optimality_tolerance(float(optimality_tolerance))
    requested_parameters = {
        "method": 1,
        "pdlp_solver_mode": 4,
        "pdlp_precision": 1,
        "per_constraint_residual": bool(per_constraint_residual),
    }
    readback = {
        name: (
            bool(settings.get_parameter(name))
            if isinstance(expected, bool)
            else int(settings.get_parameter(name))
        )
        for name, expected in requested_parameters.items()
    }
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
    native_log_metrics = _pdlp_log_metrics(native_log)
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
            optimality_tolerance=float(optimality_tolerance),
            primal_feasibility_tolerance=float(primal_feasibility_tolerance),
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
            "redundant_bounds": redundant_bounds_audit,
            "method_parameters_requested": requested_parameters,
            "method_parameters_readback": readback,
            "optimality_tolerance": float(optimality_tolerance),
            "primal_feasibility_tolerance": float(primal_feasibility_tolerance),
            "certificate_residual_tolerance": float(certificate_residual_tolerance),
            "native_log_audit": native_log_audit,
            "native_log_final_metrics": native_log_metrics,
            "native_log_branch_and_bound_markers_absent": no_branch_and_bound,
            "native_log_sha256": hashlib.sha256(native_log.encode("utf-8")).hexdigest(),
            "native_log_bytes": len(native_log.encode("utf-8")),
        },
    )
