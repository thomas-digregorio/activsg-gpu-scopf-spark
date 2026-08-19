"""Exact dispatch-only projection for fixed-commitment feasibility polishing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import fsum, inf, isfinite
from typing import Any

import numpy as np
import numpy.typing as npt

from .canonical import CanonicalMILP
from .errors import ScopfError
from .reduced import ReducedMaster

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _numeric_backend(name: str) -> tuple[Any, Any]:
    if name == "numpy":
        return np, None
    if name == "cupy":
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cupy_sparse
        except ImportError as exc:
            raise ScopfError("CuPy fixed-commitment repair is unavailable") from exc
        return cp, cupy_sparse
    raise ScopfError(f"Unknown fixed-commitment repair backend: {name}")


def project_boxed_sum(
    values: npt.ArrayLike,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    *,
    target_sum: float,
    backend: str,
    iterations: int = 100,
) -> tuple[FloatArray, dict[str, object]]:
    """Project a vector onto one box-constrained sum equality.

    This is the Euclidean projection ``clip(values - lambda, lower, upper)``
    with the scalar multiplier found by deterministic bisection.  Vector
    arithmetic stays on the selected backend; the GPU path uses CuPy
    primitives and no custom kernel.
    """

    if iterations <= 0:
        raise ScopfError("Boxed-sum projection iterations must be positive")
    xp, _ = _numeric_backend(backend)
    original = xp.asarray(values, dtype=xp.float64)
    lo = xp.asarray(lower, dtype=xp.float64)
    hi = xp.asarray(upper, dtype=xp.float64)
    if original.ndim != 1 or lo.shape != original.shape or hi.shape != original.shape:
        raise ScopfError("Boxed-sum projection arrays have incompatible shapes")
    if int(original.size) == 0:
        if abs(float(target_sum)) > 1e-10:
            raise ScopfError("Empty boxed-sum projection has a nonzero target")
        return np.empty(0, dtype=np.float64), {
            "policy": "boxed_sum_euclidean_projection_v1",
            "backend": backend,
            "column_count": 0,
            "target_sum": float(target_sum),
            "final_sum": 0.0,
            "absolute_balance_residual": 0.0,
        }
    if not bool(xp.all(xp.isfinite(original)).item()):
        raise ScopfError("Boxed-sum projection values are nonfinite")
    if not bool(xp.all(xp.isfinite(lo)).item()) or not bool(
        xp.all(xp.isfinite(hi)).item()
    ):
        raise ScopfError("Boxed-sum projection requires finite bounds")
    if bool(xp.any(lo > hi).item()):
        raise ScopfError("Boxed-sum projection has invalid bounds")
    target = float(target_sum)
    lower_sum = float(xp.sum(lo).item())
    upper_sum = float(xp.sum(hi).item())
    feasibility_tolerance = 1e-10 * max(
        1.0, abs(target), abs(lower_sum), abs(upper_sum)
    )
    if target < lower_sum - feasibility_tolerance or target > upper_sum + feasibility_tolerance:
        raise ScopfError("Boxed-sum projection target is outside aggregate bounds")
    target = min(max(target, lower_sum), upper_sum)

    lambda_lower = float(xp.min(original - hi).item())
    lambda_upper = float(xp.max(original - lo).item())
    for _ in range(iterations):
        multiplier = 0.5 * (lambda_lower + lambda_upper)
        candidate = xp.clip(original - multiplier, lo, hi)
        if float(xp.sum(candidate).item()) > target:
            lambda_lower = multiplier
        else:
            lambda_upper = multiplier
    multiplier = 0.5 * (lambda_lower + lambda_upper)
    projected = xp.clip(original - multiplier, lo, hi)

    # Correct the final few ulps without leaving the box.  This remains a set
    # of device vector/scalar operations on the CuPy path.
    for _ in range(4):
        residual = target - float(xp.sum(projected).item())
        if abs(residual) <= 5e-13 * max(1.0, abs(target)):
            break
        slack = hi - projected if residual > 0.0 else projected - lo
        index = int(xp.argmax(slack).item())
        available = float(slack[index].item())
        if available <= 0.0:
            break
        adjustment = min(abs(residual), available)
        projected[index] += adjustment if residual > 0.0 else -adjustment

    final_sum = float(xp.sum(projected).item())
    maximum_bound_violation = float(
        xp.max(xp.maximum(lo - projected, projected - hi)).item()
    )
    if backend == "cupy":
        import cupy as cp

        host = cp.asnumpy(projected)
    else:
        host = np.asarray(projected, dtype=np.float64)
    return host, {
        "policy": "boxed_sum_euclidean_projection_v1",
        "backend": backend,
        "column_count": int(host.size),
        "bisection_iterations": int(iterations),
        "target_sum": float(target),
        "initial_sum": float(xp.sum(original).item()),
        "final_sum": final_sum,
        "absolute_balance_residual": abs(final_sum - target),
        "maximum_bound_violation": max(maximum_bound_violation, 0.0),
        "l2_adjustment": float(xp.linalg.norm(projected - original).item()),
        "maximum_absolute_adjustment": float(
            xp.max(xp.abs(projected - original)).item()
        ),
    }


def repair_along_feasible_segment(
    model: CanonicalMILP,
    feasible_values: npt.ArrayLike,
    target_values: npt.ArrayLike,
    *,
    tolerance: float,
    backend: str,
) -> tuple[FloatArray, dict[str, object]]:
    """Keep the largest feasible fraction of a numerically noisy LP target."""

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ScopfError("Feasible-segment repair tolerance must be nonnegative")
    seed_host = np.asarray(feasible_values, dtype=np.float64)
    target_host = np.asarray(target_values, dtype=np.float64)
    expected = (model.num_columns,)
    if seed_host.shape != expected or target_host.shape != expected:
        raise ScopfError("Feasible-segment repair vectors have the wrong shape")
    if not np.all(np.isfinite(seed_host)) or not np.all(np.isfinite(target_host)):
        raise ScopfError("Feasible-segment repair vectors are nonfinite")

    xp, sparse_backend = _numeric_backend(backend)
    seed = xp.asarray(seed_host, dtype=xp.float64)
    target = xp.asarray(target_host, dtype=xp.float64)
    direction = target - seed
    scipy_matrix = model.matrix_csr()
    matrix = (
        scipy_matrix
        if backend == "numpy"
        else sparse_backend.csr_matrix(scipy_matrix)
    )
    activity = matrix @ seed
    activity_delta = matrix @ direction
    row_lower = xp.asarray(model.row_lower, dtype=xp.float64)
    row_upper = xp.asarray(model.row_upper, dtype=xp.float64)
    column_lower = xp.asarray(model.column_lower, dtype=xp.float64)
    column_upper = xp.asarray(model.column_upper, dtype=xp.float64)

    seed_row_violation = xp.maximum(row_lower - activity, activity - row_upper)
    seed_column_violation = xp.maximum(column_lower - seed, seed - column_upper)
    seed_maximum_violation = max(
        float(xp.max(seed_row_violation).item()) if model.num_rows else 0.0,
        float(xp.max(seed_column_violation).item()) if model.num_columns else 0.0,
        0.0,
    )
    if seed_maximum_violation > tolerance:
        raise ScopfError(
            "Feasible-segment repair seed exceeds its registered tolerance: "
            f"{seed_maximum_violation} > {tolerance}"
        )

    limits: list[float] = [1.0]

    def add_upper_limits(current: Any, delta: Any, bound: Any) -> None:
        mask = xp.isfinite(bound) & (delta > 0.0)
        if bool(xp.any(mask).item()):
            ratios = (bound[mask] + tolerance - current[mask]) / delta[mask]
            limits.append(float(xp.min(ratios).item()))

    def add_lower_limits(current: Any, delta: Any, bound: Any) -> None:
        mask = xp.isfinite(bound) & (delta < 0.0)
        if bool(xp.any(mask).item()):
            ratios = (current[mask] - (bound[mask] - tolerance)) / (-delta[mask])
            limits.append(float(xp.min(ratios).item()))

    add_upper_limits(activity, activity_delta, row_upper)
    add_lower_limits(activity, activity_delta, row_lower)
    add_upper_limits(seed, direction, column_upper)
    add_lower_limits(seed, direction, column_lower)
    raw_step = min(limits)
    if not np.isfinite(raw_step):
        raise ScopfError("Feasible-segment repair produced a nonfinite step")
    step = min(1.0, max(0.0, raw_step))
    if step < 1.0:
        step = max(0.0, step - max(1e-12, 64.0 * np.finfo(np.float64).eps))
    repaired = seed + step * direction
    repaired_activity = matrix @ repaired
    repaired_row_violation = xp.maximum(
        row_lower - repaired_activity, repaired_activity - row_upper
    )
    repaired_column_violation = xp.maximum(
        column_lower - repaired, repaired - column_upper
    )
    repaired_maximum_violation = max(
        float(xp.max(repaired_row_violation).item()) if model.num_rows else 0.0,
        float(xp.max(repaired_column_violation).item()) if model.num_columns else 0.0,
        0.0,
    )
    if repaired_maximum_violation > tolerance * (1.0 + 1e-8) + 1e-12:
        raise ScopfError("Feasible-segment repair failed its postcondition")
    if backend == "cupy":
        import cupy as cp

        host = cp.asnumpy(repaired)
    else:
        host = np.asarray(repaired, dtype=np.float64)
    return host, {
        "policy": "maximum_convex_feasible_segment_gpu_v1",
        "backend": backend,
        "step_fraction": float(step),
        "raw_limiting_step_fraction": float(raw_step),
        "target_retained_percent": 100.0 * float(step),
        "seed_maximum_violation": seed_maximum_violation,
        "repaired_maximum_violation": repaired_maximum_violation,
        "registered_tolerance": float(tolerance),
        "mathematical_feasible_set_changed": False,
    }


def condition_fixed_commitment_start_projection(
    model: CanonicalMILP,
    *,
    coefficient_zero_tolerance: float,
) -> tuple[CanonicalMILP, dict[str, object]]:
    """Drop numerical dust from an auxiliary start LP using an inner approximation.

    The ordinary reduced SCOPF cleanup relaxes row bounds outward because that
    model is used for lower bounds.  A MIP-start polisher has the opposite
    requirement: every point it returns must remain feasible for the exact
    target row.  For each dropped coefficient this routine therefore tightens
    finite row bounds by the exact boxed minimum/maximum omitted activity.
    The conditioned model is used only to construct a start and never changes
    the target MILP or any lower-bound certificate.
    """

    tolerance = float(coefficient_zero_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ScopfError("Start-projection coefficient tolerance must be nonnegative")
    conditioned = CanonicalMILP()
    for name, objective, lower, upper, integer in zip(
        model.variable_names,
        model.objective,
        model.column_lower,
        model.column_upper,
        model.integrality,
        strict=True,
    ):
        conditioned.add_variable(
            name,
            objective=float(objective),
            lower=float(lower),
            upper=float(upper),
            integer=bool(integer),
        )

    dropped_count = 0
    reverted_row_count = 0
    maximum_dropped = 0.0
    maximum_lower_tightening = 0.0
    maximum_upper_tightening = 0.0
    retained_magnitudes: list[float] = []
    for row, name in enumerate(model.row_names):
        indices, coefficients = model.row_entries(row)
        kept: dict[int, float] = {}
        dropped_minimum_terms: list[float] = []
        dropped_maximum_terms: list[float] = []
        row_dropped = 0
        row_maximum_dropped = 0.0
        for column, coefficient in zip(indices, coefficients, strict=True):
            value = float(coefficient)
            magnitude = abs(value)
            lower = float(model.column_lower[column])
            upper = float(model.column_upper[column])
            can_drop = bool(
                0.0 < magnitude < tolerance
                and isfinite(lower)
                and isfinite(upper)
            )
            if not can_drop:
                kept[int(column)] = value
                if magnitude > 0.0:
                    retained_magnitudes.append(magnitude)
                continue
            first = value * lower
            second = value * upper
            dropped_minimum_terms.append(min(first, second))
            dropped_maximum_terms.append(max(first, second))
            row_dropped += 1
            row_maximum_dropped = max(row_maximum_dropped, magnitude)

        original_lower = float(model.row_lower[row])
        original_upper = float(model.row_upper[row])
        dropped_minimum = fsum(dropped_minimum_terms)
        dropped_maximum = fsum(dropped_maximum_terms)
        tightened_lower = (
            float(np.nextafter(original_lower - dropped_minimum, inf))
            if isfinite(original_lower) and row_dropped
            else original_lower
        )
        tightened_upper = (
            float(np.nextafter(original_upper - dropped_maximum, -inf))
            if isfinite(original_upper) and row_dropped
            else original_upper
        )
        if tightened_lower > tightened_upper:
            kept = {
                int(column): float(value)
                for column, value in zip(indices, coefficients, strict=True)
            }
            retained_magnitudes.extend(
                abs(float(value)) for value in coefficients if float(value) != 0.0
            )
            tightened_lower = original_lower
            tightened_upper = original_upper
            reverted_row_count += 1
            row_dropped = 0
            row_maximum_dropped = 0.0
        else:
            dropped_count += row_dropped
            maximum_dropped = max(maximum_dropped, row_maximum_dropped)
            if isfinite(original_lower):
                maximum_lower_tightening = max(
                    maximum_lower_tightening,
                    tightened_lower - original_lower,
                )
            if isfinite(original_upper):
                maximum_upper_tightening = max(
                    maximum_upper_tightening,
                    original_upper - tightened_upper,
                )
        conditioned.add_row(
            name,
            kept,
            lower=tightened_lower,
            upper=tightened_upper,
        )

    return conditioned, {
        "policy": "boxed_dust_drop_with_inward_row_bounds_for_start_only_v1",
        "coefficient_zero_tolerance": tolerance,
        "dropped_coefficient_count": dropped_count,
        "reverted_incompatible_row_count": reverted_row_count,
        "maximum_absolute_dropped_coefficient": maximum_dropped,
        "minimum_absolute_retained_coefficient": (
            min(retained_magnitudes) if retained_magnitudes else None
        ),
        "maximum_lower_bound_tightening": maximum_lower_tightening,
        "maximum_upper_bound_tightening": maximum_upper_tightening,
        "solver_rows_are_inner_approximations_of_exact_rows": True,
        "target_milp_changed": False,
        "lower_bound_certificate_used": False,
    }


class FixedCommitmentProjectionInfeasible(ScopfError):
    """A fixed candidate violates a coupling row with no free dispatch support."""

    def __init__(self, row_name: str, *, lower: float, upper: float) -> None:
        super().__init__(
            "Fixed-commitment projection found a violated constant coupling row: "
            f"{row_name}"
        )
        self.row_name = row_name
        self.shifted_lower = float(lower)
        self.shifted_upper = float(upper)


def _maximum_column_violation(model: CanonicalMILP, values: FloatArray) -> float:
    lower = np.asarray(model.column_lower, dtype=np.float64)
    upper = np.asarray(model.column_upper, dtype=np.float64)
    return float(max(np.max(lower - values), np.max(values - upper), 0.0))


@dataclass(frozen=True)
class FixedCommitmentProjection:
    """A dispatch-only LP whose feasible set is the exact fixed-u projection.

    The commitment and PWL segment variables are local generator bookkeeping.
    Once commitment is binary and fixed, their projection onto dispatch is
    exactly ``PMIN <= Pg <= PMAX`` for an on unit and ``Pg = 0`` for an off
    unit.  Network and contingency coupling rows depend only on dispatch, so
    they can be copied without changing their mathematical meaning.
    """

    canonical: CanonicalMILP
    source_master: ReducedMaster
    commitment: npt.NDArray[np.int8]
    projected_column_by_generator: dict[int, int]
    fixed_dispatch_by_generator: dict[int, float]
    cost_column_by_generator: dict[int, int]
    audit: dict[str, object]

    def dispatch(self, projected_values: npt.ArrayLike) -> FloatArray:
        values = np.asarray(projected_values, dtype=np.float64)
        if values.shape != (self.canonical.num_columns,):
            raise ScopfError("Fixed-commitment projected primal has the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ScopfError("Fixed-commitment projected primal is nonfinite")
        dispatch = np.zeros(
            self.source_master.index.generator_source_rows.size, dtype=np.float64
        )
        for position, generator in enumerate(
            self.source_master.index.generator_source_rows
        ):
            source_row = int(generator)
            column = self.projected_column_by_generator.get(source_row)
            dispatch[position] = (
                values[column]
                if column is not None
                else self.fixed_dispatch_by_generator[source_row]
            )
        return dispatch

    def lift(self, projected_values: npt.ArrayLike) -> FloatArray:
        """Lift dispatch to the original exact-PMIN ten-segment formulation."""

        projected = np.asarray(projected_values, dtype=np.float64)
        dispatch = self.dispatch(projected)
        source = self.source_master
        lifted = np.zeros(source.canonical.num_columns, dtype=np.float64)
        for position, generator in enumerate(source.index.generator_source_rows):
            source_row = int(generator)
            committed = int(self.commitment[position])
            curve = source.costs[source_row]
            lifted[source.index.commitment_by_generator[source_row]] = float(
                committed
            )
            power = float(dispatch[position])
            lifted[source.index.dispatch_by_generator[source_row]] = power
            remaining = power - curve.pmin_mw if committed else 0.0
            for column, width in zip(
                source.index.segments_by_generator[source_row],
                curve.segment_widths_mw,
                strict=True,
            ):
                used = min(max(remaining, 0.0), float(width))
                if column is not None:
                    lifted[column] = used
                remaining -= used
        return lifted

    def project_source_values(self, source_values: npt.ArrayLike) -> FloatArray:
        """Project one source-space point into dispatch/cost epigraph space."""

        source = np.asarray(source_values, dtype=np.float64)
        if source.shape != (self.source_master.canonical.num_columns,):
            raise ScopfError("Fixed-commitment source primal has the wrong shape")
        if not np.all(np.isfinite(source)):
            raise ScopfError("Fixed-commitment source primal is nonfinite")
        projected = np.zeros(self.canonical.num_columns, dtype=np.float64)
        for position, generator in enumerate(
            self.source_master.index.generator_source_rows
        ):
            source_row = int(generator)
            dispatch_column = self.projected_column_by_generator.get(source_row)
            if dispatch_column is not None:
                projected[dispatch_column] = source[
                    self.source_master.index.dispatch_by_generator[source_row]
                ]
            cost_column = self.cost_column_by_generator.get(source_row)
            if cost_column is not None:
                curve = self.source_master.costs[source_row]
                cost = curve.committed_base_cost * float(self.commitment[position])
                cost += fsum(
                    float(slope) * source[column]
                    for slope, column in zip(
                        curve.segment_slopes_per_mwh,
                        self.source_master.index.segments_by_generator[source_row],
                        strict=True,
                    )
                    if column is not None
                )
                projected[cost_column] = cost
        return projected

    def rebalance_dispatch(
        self,
        projected_values: npt.ArrayLike,
        *,
        total_demand_mw: float,
        backend: str,
    ) -> tuple[FloatArray, dict[str, object]]:
        """Project free dispatch columns onto exact system balance and bounds."""

        projected = np.asarray(projected_values, dtype=np.float64)
        if projected.shape != (self.canonical.num_columns,):
            raise ScopfError("Fixed-commitment balance projection has the wrong shape")
        if not np.all(np.isfinite(projected)):
            raise ScopfError("Fixed-commitment balance projection is nonfinite")
        variable_columns: list[int] = []
        fixed_total = 0.0
        for generator in self.source_master.index.generator_source_rows:
            source_row = int(generator)
            column = self.projected_column_by_generator.get(source_row)
            if column is None:
                fixed_total += float(self.fixed_dispatch_by_generator[source_row])
            else:
                variable_columns.append(column)
        columns = np.asarray(variable_columns, dtype=np.int64)
        target = float(total_demand_mw) - fixed_total
        balanced, audit = project_boxed_sum(
            projected[columns],
            np.asarray(self.canonical.column_lower, dtype=np.float64)[columns],
            np.asarray(self.canonical.column_upper, dtype=np.float64)[columns],
            target_sum=target,
            backend=backend,
        )
        adjusted = projected.copy()
        adjusted[columns] = balanced
        # Reset epigraph columns from the exact source PWL lift.  This removes
        # any cost-row noise carried by the time-limited PDLP iterate.
        exact = self.project_source_values(self.lift(adjusted))
        return exact, {
            **audit,
            "fixed_dispatch_total_mw": fixed_total,
            "full_target_demand_mw": float(total_demand_mw),
            "full_dispatch_sum_mw": float(np.sum(self.dispatch(exact))),
            "cost_epigraph_reset_from_exact_pwl": bool(self.cost_column_by_generator),
            "exact_source_pmin_pmax_retained": True,
        }

    def validate_lift(
        self, projected_values: npt.ArrayLike, *, tolerance_mw: float
    ) -> dict[str, float | bool]:
        projected = np.asarray(projected_values, dtype=np.float64)
        lifted = self.lift(projected)
        projected_violation = max(
            self.canonical.max_row_violation(projected),
            _maximum_column_violation(self.canonical, projected),
        )
        source_violation = max(
            self.source_master.canonical.max_row_violation(lifted),
            _maximum_column_violation(self.source_master.canonical, lifted),
        )
        return {
            "passed": bool(
                projected_violation <= tolerance_mw
                and source_violation <= tolerance_mw
            ),
            "projected_maximum_violation_mw": projected_violation,
            "lifted_source_maximum_violation_mw": source_violation,
        }


def build_fixed_commitment_projection(
    master: ReducedMaster,
    commitment: npt.ArrayLike,
    *,
    include_cost_epigraph: bool = False,
) -> FixedCommitmentProjection:
    """Build the exact dispatch-only feasibility projection of a binary region."""

    binary = np.asarray(commitment, dtype=np.int8)
    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64)
    if binary.shape != (source_rows.size,) or np.any((binary != 0) & (binary != 1)):
        raise ScopfError("Fixed-commitment projection requires a binary commitment")

    projected = CanonicalMILP()
    projected_columns: dict[int, int] = {}
    fixed_dispatch: dict[int, float] = {}
    omitted_off = 0
    omitted_fixed_on = 0
    for position, generator in enumerate(source_rows):
        source_row = int(generator)
        u_column = master.index.commitment_by_generator[source_row]
        expected = float(binary[position])
        if (
            float(master.canonical.column_lower[u_column]) != expected
            or float(master.canonical.column_upper[u_column]) != expected
        ):
            raise ScopfError(
                "Fixed-commitment source master does not fix every commitment exactly"
            )
        curve = master.costs[source_row]
        if not binary[position]:
            fixed_dispatch[source_row] = 0.0
            omitted_off += 1
        elif curve.pmin_mw == curve.pmax_mw:
            fixed_dispatch[source_row] = float(curve.pmin_mw)
            omitted_fixed_on += 1
        else:
            projected_columns[source_row] = projected.add_variable(
                master.canonical.variable_names[
                    master.index.dispatch_by_generator[source_row]
                ],
                lower=float(curve.pmin_mw),
                upper=float(curve.pmax_mw),
            )

    source_lower, source_upper = master.canonical.row_bound_arrays()
    dropped_constant_rows: list[str] = []
    dropped_box_redundant_rows: list[str] = []
    for coupling in master.coupling_rows:
        source_row_index = int(coupling.row_index)
        coefficients: dict[int, float] = {}
        fixed_activity_terms: list[float] = []
        for position, generator in enumerate(source_rows):
            source_row = int(generator)
            coefficient = float(coupling.generator_coefficients[position])
            if coefficient == 0.0:
                continue
            projected_column = projected_columns.get(source_row)
            if projected_column is None:
                fixed_activity_terms.append(
                    coefficient * fixed_dispatch[source_row]
                )
            else:
                coefficients[projected_column] = coefficient
        fixed_activity = fsum(fixed_activity_terms)
        lower = float(source_lower[source_row_index] - fixed_activity)
        upper = float(source_upper[source_row_index] - fixed_activity)
        if not coefficients:
            if lower <= 0.0 <= upper:
                dropped_constant_rows.append(coupling.row_name)
                continue
            raise FixedCommitmentProjectionInfeasible(
                coupling.row_name,
                lower=lower,
                upper=upper,
            )

        minimum_terms: list[float] = []
        maximum_terms: list[float] = []
        for column, coefficient in coefficients.items():
            lo = float(projected.column_lower[column])
            hi = float(projected.column_upper[column])
            minimum_terms.append(coefficient * (lo if coefficient >= 0.0 else hi))
            maximum_terms.append(coefficient * (hi if coefficient >= 0.0 else lo))
        minimum = float(np.nextafter(fsum(minimum_terms), -inf))
        maximum = float(np.nextafter(fsum(maximum_terms), inf))
        if (not isfinite(lower) or lower <= minimum) and (
            not isfinite(upper) or maximum <= upper
        ):
            dropped_box_redundant_rows.append(coupling.row_name)
            continue
        projected.add_row(
            coupling.row_name,
            coefficients,
            lower=lower,
            upper=upper,
        )

    if projected.num_columns == 0:
        raise ScopfError(
            "Fixed-commitment projection has no free dispatch variables"
        )
    if projected.num_rows == 0:
        raise ScopfError("Fixed-commitment projection has no active coupling rows")
    dropped_names = sorted(dropped_constant_rows + dropped_box_redundant_rows)
    cost_columns: dict[int, int] = {}
    cost_epigraph_row_count = 0
    duplicate_cost_line_count = 0
    if include_cost_epigraph:
        for position, generator in enumerate(source_rows):
            source_row = int(generator)
            if not binary[position]:
                continue
            curve = master.costs[source_row]
            widths = np.asarray(curve.segment_widths_mw, dtype=np.float64)
            slopes = np.asarray(curve.segment_slopes_per_mwh, dtype=np.float64)
            if np.any(np.diff(slopes) < -1e-10):
                raise ScopfError("Fixed-commitment cost epigraph requires convex PWL slopes")
            breakpoint_power = np.concatenate(
                (np.asarray([curve.pmin_mw]), curve.pmin_mw + np.cumsum(widths))
            )
            breakpoint_cost = np.concatenate(
                (
                    np.asarray([curve.committed_base_cost]),
                    curve.committed_base_cost + np.cumsum(widths * slopes),
                )
            )
            row = source_row + 1
            dispatch_column = projected_columns.get(source_row)
            if dispatch_column is None:
                fixed_cost = curve.pwl_value(fixed_dispatch[source_row])
                cost_columns[source_row] = projected.add_variable(
                    f"cost_g{row:04d}",
                    objective=1.0,
                    lower=fixed_cost,
                    upper=fixed_cost,
                )
                continue
            cost_column = projected.add_variable(
                f"cost_g{row:04d}",
                objective=1.0,
                lower=float(np.min(breakpoint_cost)),
                upper=float(np.max(breakpoint_cost)),
            )
            cost_columns[source_row] = cost_column
            unique_lines: set[tuple[float, float]] = set()
            for segment, slope in enumerate(slopes, start=1):
                left = float(breakpoint_power[segment - 1])
                left_cost = float(breakpoint_cost[segment - 1])
                intercept = left_cost - float(slope) * left
                signature = (float(slope), intercept)
                if signature in unique_lines:
                    duplicate_cost_line_count += 1
                    continue
                unique_lines.add(signature)
                projected.add_row(
                    f"cost_epigraph_g{row:04d}_s{segment:02d}",
                    {cost_column: 1.0, dispatch_column: -float(slope)},
                    lower=intercept,
                )
                cost_epigraph_row_count += 1
    audit: dict[str, object] = {
        "policy": "exact_binary_commitment_dispatch_projection_v1",
        "source_column_count": master.canonical.num_columns,
        "source_row_count": master.canonical.num_rows,
        "source_local_generator_row_count_removed": (
            master.canonical.num_rows - len(master.coupling_rows)
        ),
        "source_coupling_row_count": len(master.coupling_rows),
        "projected_column_count": projected.num_columns,
        "projected_row_count": projected.num_rows,
        "omitted_off_dispatch_count": omitted_off,
        "omitted_fixed_on_dispatch_count": omitted_fixed_on,
        "removed_commitment_column_count": int(source_rows.size),
        "removed_pwl_segment_column_count": int(
            sum(
                column is not None
                for generator in source_rows
                for column in master.index.segments_by_generator[int(generator)]
            )
        ),
        "dropped_constant_coupling_row_count": len(dropped_constant_rows),
        "dropped_box_redundant_coupling_row_count": len(
            dropped_box_redundant_rows
        ),
        "dropped_coupling_row_name_sha256": hashlib.sha256(
            "\n".join(dropped_names).encode("utf-8")
        ).hexdigest(),
        "exact_source_pmin_pmax_retained": True,
        "pwl_projection_equivalence": (
            "fixed_u_local_generator_polytope_projects_exactly_to_"
            "conditional_source_pmin_pmax_dispatch_interval"
        ),
        "network_and_security_row_source": "unchanged_reduced_master_coupling_rows",
        "cost_epigraph_enabled": bool(include_cost_epigraph),
        "cost_epigraph_column_count": len(cost_columns),
        "cost_epigraph_row_count": cost_epigraph_row_count,
        "duplicate_cost_line_count_removed": duplicate_cost_line_count,
        "cost_epigraph_equivalence": (
            "exact_convex_pwl_perspective_after_binary_commitment_fixing"
            if include_cost_epigraph
            else None
        ),
    }
    return FixedCommitmentProjection(
        canonical=projected,
        source_master=master,
        commitment=binary.copy(),
        projected_column_by_generator=projected_columns,
        fixed_dispatch_by_generator=fixed_dispatch,
        cost_column_by_generator=cost_columns,
        audit=audit,
    )
