"""NVIDIA cuOpt adapter using its supported Python expression API."""

from __future__ import annotations

import hashlib
import time
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from ..canonical import CanonicalMILP
from ..errors import MipStartSolveError, ScopfError
from .common import SolveResult

NATIVE_OPTIMAL_ONLY = "native_optimal_only"
FINITE_BOUND_GAP_CERTIFICATE = (
    "finite_incumbent_bound_gap_and_native_residuals_v1"
)
SUPPORTED_CERTIFICATE_STATUSES = frozenset(
    {"Optimal", "FeasibleFound", "TimeLimit"}
)
INTEGER_ONLY_MIP_START = "integer_only"
FULL_MIP_START = "all_columns"
SUPPORTED_MIP_START_MODES = frozenset(
    {INTEGER_ONLY_MIP_START, FULL_MIP_START}
)
NO_NATIVE_SCALING = "none"
POWER_SYSTEM_PER_UNIT_SCALING = "power_system_per_unit_v1"
POWER_SYSTEM_EQUILIBRATED_SCALING = "power_system_equilibrated_v2"
POWER_SYSTEM_SAFE_EQUILIBRATED_SCALING = "power_system_equilibrated_safe_v3"
SUPPORTED_NATIVE_SCALING_MODES = frozenset(
    {
        NO_NATIVE_SCALING,
        POWER_SYSTEM_PER_UNIT_SCALING,
        POWER_SYSTEM_EQUILIBRATED_SCALING,
        POWER_SYSTEM_SAFE_EQUILIBRATED_SCALING,
    }
)
MIP_START_NATIVE_POLICY = "presolve_off_full_feasible_assignment_readback_v3"
MIP_START_REJECTION_TEXT = "Error cannot add the provided initial solution!"
BARRIER_NUMERICAL_WARNING = (
    "Barrier Solve status A numerical error was encountered."
)
CUOPT_PDLP_PROFILE_KEYS = frozenset(
    {
        "method",
        "solver_mode",
        "precision",
        "batch_strong_branching",
        "batch_reliability_branching",
        "reliability_branching_factor",
    }
)
CUOPT_PDLP_METHODS = {"pdlp": 1}
CUOPT_PDLP_SOLVER_MODES = {"stable3": 4}
CUOPT_PDLP_PRECISIONS = {"fp64": 1}
INCUMBENT_COMMITMENT_TRACE_POLICY = (
    "cuopt_get_solution_callback_distinct_commitment_v1"
)


class IncumbentCommitmentTrace:
    """Compress cuOpt incumbent callbacks into exact commitment transitions."""

    def __init__(
        self,
        canonical_columns: np.ndarray,
        variable_names: list[str],
    ) -> None:
        columns = np.asarray(canonical_columns, dtype=np.int64)
        if columns.ndim != 1 or len(variable_names) != columns.size:
            raise ValueError("Invalid incumbent commitment trace identity")
        self.canonical_columns = [int(value) for value in columns]
        self.variable_names = list(variable_names)
        self.callback_count = 0
        self.same_commitment_callback_count = 0
        self.callback_errors: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self._last_vector: np.ndarray | None = None
        self._unique_fingerprints: set[str] = set()
        self.last_callback_elapsed_seconds: float | None = None
        self.last_commitment_change_elapsed_seconds: float | None = None

    def _normalize(self, commitments: np.ndarray) -> tuple[np.ndarray, float]:
        values = np.asarray(commitments, dtype=np.float64)
        if values.shape != (len(self.canonical_columns),):
            raise ValueError("Incumbent commitment vector has the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("Incumbent commitment vector is nonfinite")
        rounded = np.rint(values)
        maximum_integrality_error = (
            float(np.max(np.abs(values - rounded))) if values.size else 0.0
        )
        if maximum_integrality_error > 1e-5:
            raise ValueError(
                "Incumbent callback returned a nonintegral commitment vector"
            )
        if np.any((rounded < 0.0) | (rounded > 1.0)):
            raise ValueError("Incumbent commitment vector is not binary")
        return rounded.astype(np.int8), maximum_integrality_error

    def _record(
        self,
        commitments: np.ndarray,
        *,
        elapsed_seconds: float,
        objective: float | None,
        bound: float | None,
        source: str,
        callback: bool,
    ) -> None:
        vector, maximum_integrality_error = self._normalize(commitments)
        fingerprint = hashlib.sha256(vector.tobytes()).hexdigest()
        if callback:
            self.callback_count += 1
            self.last_callback_elapsed_seconds = float(elapsed_seconds)
        if self._last_vector is not None and np.array_equal(
            vector, self._last_vector
        ):
            if callback:
                self.same_commitment_callback_count += 1
                self.snapshots[-1]["incumbent_callbacks_for_state"] += 1
            self.snapshots[-1].update(
                {
                    "last_seen_elapsed_seconds": float(elapsed_seconds),
                    "last_seen_objective": objective,
                    "last_seen_bound": bound,
                    "last_seen_source": source,
                }
            )
            return
        previous = self._last_vector
        changed_positions = (
            np.empty(0, dtype=np.int64)
            if previous is None
            else np.flatnonzero(vector != previous)
        )
        changes = [
            {
                "integer_position": int(position),
                "canonical_column": self.canonical_columns[int(position)],
                "variable_name": self.variable_names[int(position)],
                "from_commitment": int(previous[int(position)]),
                "to_commitment": int(vector[int(position)]),
            }
            for position in changed_positions
        ]
        snapshot = {
            "transition": len(self.snapshots) + 1,
            "first_seen_elapsed_seconds": float(elapsed_seconds),
            "last_seen_elapsed_seconds": float(elapsed_seconds),
            "first_seen_objective": objective,
            "last_seen_objective": objective,
            "first_seen_bound": bound,
            "last_seen_bound": bound,
            "first_seen_source": source,
            "last_seen_source": source,
            "commitment_count": int(np.sum(vector)),
            "commitment_fingerprint_sha256": fingerprint,
            "hamming_distance_from_previous_incumbent": (
                None if previous is None else int(changed_positions.size)
            ),
            "off_to_on_from_previous_incumbent": (
                None
                if previous is None
                else int(np.count_nonzero((previous == 0) & (vector == 1)))
            ),
            "on_to_off_from_previous_incumbent": (
                None
                if previous is None
                else int(np.count_nonzero((previous == 1) & (vector == 0)))
            ),
            "maximum_integrality_error": maximum_integrality_error,
            "incumbent_callbacks_for_state": int(callback),
            "changes_from_previous_incumbent": changes,
            "commitment_vector": [int(value) for value in vector],
        }
        self.snapshots.append(snapshot)
        self._last_vector = vector.copy()
        self._unique_fingerprints.add(fingerprint)
        self.last_commitment_change_elapsed_seconds = float(elapsed_seconds)

    def record_callback(
        self,
        commitments: np.ndarray,
        *,
        elapsed_seconds: float,
        objective: float | None,
        bound: float | None,
    ) -> None:
        self._record(
            commitments,
            elapsed_seconds=elapsed_seconds,
            objective=objective,
            bound=bound,
            source="cuopt_incumbent_callback",
            callback=True,
        )

    def record_callback_error(
        self, exc: Exception, *, elapsed_seconds: float
    ) -> None:
        self.callback_count += 1
        self.last_callback_elapsed_seconds = float(elapsed_seconds)
        self.callback_errors.append(
            {
                "elapsed_seconds": float(elapsed_seconds),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

    def finalize(
        self,
        final_commitments: np.ndarray | None,
        *,
        solve_time_seconds: float,
        objective: float | None,
        bound: float | None,
    ) -> dict[str, Any]:
        callback_fingerprint = (
            None
            if self._last_vector is None
            else hashlib.sha256(self._last_vector.tobytes()).hexdigest()
        )
        if final_commitments is not None:
            self._record(
                final_commitments,
                elapsed_seconds=float(solve_time_seconds),
                objective=objective,
                bound=bound,
                source="solver_return",
                callback=False,
            )
        final_fingerprint = (
            None
            if self._last_vector is None
            else hashlib.sha256(self._last_vector.tobytes()).hexdigest()
        )
        stabilization_window = (
            None
            if self.last_commitment_change_elapsed_seconds is None
            else max(
                0.0,
                float(solve_time_seconds)
                - self.last_commitment_change_elapsed_seconds,
            )
        )
        return {
            "enabled": True,
            "policy": INCUMBENT_COMMITMENT_TRACE_POLICY,
            "canonical_integer_columns": self.canonical_columns,
            "integer_variable_names": self.variable_names,
            "callback_count": self.callback_count,
            "callback_errors": self.callback_errors,
            "commitment_transition_count": len(self.snapshots),
            "unique_commitment_count": len(self._unique_fingerprints),
            "same_commitment_callback_count": (
                self.same_commitment_callback_count
            ),
            "last_callback_elapsed_seconds": (
                self.last_callback_elapsed_seconds
            ),
            "last_commitment_change_elapsed_seconds": (
                self.last_commitment_change_elapsed_seconds
            ),
            "stabilization_window_seconds_at_solver_return": (
                stabilization_window
            ),
            "final_commitment_fingerprint_sha256": final_fingerprint,
            "solver_return_matches_last_callback": (
                final_fingerprint is not None
                and callback_fingerprint == final_fingerprint
            ),
            "complete": final_fingerprint is not None and not self.callback_errors,
            "snapshots": self.snapshots,
        }


def normalize_cuopt_pdlp_profile(
    profile: dict[str, object] | None,
) -> dict[str, int]:
    """Map the registered descriptive PDLP profile to exact cuOpt parameters."""

    if not profile:
        return {}
    observed_keys = frozenset(profile)
    if observed_keys != CUOPT_PDLP_PROFILE_KEYS:
        missing = sorted(CUOPT_PDLP_PROFILE_KEYS - observed_keys)
        unknown = sorted(observed_keys - CUOPT_PDLP_PROFILE_KEYS)
        raise ScopfError(
            "cuOpt PDLP profile must specify the exact registered key set; "
            f"missing={missing}, unknown={unknown}"
        )
    method = str(profile["method"])
    solver_mode = str(profile["solver_mode"])
    precision = str(profile["precision"])
    if method not in CUOPT_PDLP_METHODS:
        raise ScopfError(f"Unsupported cuOpt PDLP method: {method!r}")
    if solver_mode not in CUOPT_PDLP_SOLVER_MODES:
        raise ScopfError(f"Unsupported cuOpt PDLP solver mode: {solver_mode!r}")
    if precision not in CUOPT_PDLP_PRECISIONS:
        raise ScopfError(f"Unsupported cuOpt PDLP precision: {precision!r}")
    for key in ("batch_strong_branching", "batch_reliability_branching"):
        if not isinstance(profile[key], bool):
            raise ScopfError(f"cuOpt PDLP profile {key} must be boolean")
    reliability_factor = profile["reliability_branching_factor"]
    if not isinstance(reliability_factor, int) or isinstance(
        reliability_factor, bool
    ):
        raise ScopfError(
            "cuOpt PDLP reliability_branching_factor must be an integer"
        )
    if reliability_factor != 1:
        raise ScopfError(
            "This frozen cuOpt PDLP profile requires reliability branching factor 1"
        )
    return {
        "method": CUOPT_PDLP_METHODS[method],
        "pdlp_solver_mode": CUOPT_PDLP_SOLVER_MODES[solver_mode],
        "pdlp_precision": CUOPT_PDLP_PRECISIONS[precision],
        "mip_batch_pdlp_strong_branching": int(
            profile["batch_strong_branching"]
        ),
        "mip_batch_pdlp_reliability_branching": int(
            profile["batch_reliability_branching"]
        ),
        "mip_reliability_branching": reliability_factor,
    }


def _native(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _callback_metric(value: object) -> float | None:
    """Normalize finite cuOpt callback metrics and discard native sentinels."""

    observed = float(value)
    return observed if np.isfinite(observed) and abs(observed) < 1e19 else None


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
    finite_objective = bool(objective is not None and np.isfinite(objective))
    finite_bound = bool(bound is not None and np.isfinite(bound))
    finite_reported_gap = bool(
        reported_gap is not None and np.isfinite(reported_gap)
    )
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
    reported_gap_meets_request = bool(
        finite_reported_gap and float(reported_gap) <= gap_limit
    )
    calculated_gap_meets_request = bool(
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


def prepare_mip_start(
    mip_start_values: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    integrality: np.ndarray,
    mode: str,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
    clip_to_bounds: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact cuOpt start columns/values with integer entries normalized."""

    if mode not in SUPPORTED_MIP_START_MODES:
        raise ScopfError(f"Unknown cuOpt MIP start mode: {mode!r}")
    candidate = np.asarray(mip_start_values, dtype=np.float64)
    if candidate.shape != expected_shape:
        raise ScopfError(
            "cuOpt MIP start has the wrong vector shape: "
            f"expected {expected_shape}, observed {candidate.shape}"
        )
    columns = (
        np.flatnonzero(integrality)
        if mode == INTEGER_ONLY_MIP_START
        else np.arange(candidate.size, dtype=np.int64)
    )
    values = candidate[columns].copy()
    integer_positions = np.flatnonzero(integrality[columns])
    values[integer_positions] = np.rint(values[integer_positions])
    if clip_to_bounds:
        if lower_bounds is None or upper_bounds is None:
            raise ScopfError("cuOpt MIP-start bound projection requires both bound arrays")
        lower = np.asarray(lower_bounds, dtype=np.float64)
        upper = np.asarray(upper_bounds, dtype=np.float64)
        if lower.shape != expected_shape or upper.shape != expected_shape:
            raise ScopfError("cuOpt MIP-start bound arrays have the wrong vector shape")
        values = np.maximum(values, lower[columns])
        values = np.minimum(values, upper[columns])
    if not np.all(np.isfinite(values)):
        raise ScopfError("cuOpt MIP start contains a nonfinite selected value")
    return columns, values


def validate_full_mip_start_feasibility(
    model: CanonicalMILP,
    values: np.ndarray,
    *,
    column_scale: np.ndarray,
    row_scale: np.ndarray,
    tolerance: float,
) -> dict[str, object]:
    """Fail closed before cuOpt sees a full native MIP assignment.

    The pinned cuOpt 26.6 expression API assembles variable-level starts into a
    complete assignment and rejects incomplete or infeasible assignments.  The
    check is performed in the exact diagonally scaled native coordinates used
    by the adapter, so the tolerance has the same interpretation as the native
    solver residual tolerance.
    """

    candidate = np.asarray(values, dtype=np.float64)
    if candidate.shape != (model.num_columns,) or not np.all(np.isfinite(candidate)):
        raise ScopfError("cuOpt full MIP start has invalid shape or values")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ScopfError("cuOpt full MIP-start tolerance must be finite and nonnegative")
    columns = np.asarray(column_scale, dtype=np.float64)
    rows = np.asarray(row_scale, dtype=np.float64)
    if columns.shape != candidate.shape or rows.shape != (model.num_rows,):
        raise ScopfError("cuOpt full MIP-start scaling vectors have invalid shapes")
    if (
        not np.all(np.isfinite(columns))
        or not np.all(columns > 0.0)
        or not np.all(np.isfinite(rows))
        or not np.all(rows > 0.0)
    ):
        raise ScopfError("cuOpt full MIP-start scaling vectors are invalid")

    _objective, lower, upper, integrality = model.column_arrays()
    native_values = candidate / columns
    native_lower = lower / columns
    native_upper = upper / columns
    column_violation = float(
        max(
            np.max(native_lower - native_values),
            np.max(native_values - native_upper),
            0.0,
        )
    )
    integer_columns = np.flatnonzero(integrality)
    integrality_violation = (
        0.0
        if not integer_columns.size
        else float(
            np.max(
                np.abs(
                    candidate[integer_columns]
                    - np.rint(candidate[integer_columns])
                )
            )
        )
    )
    activity = np.asarray(model.matrix_csr() @ candidate, dtype=np.float64)
    row_lower, row_upper = model.row_bound_arrays()
    native_activity = activity * rows
    native_row_lower = row_lower * rows
    native_row_upper = row_upper * rows
    row_violation = float(
        max(
            np.max(native_row_lower - native_activity),
            np.max(native_activity - native_row_upper),
            0.0,
        )
    )
    passed = bool(
        column_violation <= tolerance
        and integrality_violation <= tolerance
        and row_violation <= tolerance
    )
    audit = {
        "policy": "complete_native_feasible_assignment_required_v1",
        "passed": passed,
        "tolerance": float(tolerance),
        "maximum_native_column_bound_violation": column_violation,
        "maximum_integrality_violation": integrality_violation,
        "maximum_native_row_violation": row_violation,
        "canonical_values_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
    }
    if not passed:
        raise ScopfError(
            "cuOpt 26.6 full MIP start is not a complete feasible native "
            f"assignment: {audit}"
        )
    return audit


def audit_cuopt_native_log(native_log: str) -> dict[str, object]:
    """Fail closed on native cuOpt errors and retain known warning counts."""

    if not native_log.strip():
        raise ScopfError("cuOpt native log capture was empty")
    lines = [line.strip() for line in native_log.splitlines() if line.strip()]
    rejection_lines = [
        line for line in lines if MIP_START_REJECTION_TEXT in line
    ]
    if rejection_lines:
        raise MipStartSolveError(
            "cuOpt rejected the submitted MIP start: " + rejection_lines[0]
        )
    error_lines = [line for line in lines if line.startswith("Error ")]
    if error_lines:
        raise ScopfError("cuOpt native log reported an error: " + error_lines[0])
    return {
        "sha256": hashlib.sha256(native_log.encode("utf-8")).hexdigest(),
        "bytes": len(native_log.encode("utf-8")),
        "mip_start_rejection_count": 0,
        "barrier_numerical_warning_count": native_log.count(
            BARRIER_NUMERICAL_WARNING
        ),
        "free_variable_warning_count": native_log.count("Free variable found!"),
        "presolve_disabled_message_count": native_log.count(
            "Presolve is disabled, skipping"
        ),
    }


def audit_mip_start_readback(
    *,
    columns: np.ndarray,
    expected_native_values: np.ndarray,
    native_initial_primal: np.ndarray,
    total_columns: int,
    presolve_readback: int | None,
) -> dict[str, object]:
    """Verify the exact explicit native-space start before the pinned solver call."""

    selected = np.asarray(columns, dtype=np.int64)
    expected = np.asarray(expected_native_values, dtype=np.float64)
    observed = np.asarray(native_initial_primal, dtype=np.float64)
    if not selected.size:
        return {
            "policy": MIP_START_NATIVE_POLICY,
            "submitted": False,
            "contract_passed": None,
            "presolve_parameter_readback": presolve_readback,
            "native_translated_vector_readback": False,
        }
    if observed.shape != (total_columns,):
        raise ScopfError(
            "cuOpt MIP-start native-space readback has the wrong shape: "
            f"expected {(total_columns,)}, observed {observed.shape}"
        )
    if expected.shape != selected.shape:
        raise ScopfError("cuOpt MIP-start expected-value shape changed")
    if presolve_readback != 0:
        raise ScopfError(
            "cuOpt MIP starts require presolve=0 in the pinned 26.6.0 API"
        )
    if not np.array_equal(observed[selected], expected):
        raise ScopfError("cuOpt MIP-start native-space value readback failed")
    unselected = np.ones(total_columns, dtype=bool)
    unselected[selected] = False
    if not np.all(np.isnan(observed[unselected])):
        raise ScopfError("cuOpt MIP-start readback populated unselected columns")
    return {
        "policy": MIP_START_NATIVE_POLICY,
        "submitted": True,
        "contract_passed": True,
        "presolve_parameter_readback": 0,
        "native_translated_vector_readback": True,
        "submitted_columns": int(selected.size),
    }


def native_scaling_vectors(
    model: CanonicalMILP,
    *,
    mode: str,
    base_mva: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an exact diagonal variable/row reformulation for cuOpt only."""

    if mode not in SUPPORTED_NATIVE_SCALING_MODES:
        raise ScopfError(f"Unknown cuOpt native scaling mode: {mode!r}")
    if not np.isfinite(base_mva) or base_mva <= 0:
        raise ScopfError("cuOpt native scaling requires a positive finite base MVA")
    column_scale = np.ones(model.num_columns, dtype=np.float64)
    row_scale = np.ones(model.num_rows, dtype=np.float64)
    if mode == NO_NATIVE_SCALING:
        return column_scale, row_scale
    for column, name in enumerate(model.variable_names):
        if name.startswith(("pg_", "pseg_", "flow_")):
            column_scale[column] = float(base_mva)
    integrality = model.column_arrays()[3]
    if np.any(column_scale[integrality > 0] != 1.0):
        raise ScopfError("cuOpt native scaling cannot rescale integer columns")
    for row, name in enumerate(model.row_names):
        indices, coefficients = model.row_entries(row)
        if indices and all(
            model.variable_names[index].startswith("theta_") for index in indices
        ):
            continue
        if name.startswith("dc_flow_"):
            theta_coefficients = [
                abs(float(coefficient) * column_scale[index])
                for index, coefficient in zip(indices, coefficients, strict=True)
                if model.variable_names[index].startswith("theta_")
            ]
            if not theta_coefficients or max(theta_coefficients) == 0.0:
                raise ScopfError(f"Cannot scale malformed DC-flow row {name}")
            row_scale[row] = 1.0 / max(theta_coefficients)
        else:
            row_scale[row] = 1.0 / float(base_mva)
    if mode in {
        POWER_SYSTEM_EQUILIBRATED_SCALING,
        POWER_SYSTEM_SAFE_EQUILIBRATED_SCALING,
    }:
        row_lower, row_upper = model.row_bound_arrays()
        for row in range(model.num_rows):
            indices, coefficients = model.row_entries(row)
            native_coefficients = [
                abs(float(coefficient) * column_scale[index] * row_scale[row])
                for index, coefficient in zip(indices, coefficients, strict=True)
            ]
            magnitudes = native_coefficients + [
                abs(float(bound) * row_scale[row])
                for bound in (row_lower[row], row_upper[row])
                if np.isfinite(bound)
            ]
            maximum = max(magnitudes, default=0.0)
            if maximum > 0.0:
                # A positive diagonal row transformation is mathematically
                # exact.  v2 normalizes in both directions.  The safe v3 mode
                # only strengthens rows that are below unit scale; it never
                # reduces the established per-unit row multiplier.  Thus a
                # native absolute residual tolerance cannot map back to a
                # weaker canonical tolerance than under v1.
                if mode == POWER_SYSTEM_EQUILIBRATED_SCALING:
                    factor = float(np.clip(1.0 / maximum, 1e-6, 1e6))
                else:
                    factor = float(min(max(1.0 / maximum, 1.0), 1e6))
                row_scale[row] *= factor
    if (
        not np.all(np.isfinite(column_scale))
        or not np.all(column_scale > 0)
        or not np.all(np.isfinite(row_scale))
        or not np.all(row_scale > 0)
    ):
        raise ScopfError("cuOpt native scaling produced an invalid diagonal")
    return column_scale, row_scale


def _nonzero_range(entries: np.ndarray) -> dict[str, float | None]:
    absolute = np.abs(np.asarray(entries, dtype=np.float64))
    nonzero = absolute[np.isfinite(absolute) & (absolute > 0.0)]
    if not nonzero.size:
        return {"minimum_nonzero": None, "maximum": None, "ratio": None}
    minimum = float(np.min(nonzero))
    maximum = float(np.max(nonzero))
    return {
        "minimum_nonzero": minimum,
        "maximum": maximum,
        "ratio": maximum / minimum,
    }


def free_continuous_columns(
    lower: np.ndarray,
    upper: np.ndarray,
    integrality: np.ndarray,
) -> np.ndarray:
    """Return columns requiring an explicit exact native positive/negative split."""

    lower_values = np.asarray(lower, dtype=np.float64)
    upper_values = np.asarray(upper, dtype=np.float64)
    integer_values = np.asarray(integrality)
    if not (
        lower_values.shape == upper_values.shape == integer_values.shape
    ):
        raise ScopfError("cuOpt free-column audit arrays have different shapes")
    free = np.isneginf(lower_values) & np.isposinf(upper_values)
    if np.any(integer_values[free] > 0):
        raise ScopfError("cuOpt explicit free-column splitting supports continuous columns only")
    return np.flatnonzero(free)


def fixed_or_unused_columns(model: CanonicalMILP) -> np.ndarray:
    """Return canonical columns cuOpt otherwise removes before applying a start."""

    objective, lower, upper, _ = model.column_arrays()
    column_nonzeros = np.diff(model.matrix_csr().tocsc().indptr)
    fixed = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    unused_zero_cost = (column_nonzeros == 0) & (objective == 0.0)
    return np.flatnonzero(fixed | unused_zero_cost)


def native_scaling_audit(
    model: CanonicalMILP,
    values: np.ndarray,
    *,
    mode: str,
    base_mva: float,
) -> dict[str, object]:
    """Quantify conditioning and verify the diagonal reformulation algebra."""

    canonical_values = np.asarray(values, dtype=np.float64)
    if canonical_values.shape != (model.num_columns,):
        raise ScopfError("cuOpt native-scaling audit has the wrong vector shape")
    if not np.all(np.isfinite(canonical_values)):
        raise ScopfError("cuOpt native-scaling audit received nonfinite values")
    column_scale, row_scale = native_scaling_vectors(
        model, mode=mode, base_mva=base_mva
    )
    matrix = model.matrix_csr()
    native_matrix = matrix.multiply(column_scale).multiply(row_scale[:, None]).tocsr()
    native_values = canonical_values / column_scale
    canonical_activity = np.asarray(matrix @ canonical_values, dtype=np.float64)
    native_activity = np.asarray(native_matrix @ native_values, dtype=np.float64)
    objective, column_lower, column_upper, _ = model.column_arrays()
    row_lower, row_upper = model.row_bound_arrays()
    native_objective = objective * column_scale
    native_lower = row_lower * row_scale
    native_upper = row_upper * row_scale

    canonical_lower_violation = np.where(
        np.isfinite(row_lower), row_lower - canonical_activity, -np.inf
    )
    canonical_upper_violation = np.where(
        np.isfinite(row_upper), canonical_activity - row_upper, -np.inf
    )
    canonical_row_violation = np.maximum(
        canonical_lower_violation, canonical_upper_violation
    )
    native_lower_violation = np.where(
        np.isfinite(native_lower), native_lower - native_activity, -np.inf
    )
    native_upper_violation = np.where(
        np.isfinite(native_upper), native_activity - native_upper, -np.inf
    )
    canonicalized_native_violation = np.maximum(
        native_lower_violation, native_upper_violation
    ) / row_scale
    finite = np.isfinite(canonical_row_violation) & np.isfinite(
        canonicalized_native_violation
    )
    violation_error = (
        float(
            np.max(
                np.abs(
                    canonicalized_native_violation[finite]
                    - canonical_row_violation[finite]
                )
            )
        )
        if np.any(finite)
        else 0.0
    )
    return {
        "mode": mode,
        "base_mva": float(base_mva),
        "scaled_columns": int(np.count_nonzero(column_scale != 1.0)),
        "scaled_rows": int(np.count_nonzero(row_scale != 1.0)),
        "raw_matrix_coefficients": _nonzero_range(matrix.data),
        "native_matrix_coefficients": _nonzero_range(native_matrix.data),
        "raw_objective_coefficients": _nonzero_range(objective),
        "native_objective_coefficients": _nonzero_range(native_objective),
        "raw_finite_variable_bounds": _nonzero_range(
            np.concatenate((column_lower, column_upper))
        ),
        "native_finite_variable_bounds": _nonzero_range(
            np.concatenate((column_lower / column_scale, column_upper / column_scale))
        ),
        "raw_finite_row_bounds": _nonzero_range(
            np.concatenate((row_lower, row_upper))
        ),
        "native_finite_row_bounds": _nonzero_range(
            np.concatenate((native_lower, native_upper))
        ),
        "maximum_native_activity_identity_error": float(
            np.max(np.abs(native_activity - row_scale * canonical_activity))
        ),
        "maximum_canonicalized_row_violation_identity_error": violation_error,
        "maximum_value_round_trip_error": float(
            np.max(np.abs(native_values * column_scale - canonical_values))
        ),
        "objective_identity_error": abs(
            float(native_objective @ native_values) - float(objective @ canonical_values)
        ),
    }


def solve_cuopt(
    model: CanonicalMILP,
    *,
    time_limit_seconds: float,
    mip_relative_gap: float,
    threads: int = 0,
    mip_start_values: np.ndarray | None = None,
    mip_start_mode: str = INTEGER_ONLY_MIP_START,
    clip_mip_start_to_bounds: bool = False,
    native_scaling_mode: str = NO_NATIVE_SCALING,
    native_base_mva: float = 100.0,
    log_to_console: bool = False,
    cuopt_pdlp_profile: dict[str, object] | None = None,
    track_incumbent_commitments: bool = False,
    mip_acceptance_policy: str = NATIVE_OPTIMAL_ONLY,
    mip_certificate_residual_tolerance: float = 1e-6,
    mip_heuristics_only: bool = False,
) -> SolveResult:
    if mip_acceptance_policy not in {
        NATIVE_OPTIMAL_ONLY,
        FINITE_BOUND_GAP_CERTIFICATE,
    }:
        raise ScopfError(
            f"Unknown cuOpt MIP acceptance policy: {mip_acceptance_policy!r}"
        )
    if mip_start_values is not None and mip_start_mode != FULL_MIP_START:
        raise ScopfError(
            "cuOpt 26.6 does not reliably accept an unextended partial MIP "
            "start; first obtain a complete feasible continuous extension "
            "and submit it with mip_start_mode='all_columns'"
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
    column_scale, row_scale = native_scaling_vectors(
        model,
        mode=native_scaling_mode,
        base_mva=float(native_base_mva),
    )
    native_objective = objective * column_scale
    native_lower = lower / column_scale
    native_upper = upper / column_scale
    mip_start_columns = np.empty(0, dtype=np.int64)
    mip_start_selected_values = np.empty(0, dtype=np.float64)
    mip_start_bound_projection_count = 0
    mip_start_bound_projection_maximum_delta = 0.0
    mip_start_feasibility_audit: dict[str, object] | None = None
    if mip_start_values is not None:
        mip_start_columns, mip_start_selected_values = prepare_mip_start(
            mip_start_values,
            expected_shape=objective.shape,
            integrality=integrality,
            mode=mip_start_mode,
            lower_bounds=lower,
            upper_bounds=upper,
            clip_to_bounds=clip_mip_start_to_bounds,
        )
        unprojected = np.asarray(mip_start_values, dtype=np.float64)[
            mip_start_columns
        ].copy()
        integer_positions = np.flatnonzero(integrality[mip_start_columns])
        unprojected[integer_positions] = np.rint(unprojected[integer_positions])
        projection_delta = np.abs(mip_start_selected_values - unprojected)
        mip_start_bound_projection_count = int(np.count_nonzero(projection_delta))
        if projection_delta.size:
            mip_start_bound_projection_maximum_delta = float(
                np.max(projection_delta)
            )
        mip_start_feasibility_audit = validate_full_mip_start_feasibility(
            model,
            mip_start_selected_values,
            column_scale=column_scale,
            row_scale=row_scale,
            tolerance=float(mip_certificate_residual_tolerance),
        )
    explicit_free_split = bool(mip_start_columns.size)
    split_columns = (
        free_continuous_columns(native_lower, native_upper, integrality)
        if explicit_free_split
        else np.empty(0, dtype=np.int64)
    )
    eliminated_columns = (
        fixed_or_unused_columns(model)
        if explicit_free_split
        else np.empty(0, dtype=np.int64)
    )
    split_column_set = {int(index) for index in split_columns}
    eliminated_column_set = {int(index) for index in eliminated_columns}
    selected_native_value_by_column = {
        int(index): float(value / column_scale[int(index)])
        for index, value in zip(
            mip_start_columns, mip_start_selected_values, strict=True
        )
    }
    eliminated_native_value_by_column: dict[int, float] = {}
    for index in eliminated_columns:
        canonical_index = int(index)
        if (
            isfinite(float(native_lower[canonical_index]))
            and native_lower[canonical_index] == native_upper[canonical_index]
        ):
            native_value = float(native_lower[canonical_index])
            selected_value = selected_native_value_by_column.get(canonical_index)
            if selected_value is not None and not np.isclose(
                selected_value, native_value, rtol=0.0, atol=1e-9
            ):
                raise ScopfError(
                    "cuOpt MIP start conflicts with an eliminated fixed column: "
                    f"{model.variable_names[canonical_index]}"
                )
        elif canonical_index in selected_native_value_by_column:
            native_value = selected_native_value_by_column[canonical_index]
        elif isfinite(float(native_lower[canonical_index])):
            native_value = float(native_lower[canonical_index])
        elif isfinite(float(native_upper[canonical_index])):
            native_value = float(native_upper[canonical_index])
        else:
            native_value = 0.0
        eliminated_native_value_by_column[canonical_index] = native_value
    problem = Problem("activsg_preventive_scuc")
    native_variables = []
    column_components = []
    for index, name in enumerate(model.variable_names):
        if index in eliminated_column_set:
            column_components.append([])
        elif index in split_column_set:
            positive = problem.addVariable(
                lb=0.0,
                ub=float("inf"),
                obj=float(native_objective[index]),
                vtype=CONTINUOUS,
                name=f"{name}__native_positive",
            )
            positive_index = len(native_variables)
            native_variables.append(positive)
            negative = problem.addVariable(
                lb=0.0,
                ub=float("inf"),
                obj=-float(native_objective[index]),
                vtype=CONTINUOUS,
                name=f"{name}__native_negative",
            )
            negative_index = len(native_variables)
            native_variables.append(negative)
            column_components.append(
                [
                    (positive, 1.0, positive_index),
                    (negative, -1.0, negative_index),
                ]
            )
        else:
            variable = problem.addVariable(
                lb=float(native_lower[index]),
                ub=float(native_upper[index]),
                obj=float(native_objective[index]),
                vtype=INTEGER if integrality[index] else CONTINUOUS,
                name=name,
            )
            native_index = len(native_variables)
            native_variables.append(variable)
            column_components.append([(variable, 1.0, native_index)])
    native_mip_start_columns: list[int] = []
    native_mip_start_values: list[float] = []
    for index, value in zip(
        mip_start_columns, mip_start_selected_values, strict=True
    ):
        canonical_index = int(index)
        native_value = float(value / column_scale[canonical_index])
        components = column_components[canonical_index]
        if not components:
            continue
        component_values = (
            (max(native_value, 0.0), max(-native_value, 0.0))
            if canonical_index in split_column_set
            else (native_value,)
        )
        for (variable, _, native_index), component_value in zip(
            components, component_values, strict=True
        ):
            variable.setMIPStart(component_value)
            native_mip_start_columns.append(native_index)
            native_mip_start_values.append(component_value)
    objective_columns = np.flatnonzero(native_objective)
    objective_variables = []
    objective_coefficients: list[float] = []
    objective_constant = 0.0
    for index in objective_columns:
        if int(index) in eliminated_column_set:
            objective_constant += float(native_objective[int(index)]) * (
                eliminated_native_value_by_column[int(index)]
            )
            continue
        for variable, multiplier, _ in column_components[int(index)]:
            objective_variables.append(variable)
            objective_coefficients.append(
                float(native_objective[int(index)]) * multiplier
            )
    objective_expression = LinearExpression(
        objective_variables,
        objective_coefficients,
        objective_constant,
    )
    problem.setObjective(objective_expression, sense=MINIMIZE)
    row_lower, row_upper = model.row_bound_arrays()
    native_constraint_count = 0
    for row, name in enumerate(model.row_names):
        indices, coefficients = model.row_entries(row)
        expression_variables = []
        native_coefficients: list[float] = []
        expression_constant = 0.0
        for index, coefficient in zip(indices, coefficients, strict=True):
            scaled_coefficient = (
                float(coefficient) * column_scale[index] * row_scale[row]
            )
            if index in eliminated_column_set:
                expression_constant += scaled_coefficient * (
                    eliminated_native_value_by_column[index]
                )
                continue
            for variable, multiplier, _ in column_components[index]:
                expression_variables.append(variable)
                native_coefficients.append(scaled_coefficient * multiplier)
        expression = LinearExpression(
            expression_variables, native_coefficients, 0.0
        )
        lo = float(row_lower[row] * row_scale[row] - expression_constant)
        hi = float(row_upper[row] * row_scale[row] - expression_constant)
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
    pdlp_settings = normalize_cuopt_pdlp_profile(cuopt_pdlp_profile)
    settings = SolverSettings()
    settings.set_parameter("time_limit", float(time_limit_seconds))
    settings.set_parameter("mip_relative_gap", float(mip_relative_gap))
    settings.set_parameter("random_seed", 0)
    settings.set_parameter("log_to_console", bool(log_to_console))
    settings.set_parameter("mip_heuristics_only", bool(mip_heuristics_only))
    mip_heuristics_only_readback = bool(
        _native(settings.get_parameter("mip_heuristics_only"))
    )
    if mip_heuristics_only_readback != bool(mip_heuristics_only):
        raise ScopfError(
            "cuOpt MIP heuristics-only parameter readback mismatch: "
            f"requested={bool(mip_heuristics_only)}, "
            f"observed={mip_heuristics_only_readback}"
        )
    mip_start_presolve_readback: int | None = None
    if mip_start_columns.size:
        settings.set_parameter("presolve", 0)
        mip_start_presolve_readback = int(
            _native(settings.get_parameter("presolve"))
        )
    if threads > 0:
        settings.set_parameter("num_cpu_threads", int(threads))
    for name, value in pdlp_settings.items():
        settings.set_parameter(name, value)
    pdlp_settings_readback = {
        name: int(_native(settings.get_parameter(name)))
        for name in pdlp_settings
    }
    if pdlp_settings_readback != pdlp_settings:
        raise ScopfError(
            "cuOpt PDLP parameter readback did not match the requested profile: "
            f"requested={pdlp_settings}, observed={pdlp_settings_readback}"
        )
    if mip_start_columns.size:
        problem._to_data_model()
        native_initial_primal = np.asarray(
            problem.model.get_initial_primal_solution(), dtype=np.float64
        )
    else:
        native_initial_primal = np.empty(0, dtype=np.float64)
    native_mip_start_column_array = np.asarray(
        native_mip_start_columns, dtype=np.int64
    )
    native_mip_start_value_array = np.asarray(
        native_mip_start_values, dtype=np.float64
    )
    mip_start_contract = audit_mip_start_readback(
        columns=native_mip_start_column_array,
        expected_native_values=native_mip_start_value_array,
        native_initial_primal=native_initial_primal,
        total_columns=len(native_variables),
        presolve_readback=mip_start_presolve_readback,
    )
    mip_start_contract["submitted_canonical_columns"] = int(
        mip_start_columns.size
    )
    mip_start_contract["submitted_native_columns"] = int(
        native_mip_start_column_array.size
    )
    mip_start_contract["explicit_free_split_columns"] = int(
        split_columns.size
    )
    mip_start_contract["explicit_eliminated_columns"] = int(
        eliminated_columns.size
    )
    mip_start_contract["full_feasibility_precheck"] = (
        mip_start_feasibility_audit
    )
    mip_start_contract["eliminated_submitted_canonical_columns"] = int(
        sum(
            int(index) in eliminated_column_set
            for index in mip_start_columns
        )
    )
    incumbent_trace: IncumbentCommitmentTrace | None = None
    incumbent_callback = None
    incumbent_callback_clock: dict[str, float | None] = {"started": None}
    if track_incumbent_commitments:
        from cuopt.linear_programming.internals import GetSolutionCallback

        integer_columns = np.flatnonzero(integrality)
        incumbent_trace = IncumbentCommitmentTrace(
            integer_columns,
            [model.variable_names[int(index)] for index in integer_columns],
        )

        def extract_commitments(solution: object) -> np.ndarray:
            commitments: list[float] = []
            for index in integer_columns:
                canonical_index = int(index)
                components = column_components[canonical_index]
                native_value = (
                    eliminated_native_value_by_column[canonical_index]
                    if not components
                    else sum(
                        multiplier * float(solution[native_index])
                        for _, multiplier, native_index in components
                    )
                )
                commitments.append(
                    native_value * float(column_scale[canonical_index])
                )
            return np.asarray(commitments, dtype=np.float64)

        callback_user_data = {
            "policy": INCUMBENT_COMMITMENT_TRACE_POLICY,
        }

        class CommitmentCallback(GetSolutionCallback):
            def __init__(self) -> None:
                super().__init__()

            def get_solution(
                self,
                solution: object,
                solution_cost: object,
                solution_bound: object,
                user_data: object,
            ) -> None:
                started = incumbent_callback_clock["started"]
                elapsed = 0.0 if started is None else time.perf_counter() - started
                try:
                    if user_data is not callback_user_data:
                        raise ValueError("cuOpt changed callback user data")
                    incumbent_trace.record_callback(
                        extract_commitments(solution),
                        elapsed_seconds=elapsed,
                        objective=_callback_metric(solution_cost[0]),
                        bound=_callback_metric(solution_bound[0]),
                    )
                except Exception as exc:  # pragma: no cover - native callback
                    incumbent_trace.record_callback_error(
                        exc, elapsed_seconds=elapsed
                    )

        incumbent_callback = CommitmentCallback()
        settings.set_mip_callback(incumbent_callback, callback_user_data)
    with NamedTemporaryFile(
        mode="w", prefix="activsg-cuopt-native-", suffix=".log", delete=False
    ) as native_log_stream:
        native_log_path = Path(native_log_stream.name)
    settings.set_parameter("log_file", str(native_log_path))
    native_log = ""
    callback_solve_wall_time_seconds: float | None = None
    try:
        incumbent_callback_clock["started"] = time.perf_counter()
        problem.solve(settings)
        callback_solve_wall_time_seconds = (
            time.perf_counter() - incumbent_callback_clock["started"]
        )
        native_log = native_log_path.read_text(encoding="utf-8", errors="replace")
    finally:
        native_log_path.unlink(missing_ok=True)
    native_log_audit = audit_cuopt_native_log(native_log)
    if mip_start_columns.size:
        mip_start_contract["native_rejection_detected"] = False
        mip_start_contract["native_log_contract_passed"] = True
    status = problem.Status.name
    stats = problem.SolutionStats
    has_incumbent = (
        status in {"Optimal", "FeasibleFound", "TimeLimit"}
        and np.isfinite(problem.ObjValue)
        and all(np.isfinite(variable.getValue()) for variable in native_variables)
    )
    canonical_native_values = None
    if has_incumbent:
        canonical_native_values = np.asarray(
            [
                (
                    eliminated_native_value_by_column[index]
                    if not components
                    else sum(
                        multiplier * float(variable.getValue())
                        for variable, multiplier, _ in components
                    )
                )
                for index, components in enumerate(column_components)
            ],
            dtype=np.float64,
        )
    values = (
        None
        if canonical_native_values is None
        else canonical_native_values * column_scale
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
    incumbent_trace_payload: dict[str, Any] = {
        "enabled": False,
        "policy": INCUMBENT_COMMITMENT_TRACE_POLICY,
    }
    if incumbent_trace is not None:
        if callback_solve_wall_time_seconds is None:
            raise ScopfError("cuOpt incumbent trace has no solver wall time")
        incumbent_trace_payload = incumbent_trace.finalize(
            (
                None
                if values is None
                else values[np.flatnonzero(integrality)]
            ),
            solve_time_seconds=callback_solve_wall_time_seconds,
            objective=objective_value,
            bound=bound_value,
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
            "mip_start_mode": mip_start_mode,
            "mip_start_columns": int(mip_start_columns.size),
            "mip_start_bound_projection_enabled": bool(
                clip_mip_start_to_bounds
            ),
            "mip_start_bound_projection_count": (
                mip_start_bound_projection_count
            ),
            "mip_start_bound_projection_maximum_delta": (
                mip_start_bound_projection_maximum_delta
            ),
            "partial_integer_mip_start_columns": int(
                np.count_nonzero(integrality[mip_start_columns])
            ),
            "mip_start_native_contract": mip_start_contract,
            "native_log_audit": native_log_audit,
            "canonical_columns_translated": model.num_columns,
            "canonical_rows_translated": model.num_rows,
            "canonical_nonzeros_translated": int(model.matrix_csr().nnz),
            "native_constraints_added": native_constraint_count,
            "native_columns_translated": len(native_variables),
            "native_free_variable_split_columns": int(split_columns.size),
            "native_fixed_or_unused_columns_eliminated": int(
                eliminated_columns.size
            ),
            "native_explicit_free_variable_split": explicit_free_split,
            "native_scaling_mode": native_scaling_mode,
            "native_base_mva": float(native_base_mva),
            "native_column_scale_minimum": float(np.min(column_scale)),
            "native_column_scale_maximum": float(np.max(column_scale)),
            "native_row_scale_minimum": float(np.min(row_scale)),
            "native_row_scale_maximum": float(np.max(row_scale)),
            "log_to_console": bool(log_to_console),
            "cuopt_pdlp_profile": dict(cuopt_pdlp_profile or {}),
            "cuopt_pdlp_parameters_requested": pdlp_settings,
            "cuopt_pdlp_parameters_readback": pdlp_settings_readback,
            "incumbent_commitment_trace": incumbent_trace_payload,
            "mip_heuristics_only_requested": bool(mip_heuristics_only),
            "mip_heuristics_only_readback": mip_heuristics_only_readback,
            "dual_bound_authority_disabled_by_policy": bool(mip_heuristics_only),
        },
    )
