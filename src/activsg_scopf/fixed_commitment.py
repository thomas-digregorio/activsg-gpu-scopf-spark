"""Exact dispatch-only projection for fixed-commitment feasibility polishing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import fsum, inf, isfinite

import numpy as np
import numpy.typing as npt

from .canonical import CanonicalMILP
from .errors import ScopfError
from .reduced import ReducedMaster

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


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
    master: ReducedMaster, commitment: npt.ArrayLike
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
    }
    return FixedCommitmentProjection(
        canonical=projected,
        source_master=master,
        commitment=binary.copy(),
        projected_column_by_generator=projected_columns,
        fixed_dispatch_by_generator=fixed_dispatch,
        audit=audit,
    )
