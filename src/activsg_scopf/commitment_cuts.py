"""Replayable Phase-I feasibility cuts in commitment space.

A fixed-commitment dispatch Phase I supplies a box-constrained dual lower
bound.  Reintroducing every source-online generator's conditional
``[PMIN * u, PMAX * u]`` interval turns the same dual into a globally valid
linear necessary condition on the binary commitment.  This is a classical
Benders/Farkas feasibility cut, evaluated here with the same conservative
floating-point safety margin as the underlying Phase-I certificate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Any

import numpy as np
import numpy.typing as npt

from .errors import ScopfError
from .phase_one import phase_one_semantic_row_key, replay_phase_one_certificate
from .reduced import ReducedMaster

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int8]


@dataclass(frozen=True)
class CommitmentFeasibilityCut:
    """One valid upper inequality ``coefficients @ u <= rhs``."""

    cut_id: str
    coefficients: FloatArray
    rhs: float
    source_commitment_sha256: str
    conservative_source_violation_pu: float

    def validate(self, generator_count: int) -> None:
        if self.coefficients.shape != (generator_count,) or not np.all(
            np.isfinite(self.coefficients)
        ):
            raise ScopfError("Commitment feasibility cut has invalid coefficients")
        if not isfinite(self.rhs):
            raise ScopfError("Commitment feasibility cut has a nonfinite RHS")
        if not isfinite(self.conservative_source_violation_pu):
            raise ScopfError("Commitment feasibility cut has a nonfinite source violation")

    def violation(self, commitment: npt.ArrayLike) -> float:
        binary = np.asarray(commitment, dtype=np.int8)
        self.validate(binary.size)
        if np.any((binary != 0) & (binary != 1)):
            raise ScopfError("Commitment feasibility cut requires a binary commitment")
        return float(self.coefficients @ binary - self.rhs)

    def as_dict(self, generator_source_rows: npt.ArrayLike) -> dict[str, Any]:
        rows = np.asarray(generator_source_rows, dtype=np.int64)
        self.validate(rows.size)
        canonical_coefficients = np.where(self.coefficients == 0.0, 0.0, self.coefficients).astype(
            np.float64, copy=False
        )
        nonzero = np.flatnonzero(canonical_coefficients != 0.0)
        return {
            "certificate_kind": "phase_one_binary_benders_feasibility_cut_v1",
            "cut_id": self.cut_id,
            "source_commitment_sha256": self.source_commitment_sha256,
            "rhs": self.rhs,
            "conservative_source_violation_pu": (self.conservative_source_violation_pu),
            "generator_coefficient_count": int(rows.size),
            "nonzero_generator_coefficient_count": int(nonzero.size),
            "generator_coefficients": [
                {
                    "source_row": int(rows[position]) + 1,
                    "coefficient_pu": float(canonical_coefficients[position]),
                }
                for position in nonzero
            ],
            "coefficient_sha256": hashlib.sha256(canonical_coefficients.tobytes()).hexdigest(),
            "validity": (
                "necessary_for_zero_violation_dispatch_with_exact_conditional_source_pmin_pmax"
            ),
            "exact_source_pmin_pmax_changed": False,
        }


@dataclass(frozen=True)
class CommitmentCapacityCut:
    """One row-wise necessary condition induced by conditional PMIN/PMAX.

    If a dispatch row is ``a @ p <= b``, then every feasible commitment must
    satisfy ``sum(min(a_g PMIN_g, a_g PMAX_g) u_g) <= b``.  A lower row uses
    the analogous maximum activity and is negated into this common upper form.
    The serialized source-row identity lets the independent checker rebuild
    the inequality directly from raw inputs without trusting solver output.
    """

    cut_id: str
    coefficients: FloatArray
    rhs: float
    source_commitment_sha256: str
    conservative_source_violation_pu: float
    source_row_name: str
    source_row_side: str
    source_row_bound_pu: float
    outward_rhs_relaxation_pu: float

    def validate(self, generator_count: int) -> None:
        if self.coefficients.shape != (generator_count,) or not np.all(
            np.isfinite(self.coefficients)
        ):
            raise ScopfError("Commitment capacity cut has invalid coefficients")
        if self.source_row_side not in {"upper", "lower"}:
            raise ScopfError("Commitment capacity cut has an invalid row side")
        if not self.source_row_name:
            raise ScopfError("Commitment capacity cut has an empty source row")
        if not all(
            isfinite(value)
            for value in (
                self.rhs,
                self.conservative_source_violation_pu,
                self.source_row_bound_pu,
                self.outward_rhs_relaxation_pu,
            )
        ):
            raise ScopfError("Commitment capacity cut has a nonfinite scalar")
        if self.outward_rhs_relaxation_pu < 0.0:
            raise ScopfError("Commitment capacity cut has a negative relaxation")

    def violation(self, commitment: npt.ArrayLike) -> float:
        binary = np.asarray(commitment, dtype=np.int8)
        self.validate(binary.size)
        if np.any((binary != 0) & (binary != 1)):
            raise ScopfError("Commitment capacity cut requires a binary commitment")
        return float(self.coefficients @ binary - self.rhs)

    def as_dict(self, generator_source_rows: npt.ArrayLike) -> dict[str, Any]:
        rows = np.asarray(generator_source_rows, dtype=np.int64)
        self.validate(rows.size)
        canonical_coefficients = np.where(
            self.coefficients == 0.0, 0.0, self.coefficients
        ).astype(np.float64, copy=False)
        nonzero = np.flatnonzero(canonical_coefficients != 0.0)
        return {
            "certificate_kind": "conditional_dispatch_row_capacity_cut_v1",
            "cut_id": self.cut_id,
            "source_commitment_sha256": self.source_commitment_sha256,
            "source_row_name": self.source_row_name,
            "source_row_side": self.source_row_side,
            "source_row_bound_pu": self.source_row_bound_pu,
            "rhs": self.rhs,
            "outward_rhs_relaxation_pu": self.outward_rhs_relaxation_pu,
            "conservative_source_violation_pu": (
                self.conservative_source_violation_pu
            ),
            "generator_coefficient_count": int(rows.size),
            "nonzero_generator_coefficient_count": int(nonzero.size),
            "generator_coefficients": [
                {
                    "source_row": int(rows[position]),
                    "coefficient_pu": float(canonical_coefficients[position]),
                }
                for position in nonzero
            ],
            "coefficient_sha256": hashlib.sha256(
                canonical_coefficients.tobytes()
            ).hexdigest(),
            "validity": (
                "necessary_row_activity_envelope_with_exact_conditional_source_pmin_pmax"
            ),
            "exact_source_pmin_pmax_changed": False,
        }


@dataclass(frozen=True)
class CommitmentCoverCut:
    """A binary knapsack-cover inequality implied by another valid cut.

    A mixed-sign parent inequality ``a @ u <= b`` is rewritten with
    ``z_g = u_g`` for positive ``a_g`` and ``z_g = 1 - u_g`` for negative
    ``a_g``.  This gives ``sum(abs(a_g) z_g) <= B``.  Any subset whose
    weight exceeds ``B`` cannot have every ``z_g`` equal to one, yielding a
    unit-coefficient cover inequality.  Unlike the parent projection cut,
    that cover can strengthen the continuous commitment relaxation.
    """

    cut_id: str
    coefficients: FloatArray
    rhs: float
    source_cut_id: str
    source_cut_kind: str
    cover_source_rows: tuple[int, ...]
    complemented_source_rows: tuple[int, ...]
    transformed_rhs: float
    cover_weight_sum: float
    conservative_cover_excess: float
    cover_safety_margin: float
    source_commitment_sha256: str
    separation_reference_sha256: str
    separation_reference_violation: float
    derivation_strategy: str
    core_cover_source_rows: tuple[int, ...] = ()
    extended_source_rows: tuple[int, ...] = ()
    extension_weight_threshold: float | None = None

    def validate(self, generator_count: int) -> None:
        if self.coefficients.shape != (generator_count,) or not np.all(
            np.isfinite(self.coefficients)
        ):
            raise ScopfError("Commitment cover cut has invalid coefficients")
        nonzero = np.flatnonzero(self.coefficients != 0.0)
        if nonzero.size == 0 or not np.all(
            np.isin(self.coefficients[nonzero], (-1.0, 1.0))
        ):
            raise ScopfError("Commitment cover cut coefficients are not signed unit values")
        if tuple(sorted(self.cover_source_rows)) != self.cover_source_rows or len(
            set(self.cover_source_rows)
        ) != len(self.cover_source_rows):
            raise ScopfError("Commitment cover cut source rows are not unique and sorted")
        if len(self.cover_source_rows) != int(nonzero.size):
            raise ScopfError("Commitment cover cut source-row identity is inconsistent")
        if tuple(sorted(self.complemented_source_rows)) != self.complemented_source_rows:
            raise ScopfError("Commitment cover cut complemented rows are not sorted")
        if not set(self.complemented_source_rows).issubset(self.cover_source_rows):
            raise ScopfError("Commitment cover cut complements a row outside its cover")
        core_rows = (
            self.core_cover_source_rows
            if self.core_cover_source_rows
            else self.cover_source_rows
        )
        if (
            not core_rows
            or tuple(sorted(core_rows)) != core_rows
            or len(set(core_rows)) != len(core_rows)
            or not set(core_rows).issubset(self.cover_source_rows)
        ):
            raise ScopfError("Commitment cover cut core identity is invalid")
        expected_extended = tuple(
            row for row in self.cover_source_rows if row not in set(core_rows)
        )
        if self.extended_source_rows != expected_extended:
            raise ScopfError("Commitment cover cut extension identity is inconsistent")
        if bool(self.core_cover_source_rows) != bool(self.extended_source_rows):
            raise ScopfError("Commitment cover cut core/extension representation is invalid")
        if self.extended_source_rows:
            if (
                self.extension_weight_threshold is None
                or not isfinite(self.extension_weight_threshold)
                or self.extension_weight_threshold <= 0.0
            ):
                raise ScopfError("Extended cover cut has an invalid weight threshold")
        elif self.extension_weight_threshold is not None:
            raise ScopfError("Basic cover cut unexpectedly records an extension threshold")
        expected_rhs = float(
            len(core_rows) - 1 - len(self.complemented_source_rows)
        )
        if self.rhs != expected_rhs:
            raise ScopfError("Commitment cover cut RHS is inconsistent")
        scalars = (
            self.rhs,
            self.transformed_rhs,
            self.cover_weight_sum,
            self.conservative_cover_excess,
            self.cover_safety_margin,
            self.separation_reference_violation,
        )
        if not all(isfinite(value) for value in scalars):
            raise ScopfError("Commitment cover cut has a nonfinite scalar")
        if self.cover_safety_margin < 0.0 or self.conservative_cover_excess <= 0.0:
            raise ScopfError("Commitment cover cut lacks a positive conservative excess")
        if not self.source_cut_id or not self.source_cut_kind or not self.derivation_strategy:
            raise ScopfError("Commitment cover cut lacks derivation identity")
        for digest in (
            self.source_commitment_sha256,
            self.separation_reference_sha256,
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ScopfError("Commitment cover cut has an invalid SHA-256 identity")

    def violation(self, commitment: npt.ArrayLike) -> float:
        values = np.asarray(commitment, dtype=np.float64)
        self.validate(values.size)
        if values.shape != self.coefficients.shape or not np.all(np.isfinite(values)):
            raise ScopfError("Commitment cover cut received invalid commitment values")
        return fsum(
            [-self.rhs]
            + [
                float(self.coefficients[position] * values[position])
                for position in np.flatnonzero(self.coefficients)
            ]
        )

    def as_dict(self, generator_source_rows: npt.ArrayLike) -> dict[str, Any]:
        rows = np.asarray(generator_source_rows, dtype=np.int64)
        self.validate(rows.size)
        nonzero = np.flatnonzero(self.coefficients != 0.0)
        observed_cover = tuple(int(row) for row in rows[nonzero])
        observed_complemented = tuple(
            int(rows[position])
            for position in nonzero
            if self.coefficients[position] < 0.0
        )
        if observed_cover != self.cover_source_rows:
            raise ScopfError("Commitment cover cut source-row replay mismatch")
        if observed_complemented != self.complemented_source_rows:
            raise ScopfError("Commitment cover cut complement replay mismatch")
        extended = bool(self.extended_source_rows)
        payload = {
            "certificate_kind": (
                "binary_knapsack_extended_cover_from_commitment_cut_v2"
                if extended
                else "binary_knapsack_cover_from_commitment_cut_v1"
            ),
            "cut_id": self.cut_id,
            "source_cut_id": self.source_cut_id,
            "source_cut_kind": self.source_cut_kind,
            "cover_source_rows": list(self.cover_source_rows),
            "complemented_source_rows": list(self.complemented_source_rows),
            "rhs": self.rhs,
            "transformed_rhs": self.transformed_rhs,
            "cover_weight_sum": self.cover_weight_sum,
            "conservative_cover_excess": self.conservative_cover_excess,
            "cover_safety_margin": self.cover_safety_margin,
            "source_commitment_sha256": self.source_commitment_sha256,
            "separation_reference_sha256": self.separation_reference_sha256,
            "separation_reference_violation": self.separation_reference_violation,
            "derivation_strategy": self.derivation_strategy,
            "coefficient_sha256": hashlib.sha256(self.coefficients.tobytes()).hexdigest(),
            "validity": (
                "binary_knapsack_extended_cover_implied_by_replayable_commitment_upper_cut"
                if extended
                else "binary_knapsack_cover_implied_by_replayable_commitment_upper_cut"
            ),
            "exact_source_pmin_pmax_changed": False,
        }
        if extended:
            payload.update(
                {
                    "core_cover_source_rows": list(self.core_cover_source_rows),
                    "extended_source_rows": list(self.extended_source_rows),
                    "extension_weight_threshold": self.extension_weight_threshold,
                }
            )
        return payload


@dataclass(frozen=True)
class CommitmentCardinalityCut:
    """One branch inequality on an integer-valued commitment subset.

    Every source commitment is binary, so ``sum(u[g] for g in S)`` is an
    integer for any deterministic subset ``S``.  A fractional LP value ``v``
    therefore admits the exhaustive disjunction ``sum(u[S]) <= floor(v)`` or
    ``sum(u[S]) >= ceil(v)``.  The lower branch is stored in the common upper
    inequality form by negating its coefficients and RHS.
    """

    cut_id: str
    coefficients: FloatArray
    rhs: float
    subset_id: str
    subset_source_rows: tuple[int, ...]
    branch_side: str
    integer_threshold: int

    def validate(self, generator_count: int) -> None:
        if self.coefficients.shape != (generator_count,) or not np.all(
            np.isfinite(self.coefficients)
        ):
            raise ScopfError("Commitment cardinality cut has invalid coefficients")
        if self.branch_side not in {"at_most", "at_least"}:
            raise ScopfError("Commitment cardinality cut has an invalid branch side")
        nonzero = np.flatnonzero(self.coefficients != 0.0)
        if nonzero.size == 0:
            raise ScopfError("Commitment cardinality cut has an empty subset")
        expected_sign = 1.0 if self.branch_side == "at_most" else -1.0
        if not np.all(self.coefficients[nonzero] == expected_sign):
            raise ScopfError("Commitment cardinality cut coefficients are not unit signed")
        if len(self.subset_source_rows) != int(nonzero.size):
            raise ScopfError("Commitment cardinality cut subset identity is inconsistent")
        if tuple(sorted(self.subset_source_rows)) != self.subset_source_rows:
            raise ScopfError("Commitment cardinality cut source rows are not sorted")
        if len(set(self.subset_source_rows)) != len(self.subset_source_rows):
            raise ScopfError("Commitment cardinality cut source rows are duplicated")
        expected_rhs = (
            float(self.integer_threshold)
            if self.branch_side == "at_most"
            else -float(self.integer_threshold)
        )
        if not isfinite(self.rhs) or self.rhs != expected_rhs:
            raise ScopfError("Commitment cardinality cut RHS is inconsistent")

    def as_dict(self, generator_source_rows: npt.ArrayLike) -> dict[str, Any]:
        rows = np.asarray(generator_source_rows, dtype=np.int64)
        self.validate(rows.size)
        nonzero = np.flatnonzero(self.coefficients != 0.0)
        observed_rows = tuple(int(row) for row in rows[nonzero])
        if observed_rows != self.subset_source_rows:
            raise ScopfError("Commitment cardinality cut source-row replay mismatch")
        return {
            "certificate_kind": "binary_commitment_cardinality_branch_v1",
            "cut_id": self.cut_id,
            "subset_id": self.subset_id,
            "subset_source_rows": list(self.subset_source_rows),
            "branch_side": self.branch_side,
            "integer_threshold": self.integer_threshold,
            "rhs": self.rhs,
            "coefficient_sha256": hashlib.sha256(self.coefficients.tobytes()).hexdigest(),
            "validity": "integer_sum_disjunction_over_source_binary_commitments",
            "exact_source_pmin_pmax_changed": False,
        }


type CommitmentUpperCut = (
    CommitmentFeasibilityCut
    | CommitmentCapacityCut
    | CommitmentCoverCut
    | CommitmentCardinalityCut
)


def build_commitment_cardinality_cut(
    *,
    generator_source_rows: npt.ArrayLike,
    subset_positions: npt.ArrayLike,
    subset_id: str,
    branch_side: str,
    integer_threshold: int,
) -> CommitmentCardinalityCut:
    """Build a deterministic upper-form cardinality branch inequality."""

    rows = np.asarray(generator_source_rows, dtype=np.int64)
    positions = np.asarray(subset_positions, dtype=np.int64)
    if rows.ndim != 1 or positions.ndim != 1 or positions.size == 0:
        raise ScopfError("Commitment cardinality subset has an invalid shape")
    positions = np.unique(positions)
    if positions[0] < 0 or positions[-1] >= rows.size:
        raise ScopfError("Commitment cardinality subset position is out of range")
    if branch_side not in {"at_most", "at_least"}:
        raise ScopfError("Commitment cardinality branch side is invalid")
    selected_rows = tuple(sorted(int(row) for row in rows[positions]))
    coefficients = np.zeros(rows.size, dtype=np.float64)
    sign = 1.0 if branch_side == "at_most" else -1.0
    coefficients[positions] = sign
    rhs = float(integer_threshold) if branch_side == "at_most" else -float(integer_threshold)
    identity = json.dumps(
        {
            "subset_id": subset_id,
            "subset_source_rows": selected_rows,
            "branch_side": branch_side,
            "integer_threshold": int(integer_threshold),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    cut = CommitmentCardinalityCut(
        cut_id="cc_" + hashlib.sha256(identity).hexdigest()[:24],
        coefficients=coefficients,
        rhs=rhs,
        subset_id=str(subset_id),
        subset_source_rows=selected_rows,
        branch_side=branch_side,
        integer_threshold=int(integer_threshold),
    )
    cut.validate(rows.size)
    return cut


def _commitment_cut_certificate_kind(cut: CommitmentUpperCut) -> str:
    if isinstance(cut, CommitmentFeasibilityCut):
        return "phase_one_binary_benders_feasibility_cut_v1"
    if isinstance(cut, CommitmentCapacityCut):
        return "conditional_dispatch_row_capacity_cut_v1"
    if isinstance(cut, CommitmentCoverCut):
        return (
            "binary_knapsack_extended_cover_from_commitment_cut_v2"
            if cut.extended_source_rows
            else "binary_knapsack_cover_from_commitment_cut_v1"
        )
    if isinstance(cut, CommitmentCardinalityCut):
        return "binary_commitment_cardinality_branch_v1"
    raise ScopfError("Unknown commitment upper-cut type")


def derive_binary_knapsack_cover_cut(
    *,
    source_cut: CommitmentFeasibilityCut | CommitmentCapacityCut,
    generator_source_rows: npt.ArrayLike,
    source_commitment: npt.ArrayLike | None,
    separation_reference: npt.ArrayLike | None = None,
    extend_cover: bool = False,
) -> tuple[CommitmentCoverCut, dict[str, Any]]:
    """Strengthen one valid binary upper inequality with a cover cut.

    The returned inequality is valid for every binary point satisfying the
    source cut.  Candidate cover selection may use a fractional GPU LP point,
    but that point affects only which valid cover is selected, never validity.
    """

    rows = np.asarray(generator_source_rows, dtype=np.int64)
    source_cut.validate(rows.size)
    if rows.ndim != 1 or len(set(int(row) for row in rows)) != int(rows.size):
        raise ScopfError("Binary cover derivation received invalid source identity")
    binary: IntArray | None = None
    if source_commitment is not None:
        binary = np.asarray(source_commitment, dtype=np.int8)
        if binary.shape != (rows.size,) or np.any((binary != 0) & (binary != 1)):
            raise ScopfError("Binary cover derivation received an invalid commitment")
        source_digest = hashlib.sha256(binary.tobytes()).hexdigest()
        if source_digest != source_cut.source_commitment_sha256:
            raise ScopfError("Binary cover source commitment does not match its parent cut")
    else:
        source_digest = source_cut.source_commitment_sha256
        if separation_reference is None:
            raise ScopfError(
                "Binary cover without a source commitment requires a separation reference"
            )

    if separation_reference is None:
        assert binary is not None
        reference = binary.astype(np.float64)
    else:
        reference = np.asarray(separation_reference, dtype=np.float64)
        if reference.shape != (rows.size,) or not np.all(np.isfinite(reference)):
            raise ScopfError("Binary cover separation reference has invalid shape or values")
        reference_tolerance = 1e-7
        if np.any(reference < -reference_tolerance) or np.any(
            reference > 1.0 + reference_tolerance
        ):
            raise ScopfError("Binary cover separation reference lies outside [0, 1]")
        reference = np.clip(reference, 0.0, 1.0)

    parent = np.asarray(source_cut.coefficients, dtype=np.float64)
    positive = parent > 0.0
    negative = parent < 0.0
    support = np.flatnonzero(positive | negative)
    if support.size == 0:
        raise ScopfError("Binary cover parent cut has empty support")
    weights = np.abs(parent)
    transformed_rhs = fsum(
        [float(source_cut.rhs)]
        + [-float(parent[position]) for position in np.flatnonzero(negative)]
    )
    total_weight = fsum(float(weights[position]) for position in support)
    safety_margin = 64.0 * np.finfo(np.float64).eps * max(
        1.0, abs(transformed_rhs), total_weight
    )
    source_z: FloatArray | None = None
    if binary is not None:
        source_z = np.where(positive, binary, 1 - binary).astype(np.float64)
        source_weight = fsum(
            float(weights[position] * source_z[position]) for position in support
        )
        if source_weight - transformed_rhs <= safety_margin:
            raise ScopfError(
                "Binary cover parent is not conservatively violated by its source"
            )
    reference_z = np.where(positive, reference, 1.0 - reference)
    deficits = 1.0 - reference_z

    def candidate_from_order(
        ordered_positions: list[int], strategy: str
    ) -> tuple[tuple[int, ...], str] | None:
        selected: list[int] = []
        selected_weight = 0.0
        for position in ordered_positions:
            if weights[position] <= 0.0:
                continue
            selected.append(position)
            selected_weight = fsum(float(weights[index]) for index in selected)
            if selected_weight - transformed_rhs > safety_margin:
                break
        if selected_weight - transformed_rhs <= safety_margin:
            return None
        # Removing a member always strengthens the fractional separation.  Try
        # the largest reference deficits first while preserving a conservative
        # strict cover.  Stable source-row ties make the result deterministic.
        removal_order = sorted(
            selected,
            key=lambda position: (
                -float(deficits[position]),
                float(weights[position]),
                int(rows[position]),
            ),
        )
        for position in removal_order:
            if position not in selected:
                continue
            trial = [index for index in selected if index != position]
            trial_weight = fsum(float(weights[index]) for index in trial)
            if trial_weight - transformed_rhs > safety_margin:
                selected = trial
        return tuple(sorted(selected)), strategy

    orderings: list[tuple[list[int], str]] = [
        (
            sorted(
                (int(position) for position in support),
                key=lambda position: (
                    float(deficits[position]) / float(weights[position]),
                    float(deficits[position]),
                    -float(weights[position]),
                    int(rows[position]),
                ),
            ),
            "fractional_deficit_per_weight_v1",
        ),
        (
            sorted(
                (int(position) for position in support),
                key=lambda position: (
                    -float(reference_z[position]),
                    -float(weights[position]),
                    int(rows[position]),
                ),
            ),
            "fractional_value_then_weight_v1",
        ),
    ]
    if source_z is not None:
        source_active = [
            int(position)
            for position in support
            if source_z[int(position)] == 1.0
        ]
        orderings.insert(
            0,
            (
                sorted(
                    source_active,
                    key=lambda position: (
                        -float(weights[position]),
                        int(rows[position]),
                    ),
                ),
                "source_active_largest_weight_then_reference_minimal_v1",
            ),
        )
    candidates: list[tuple[float, int, float, str, tuple[int, ...]]] = []
    for ordering, strategy in orderings:
        candidate = candidate_from_order(ordering, strategy)
        if candidate is None:
            continue
        positions, candidate_strategy = candidate
        reference_cover_violation = 1.0 - fsum(
            float(deficits[position]) for position in positions
        )
        candidate_weight = fsum(float(weights[position]) for position in positions)
        candidates.append(
            (
                reference_cover_violation,
                -len(positions),
                candidate_weight - transformed_rhs,
                candidate_strategy,
                positions,
            )
        )
    if not candidates:
        raise ScopfError("Binary cover derivation could not construct a strict cover")
    _, _, raw_excess, strategy, selected_positions = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    core = np.asarray(selected_positions, dtype=np.int64)
    cover_weight_sum = fsum(float(weights[position]) for position in core)
    conservative_excess = cover_weight_sum - transformed_rhs - safety_margin
    if conservative_excess <= 0.0:
        raise ScopfError("Binary cover lost its strict excess after safety margin")

    extension_weight_threshold: float | None = None
    extended_positions = np.empty(0, dtype=np.int64)
    if extend_cover:
        extension_weight_threshold = float(np.max(weights[core]))
        core_set = set(int(position) for position in core)
        extended_positions = np.asarray(
            [
                int(position)
                for position in support
                if int(position) not in core_set
                and float(weights[int(position)]) >= extension_weight_threshold
            ],
            dtype=np.int64,
        )
    selected = np.asarray(
        sorted(
            set(int(position) for position in core)
            | set(int(position) for position in extended_positions)
        ),
        dtype=np.int64,
    )
    coefficients = np.zeros(rows.size, dtype=np.float64)
    coefficients[selected[positive[selected]]] = 1.0
    coefficients[selected[negative[selected]]] = -1.0
    negative_count = int(np.count_nonzero(coefficients < 0.0))
    rhs = float(core.size - 1 - negative_count)
    cover_rows = tuple(int(rows[position]) for position in selected)
    core_rows = tuple(int(rows[position]) for position in core)
    extended_rows = tuple(int(rows[position]) for position in extended_positions)
    extended_cover = bool(extended_rows)
    complemented_rows = tuple(
        int(rows[position]) for position in selected if negative[position]
    )
    reference_sha = hashlib.sha256(reference.astype(np.float64).tobytes()).hexdigest()
    identity_fields: dict[str, object] = {
        "source_cut_id": source_cut.cut_id,
        "cover_source_rows": cover_rows,
        "complemented_source_rows": complemented_rows,
    }
    if extended_cover:
        identity_fields.update(
            {
                "core_cover_source_rows": core_rows,
                "extended_source_rows": extended_rows,
            }
        )
    identity = json.dumps(
        identity_fields,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    cut = CommitmentCoverCut(
        cut_id="kc_" + hashlib.sha256(identity).hexdigest()[:24],
        coefficients=coefficients,
        rhs=rhs,
        source_cut_id=source_cut.cut_id,
        source_cut_kind=_commitment_cut_certificate_kind(source_cut),
        cover_source_rows=cover_rows,
        complemented_source_rows=complemented_rows,
        transformed_rhs=transformed_rhs,
        cover_weight_sum=cover_weight_sum,
        conservative_cover_excess=conservative_excess,
        cover_safety_margin=safety_margin,
        source_commitment_sha256=source_digest,
        separation_reference_sha256=reference_sha,
        separation_reference_violation=0.0,
        derivation_strategy=(
            f"{strategy}__weight_dominance_extended_cover_v2"
            if extended_cover
            else strategy
        ),
        core_cover_source_rows=core_rows if extended_cover else (),
        extended_source_rows=extended_rows if extended_cover else (),
        extension_weight_threshold=(
            extension_weight_threshold if extended_rows else None
        ),
    )
    reference_violation = cut.violation(reference)
    cut = CommitmentCoverCut(
        **{
            **cut.__dict__,
            "separation_reference_violation": reference_violation,
        }
    )
    cut.validate(rows.size)
    verification = verify_binary_knapsack_cover_derivation(source_cut, cut, rows)
    return cut, {
        "derivation": (
            "mixed_sign_binary_knapsack_weight_dominance_extended_cover_v2"
            if extended_rows
            else "mixed_sign_binary_knapsack_cover_v1"
        ),
        "source_cut_id": source_cut.cut_id,
        "source_cut_kind": _commitment_cut_certificate_kind(source_cut),
        "source_parent_violation_pu": (
            None if binary is None else source_cut.violation(binary)
        ),
        "source_cover_violation": None if binary is None else cut.violation(binary),
        "separation_reference_violation": reference_violation,
        "verification": verification,
        "cut": cut.as_dict(rows),
    }


def verify_binary_knapsack_cover_derivation(
    source_cut: CommitmentFeasibilityCut | CommitmentCapacityCut,
    cover_cut: CommitmentCoverCut,
    generator_source_rows: npt.ArrayLike,
) -> dict[str, Any]:
    """Replay a cover's strict-weight proof against its source inequality."""

    rows = np.asarray(generator_source_rows, dtype=np.int64)
    source_cut.validate(rows.size)
    cover_cut.validate(rows.size)
    if cover_cut.source_cut_id != source_cut.cut_id or (
        cover_cut.source_cut_kind != _commitment_cut_certificate_kind(source_cut)
    ):
        raise ScopfError("Binary cover references the wrong parent cut")
    by_row = {int(row): position for position, row in enumerate(rows)}
    if any(row not in by_row for row in cover_cut.cover_source_rows):
        raise ScopfError("Binary cover references an unknown generator row")
    selected = np.asarray(
        [by_row[row] for row in cover_cut.cover_source_rows], dtype=np.int64
    )
    core_rows = (
        cover_cut.core_cover_source_rows
        if cover_cut.core_cover_source_rows
        else cover_cut.cover_source_rows
    )
    core = np.asarray([by_row[row] for row in core_rows], dtype=np.int64)
    extended = np.asarray(
        [by_row[row] for row in cover_cut.extended_source_rows], dtype=np.int64
    )
    parent = np.asarray(source_cut.coefficients, dtype=np.float64)
    if np.any(parent[selected] == 0.0):
        raise ScopfError("Binary cover selects a zero-weight parent coefficient")
    expected_complemented = tuple(
        int(rows[position]) for position in selected if parent[position] < 0.0
    )
    if expected_complemented != cover_cut.complemented_source_rows:
        raise ScopfError("Binary cover complement signs do not match its parent")
    expected_coefficients = np.zeros(rows.size, dtype=np.float64)
    expected_coefficients[selected[parent[selected] > 0.0]] = 1.0
    expected_coefficients[selected[parent[selected] < 0.0]] = -1.0
    if not np.array_equal(expected_coefficients, cover_cut.coefficients):
        raise ScopfError("Binary cover coefficients do not replay from its parent")
    negative = np.flatnonzero(parent < 0.0)
    transformed_rhs = fsum(
        [float(source_cut.rhs)]
        + [-float(parent[position]) for position in negative]
    )
    total_weight = fsum(float(abs(parent[position])) for position in np.flatnonzero(parent))
    safety_margin = 64.0 * np.finfo(np.float64).eps * max(
        1.0, abs(transformed_rhs), total_weight
    )
    cover_weight_sum = fsum(float(abs(parent[position])) for position in core)
    conservative_excess = cover_weight_sum - transformed_rhs - safety_margin
    tolerance = 8.0 * np.finfo(np.float64).eps * max(
        1.0, abs(transformed_rhs), cover_weight_sum, total_weight
    )
    for observed, expected, label in (
        (cover_cut.transformed_rhs, transformed_rhs, "transformed RHS"),
        (cover_cut.cover_weight_sum, cover_weight_sum, "cover weight"),
        (cover_cut.cover_safety_margin, safety_margin, "cover safety margin"),
        (
            cover_cut.conservative_cover_excess,
            conservative_excess,
            "conservative cover excess",
        ),
    ):
        if abs(observed - expected) > tolerance:
            raise ScopfError(f"Binary cover {label} failed replay")
    if conservative_excess <= 0.0:
        raise ScopfError("Binary cover is not strictly valid after numerical margin")
    if extended.size:
        threshold = float(np.max(np.abs(parent[core])))
        if cover_cut.extension_weight_threshold != threshold:
            raise ScopfError("Extended cover weight threshold failed replay")
        if np.any(np.abs(parent[extended]) < threshold):
            raise ScopfError("Extended cover contains a nondominating item")
    return {
        "passed": True,
        "source_cut_id": source_cut.cut_id,
        "cover_size": int(selected.size),
        "core_cover_size": int(core.size),
        "extended_item_count": int(extended.size),
        "transformed_rhs": transformed_rhs,
        "cover_weight_sum": cover_weight_sum,
        "cover_safety_margin": safety_margin,
        "conservative_cover_excess": conservative_excess,
        "original_binary_feasible_set_changed": False,
        "continuous_relaxation_strengthened": True,
    }


def commitment_cardinality_cut_from_record(
    record: dict[str, Any], generator_source_rows: npt.ArrayLike
) -> CommitmentCardinalityCut:
    """Rebuild one serialized cardinality cut from stable source-row identity."""

    rows = np.asarray(generator_source_rows, dtype=np.int64)
    by_row = {int(row): position for position, row in enumerate(rows)}
    serialized_rows = tuple(int(row) for row in record["subset_source_rows"])
    if any(row not in by_row for row in serialized_rows):
        raise ScopfError("Serialized cardinality cut references an unknown generator")
    cut = build_commitment_cardinality_cut(
        generator_source_rows=rows,
        subset_positions=np.asarray([by_row[row] for row in serialized_rows]),
        subset_id=str(record["subset_id"]),
        branch_side=str(record["branch_side"]),
        integer_threshold=int(record["integer_threshold"]),
    )
    expected = cut.as_dict(rows)
    for key in (
        "certificate_kind",
        "cut_id",
        "subset_id",
        "subset_source_rows",
        "branch_side",
        "integer_threshold",
        "rhs",
        "coefficient_sha256",
        "validity",
        "exact_source_pmin_pmax_changed",
    ):
        if record.get(key) != expected[key]:
            raise ScopfError("Serialized cardinality cut identity mismatch")
    return cut


def commitment_feasibility_cut_from_record(
    record: dict[str, Any], generator_source_rows: npt.ArrayLike
) -> CommitmentFeasibilityCut:
    """Rebuild one serialized Phase-I feasibility cut by source-row identity.

    ``generator_source_rows`` uses the public one-based MATPOWER row identity,
    while the in-memory feasibility-cut serializer historically accepts the
    zero-based source indices and adds one.  Keeping that conversion here
    makes old v8-v10 cut evidence replayable and gives new frontier
    certificates one deterministic reconstruction path.
    """

    rows = np.asarray(generator_source_rows, dtype=np.int64)
    if rows.ndim != 1 or len(set(int(row) for row in rows)) != int(rows.size):
        raise ScopfError("Serialized feasibility-cut generator identity is invalid")
    by_row = {int(row): position for position, row in enumerate(rows)}
    coefficients = np.zeros(rows.size, dtype=np.float64)
    serialized = record.get("generator_coefficients", [])
    seen: set[int] = set()
    for item in serialized:
        source_row = int(item["source_row"])
        if source_row not in by_row or source_row in seen:
            raise ScopfError(
                "Serialized feasibility cut references an unknown or duplicate generator"
            )
        seen.add(source_row)
        coefficients[by_row[source_row]] = float(item["coefficient_pu"])
    serialized_hash = str(record.get("coefficient_sha256", ""))
    if len(serialized_hash) != 64 or any(
        character not in "0123456789abcdef" for character in serialized_hash
    ):
        raise ScopfError("Serialized feasibility cut has an invalid coefficient hash")
    cut = CommitmentFeasibilityCut(
        cut_id=str(record["cut_id"]),
        coefficients=coefficients,
        rhs=float(record["rhs"]),
        source_commitment_sha256=str(record["source_commitment_sha256"]),
        conservative_source_violation_pu=float(record["conservative_source_violation_pu"]),
    )
    cut.validate(rows.size)
    expected = cut.as_dict(rows - 1)
    for key in (
        "certificate_kind",
        "cut_id",
        "source_commitment_sha256",
        "rhs",
        "conservative_source_violation_pu",
        "generator_coefficient_count",
        "nonzero_generator_coefficient_count",
        "generator_coefficients",
        "validity",
        "exact_source_pmin_pmax_changed",
    ):
        if record.get(key) != expected[key]:
            raise ScopfError(
                "Serialized feasibility cut identity mismatch for "
                f"{key}: expected={expected[key]!r}, observed={record.get(key)!r}"
            )
    return cut


def commitment_capacity_cut_from_record(
    record: dict[str, Any], generator_source_rows: npt.ArrayLike
) -> CommitmentCapacityCut:
    """Rebuild one row-capacity cut using public one-based generator rows."""

    rows = np.asarray(generator_source_rows, dtype=np.int64)
    if rows.ndim != 1 or len(set(int(row) for row in rows)) != int(rows.size):
        raise ScopfError("Serialized capacity-cut generator identity is invalid")
    by_row = {int(row): position for position, row in enumerate(rows)}
    coefficients = np.zeros(rows.size, dtype=np.float64)
    seen: set[int] = set()
    for item in record.get("generator_coefficients", []):
        source_row = int(item["source_row"])
        if source_row not in by_row or source_row in seen:
            raise ScopfError(
                "Serialized capacity cut references an unknown or duplicate generator"
            )
        seen.add(source_row)
        coefficients[by_row[source_row]] = float(item["coefficient_pu"])
    cut = CommitmentCapacityCut(
        cut_id=str(record["cut_id"]),
        coefficients=coefficients,
        rhs=float(record["rhs"]),
        source_commitment_sha256=str(record["source_commitment_sha256"]),
        conservative_source_violation_pu=float(
            record["conservative_source_violation_pu"]
        ),
        source_row_name=str(record["source_row_name"]),
        source_row_side=str(record["source_row_side"]),
        source_row_bound_pu=float(record["source_row_bound_pu"]),
        outward_rhs_relaxation_pu=float(record["outward_rhs_relaxation_pu"]),
    )
    cut.validate(rows.size)
    expected = cut.as_dict(rows)
    for key in (
        "certificate_kind",
        "cut_id",
        "source_commitment_sha256",
        "source_row_name",
        "source_row_side",
        "source_row_bound_pu",
        "rhs",
        "outward_rhs_relaxation_pu",
        "conservative_source_violation_pu",
        "generator_coefficient_count",
        "nonzero_generator_coefficient_count",
        "generator_coefficients",
        "coefficient_sha256",
        "validity",
        "exact_source_pmin_pmax_changed",
    ):
        if record.get(key) != expected[key]:
            raise ScopfError(
                f"Serialized capacity cut identity mismatch for {key}"
            )
    return cut


def commitment_cover_cut_from_record(
    record: dict[str, Any], generator_source_rows: npt.ArrayLike
) -> CommitmentCoverCut:
    """Rebuild a serialized binary cover from public generator-row identity."""

    rows = np.asarray(generator_source_rows, dtype=np.int64)
    if rows.ndim != 1 or len(set(int(row) for row in rows)) != int(rows.size):
        raise ScopfError("Serialized cover-cut generator identity is invalid")
    by_row = {int(row): position for position, row in enumerate(rows)}
    cover_rows = tuple(int(row) for row in record["cover_source_rows"])
    core_rows = tuple(int(row) for row in record.get("core_cover_source_rows", ()))
    extended_rows = tuple(int(row) for row in record.get("extended_source_rows", ()))
    complemented_rows = tuple(
        int(row) for row in record["complemented_source_rows"]
    )
    if any(row not in by_row for row in cover_rows):
        raise ScopfError("Serialized cover cut references an unknown generator")
    if not set(complemented_rows).issubset(cover_rows):
        raise ScopfError("Serialized cover cut has an invalid complemented subset")
    complemented = set(complemented_rows)
    coefficients = np.zeros(rows.size, dtype=np.float64)
    for source_row in cover_rows:
        coefficients[by_row[source_row]] = (
            -1.0 if source_row in complemented else 1.0
        )
    cut = CommitmentCoverCut(
        cut_id=str(record["cut_id"]),
        coefficients=coefficients,
        rhs=float(record["rhs"]),
        source_cut_id=str(record["source_cut_id"]),
        source_cut_kind=str(record["source_cut_kind"]),
        cover_source_rows=cover_rows,
        complemented_source_rows=complemented_rows,
        transformed_rhs=float(record["transformed_rhs"]),
        cover_weight_sum=float(record["cover_weight_sum"]),
        conservative_cover_excess=float(record["conservative_cover_excess"]),
        cover_safety_margin=float(record["cover_safety_margin"]),
        source_commitment_sha256=str(record["source_commitment_sha256"]),
        separation_reference_sha256=str(record["separation_reference_sha256"]),
        separation_reference_violation=float(
            record["separation_reference_violation"]
        ),
        derivation_strategy=str(record["derivation_strategy"]),
        core_cover_source_rows=core_rows,
        extended_source_rows=extended_rows,
        extension_weight_threshold=(
            None
            if record.get("extension_weight_threshold") is None
            else float(record["extension_weight_threshold"])
        ),
    )
    cut.validate(rows.size)
    expected = cut.as_dict(rows)
    keys = [
        "certificate_kind",
        "cut_id",
        "source_cut_id",
        "source_cut_kind",
        "cover_source_rows",
        "complemented_source_rows",
        "rhs",
        "transformed_rhs",
        "cover_weight_sum",
        "conservative_cover_excess",
        "cover_safety_margin",
        "source_commitment_sha256",
        "separation_reference_sha256",
        "separation_reference_violation",
        "derivation_strategy",
        "coefficient_sha256",
        "validity",
        "exact_source_pmin_pmax_changed",
    ]
    if core_rows:
        keys.extend(
            [
                "core_cover_source_rows",
                "extended_source_rows",
                "extension_weight_threshold",
            ]
        )
    for key in keys:
        if record.get(key) != expected[key]:
            raise ScopfError(f"Serialized cover cut identity mismatch for {key}")
    return cut


def commitment_upper_cut_from_record(
    record: dict[str, Any], generator_source_rows: npt.ArrayLike
) -> CommitmentUpperCut:
    """Rebuild either supported replayable commitment upper inequality."""

    kind = str(record.get("certificate_kind", ""))
    if kind == "phase_one_binary_benders_feasibility_cut_v1":
        return commitment_feasibility_cut_from_record(record, generator_source_rows)
    if kind == "conditional_dispatch_row_capacity_cut_v1":
        return commitment_capacity_cut_from_record(record, generator_source_rows)
    if kind in {
        "binary_knapsack_cover_from_commitment_cut_v1",
        "binary_knapsack_extended_cover_from_commitment_cut_v2",
    }:
        return commitment_cover_cut_from_record(record, generator_source_rows)
    if kind == "binary_commitment_cardinality_branch_v1":
        return commitment_cardinality_cut_from_record(record, generator_source_rows)
    raise ScopfError(f"Unknown serialized commitment-cut kind: {kind!r}")


def add_commitment_upper_cuts(
    master: ReducedMaster, cuts: tuple[CommitmentUpperCut, ...]
) -> dict[str, int]:
    """Append commitment-only upper inequalities to a reduced master."""

    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64)
    row_by_cut: dict[str, int] = {}
    for cut in cuts:
        cut.validate(source_rows.size)
        if cut.cut_id in row_by_cut:
            raise ScopfError("Duplicate commitment upper-cut id")
        coefficients = {
            master.index.commitment_by_generator[int(source_rows[position])]: float(value)
            for position, value in enumerate(cut.coefficients)
            if value != 0.0
        }
        row_by_cut[cut.cut_id] = master.canonical.add_row(
            cut.cut_id, coefficients, upper=float(cut.rhs)
        )
    return row_by_cut


def _phase_dual_by_source_side(
    phase_model: Any, certificate: dict[str, Any]
) -> dict[tuple[str, str], float]:
    records = certificate["canonical_row_duals"]
    by_semantic = {
        str(
            record.get(
                "semantic_row_key",
                phase_one_semantic_row_key(str(record["row_name"])),
            )
        ): float(record["canonical_row_dual"])
        for record in records
    }
    expected = {phase_one_semantic_row_key(name) for name in phase_model.row_names}
    if len(by_semantic) != len(records) or set(by_semantic) != expected:
        raise ScopfError("Phase-I cut derivation found a row-identity mismatch")
    by_source_side: dict[tuple[str, str], float] = {}
    for semantic, value in by_semantic.items():
        parts = semantic.split("__", 2)
        if (
            len(parts) != 3
            or parts[0] != "phase1"
            or parts[1]
            not in {
                "lower",
                "upper",
            }
        ):
            raise ScopfError(f"Malformed semantic Phase-I row key: {semantic!r}")
        key = (parts[2], parts[1])
        if key in by_source_side:
            raise ScopfError("Phase-I cut derivation found a duplicate source row side")
        by_source_side[key] = min(float(value), 0.0)
    return by_source_side


def _relax_commitment_cut_coefficient_dust(
    *,
    coefficients: npt.ArrayLike,
    rhs: float,
    source_commitment: npt.ArrayLike,
    source_violation_pu: float,
    requested_zero_tolerance: float,
) -> tuple[FloatArray, float, float, dict[str, Any]]:
    """Drop tiny upper-cut coefficients without excluding any binary point.

    For ``a @ u <= b`` and dropped coefficients ``d``, use ``a' = a - d``
    and ``b' = b + sum(max(-d, 0))``.  Then ``a' @ u <= b'`` for every
    ``u in [0, 1]`` satisfying the original inequality.  The tolerance is
    also capped so the certificate's source commitment remains cut by a
    wide, auditable margin.
    """

    values = np.asarray(coefficients, dtype=np.float64)
    binary = np.asarray(source_commitment, dtype=np.int8)
    if values.ndim != 1 or binary.shape != values.shape:
        raise ScopfError("Commitment-cut coefficient cleanup received invalid shapes")
    if np.any(~np.isfinite(values)) or np.any((binary != 0) & (binary != 1)):
        raise ScopfError("Commitment-cut coefficient cleanup received invalid values")
    if not isfinite(rhs) or not isfinite(source_violation_pu):
        raise ScopfError("Commitment-cut coefficient cleanup received a nonfinite scalar")
    if source_violation_pu <= 0.0:
        raise ScopfError("Commitment-cut coefficient cleanup requires a positive source cut")
    if not isfinite(requested_zero_tolerance) or requested_zero_tolerance < 0.0:
        raise ScopfError("Commitment-cut coefficient tolerance must be nonnegative")

    adaptive_cap = source_violation_pu / (4.0 * max(1, values.size))
    effective_tolerance = min(requested_zero_tolerance, adaptive_cap)
    drop = (
        (values != 0.0) & (np.abs(values) <= effective_tolerance)
        if effective_tolerance > 0.0
        else np.zeros(values.shape, dtype=bool)
    )
    removed = values[drop]
    outward_rhs_relaxation = fsum(max(-float(value), 0.0) for value in removed)
    cleaned = values.copy()
    cleaned[drop] = 0.0
    cleaned = np.where(cleaned == 0.0, 0.0, cleaned).astype(np.float64, copy=False)
    cleaned_rhs = float(rhs + outward_rhs_relaxation)
    cleaned_source_violation = fsum(
        [
            -cleaned_rhs,
            *(float(cleaned[position]) for position in np.flatnonzero(binary)),
        ]
    )
    removed_absolute_mass = fsum(abs(float(value)) for value in removed)
    worst_binary_strengthening = max(
        0.0,
        fsum(max(-float(value), 0.0) for value in removed) - outward_rhs_relaxation,
    )
    if worst_binary_strengthening > 8.0 * np.finfo(np.float64).eps * max(
        1.0, abs(rhs), abs(cleaned_rhs)
    ):
        raise ScopfError("Commitment-cut coefficient cleanup strengthened the cut")
    if cleaned_source_violation <= 0.0:
        raise ScopfError("Commitment-cut coefficient cleanup removed the source violation")
    if source_violation_pu - cleaned_source_violation > (
        removed_absolute_mass + 8.0 * np.finfo(np.float64).eps * max(1.0, abs(source_violation_pu))
    ):
        raise ScopfError("Commitment-cut cleanup loss exceeded removed coefficient mass")

    return (
        cleaned,
        cleaned_rhs,
        cleaned_source_violation,
        {
            "policy": "drop_coefficient_dust_with_outward_rhs_relaxation_v1",
            "requested_zero_tolerance": requested_zero_tolerance,
            "adaptive_source_violation_cap": adaptive_cap,
            "effective_zero_tolerance": effective_tolerance,
            "dropped_coefficient_count": int(np.count_nonzero(drop)),
            "maximum_dropped_absolute_coefficient": (
                0.0 if removed.size == 0 else float(np.max(np.abs(removed)))
            ),
            "dropped_absolute_coefficient_mass": removed_absolute_mass,
            "outward_rhs_relaxation": outward_rhs_relaxation,
            "source_violation_before_cleanup_pu": source_violation_pu,
            "source_violation_after_cleanup_pu": cleaned_source_violation,
            "worst_binary_strengthening_pu": worst_binary_strengthening,
            "original_feasible_commitment_can_be_removed": False,
            "original_milp_feasible_set_changed": False,
        },
    )


def derive_commitment_capacity_cut(
    *,
    master: ReducedMaster,
    source_row_name: str,
    source_commitment: npt.ArrayLike,
    base_mva: float,
    safety_margin_pu: float,
) -> tuple[CommitmentCapacityCut, dict[str, Any]]:
    """Derive one direct PMIN/PMAX commitment envelope for a dispatch row.

    This is a solver-independent certificate.  It uses only the cleaned
    reduced dispatch row and the exact source PMIN/PMAX intervals.  The source
    commitment selects which violated row side generated the cut; it does not
    otherwise affect the globally valid inequality.
    """

    if not isfinite(base_mva) or base_mva <= 0.0:
        raise ScopfError("Commitment capacity cut requires positive base MVA")
    if not isfinite(safety_margin_pu) or safety_margin_pu < 0.0:
        raise ScopfError("Commitment capacity cut safety margin must be nonnegative")
    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64)
    binary = np.asarray(source_commitment, dtype=np.int8)
    if binary.shape != (source_rows.size,) or np.any((binary != 0) & (binary != 1)):
        raise ScopfError("Commitment capacity cut requires an exact binary source")
    matches = [row for row in master.coupling_rows if row.row_name == source_row_name]
    if len(matches) != 1:
        raise ScopfError("Commitment capacity cut source row is missing or duplicated")
    coupling = matches[0]
    coefficients = np.asarray(coupling.generator_coefficients, dtype=np.float64)
    curves = [master.costs[int(row)] for row in source_rows]
    pmin = np.asarray([curve.pmin_mw for curve in curves], dtype=np.float64)
    pmax = np.asarray([curve.pmax_mw for curve in curves], dtype=np.float64)
    if (
        coefficients.shape != binary.shape
        or np.any(~np.isfinite(coefficients))
        or np.any(~np.isfinite(pmin))
        or np.any(~np.isfinite(pmax))
        or np.any(pmin > pmax)
    ):
        raise ScopfError("Commitment capacity cut found invalid row or generator bounds")
    minimum_activity = np.minimum(coefficients * pmin, coefficients * pmax)
    maximum_activity = np.maximum(coefficients * pmin, coefficients * pmax)
    row_lower, row_upper = master.canonical.row_bound_arrays()
    lower = float(row_lower[coupling.row_index])
    upper = float(row_upper[coupling.row_index])
    candidates: list[tuple[str, FloatArray, float, float, float]] = []
    if isfinite(upper):
        candidate = minimum_activity / base_mva
        transformed_rhs = upper / base_mva
        violation = fsum(
            [
                -transformed_rhs,
                *(float(candidate[position]) for position in np.flatnonzero(binary)),
            ]
        )
        candidates.append(("upper", candidate, transformed_rhs, violation, upper / base_mva))
    if isfinite(lower):
        candidate = -maximum_activity / base_mva
        transformed_rhs = -lower / base_mva
        violation = fsum(
            [
                -transformed_rhs,
                *(float(candidate[position]) for position in np.flatnonzero(binary)),
            ]
        )
        candidates.append(("lower", candidate, transformed_rhs, violation, lower / base_mva))
    if not candidates:
        raise ScopfError("Commitment capacity cut source row has no finite side")
    side, cut_coefficients, raw_rhs, raw_violation, source_bound_pu = max(
        candidates, key=lambda item: (item[3], item[0])
    )
    if raw_violation <= safety_margin_pu:
        raise ScopfError(
            "Commitment capacity cut source does not violate a row envelope beyond its margin"
        )
    cut_coefficients = np.where(cut_coefficients == 0.0, 0.0, cut_coefficients).astype(
        np.float64, copy=False
    )
    rounding_margin = 32.0 * np.finfo(np.float64).eps * max(
        1.0,
        abs(raw_rhs),
        float(np.sum(np.abs(cut_coefficients))),
    )
    outward_relaxation = float(safety_margin_pu + rounding_margin)
    rhs = float(raw_rhs + outward_relaxation)
    source_violation = fsum(
        [
            -rhs,
            *(float(cut_coefficients[position]) for position in np.flatnonzero(binary)),
        ]
    )
    if source_violation <= 0.0:
        raise ScopfError("Commitment capacity cut margin consumed its source violation")
    source_sha = hashlib.sha256(binary.tobytes()).hexdigest()
    identity_bytes = (
        b"conditional_dispatch_row_capacity_cut_v1\0"
        + source_row_name.encode("utf-8")
        + b"\0"
        + side.encode("ascii")
        + np.asarray([rhs], dtype=np.float64).tobytes()
        + cut_coefficients.tobytes()
    )
    cut = CommitmentCapacityCut(
        cut_id="rc_" + hashlib.sha256(identity_bytes).hexdigest()[:24],
        coefficients=cut_coefficients,
        rhs=rhs,
        source_commitment_sha256=source_sha,
        conservative_source_violation_pu=source_violation,
        source_row_name=source_row_name,
        source_row_side=side,
        source_row_bound_pu=source_bound_pu,
        outward_rhs_relaxation_pu=outward_relaxation,
    )
    cut.validate(source_rows.size)
    return cut, {
        "derivation": "direct_conditional_pmin_pmax_row_activity_envelope_v1",
        "source_row_name": source_row_name,
        "source_row_side": side,
        "source_row_kind": coupling.kind,
        "source_row_bound_pu": source_bound_pu,
        "raw_source_violation_pu": raw_violation,
        "conservative_source_violation_pu": source_violation,
        "outward_rhs_relaxation_pu": outward_relaxation,
        "nonzero_coefficient_count": int(np.count_nonzero(cut_coefficients)),
        "cut": cut.as_dict(source_rows + 1),
    }


def derive_commitment_capacity_cut_for_side(
    *,
    master: ReducedMaster,
    source_row_name: str,
    source_row_side: str,
    base_mva: float,
    safety_margin_pu: float,
) -> tuple[CommitmentCapacityCut, dict[str, Any]]:
    """Derive a deterministic row envelope without a trial commitment.

    For one finite side of a reduced dispatch row, the maximally violating
    binary commitment is known analytically: turn on every generator whose
    projected coefficient is positive and turn off every generator whose
    coefficient is nonpositive.  If even that binary point does not violate
    the envelope, the row cannot yield a nontrivial binary cover.  Otherwise
    the resulting capacity cut is the same globally valid exact-PMIN/PMAX
    necessary condition used by :func:`derive_commitment_capacity_cut`, but
    its source identity is independent of a solver-generated candidate.
    """

    if source_row_side not in {"upper", "lower"}:
        raise ScopfError("Commitment capacity-cut side must be upper or lower")
    if not isfinite(base_mva) or base_mva <= 0.0:
        raise ScopfError("Commitment capacity cut requires positive base MVA")
    if not isfinite(safety_margin_pu) or safety_margin_pu < 0.0:
        raise ScopfError("Commitment capacity cut safety margin must be nonnegative")

    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64)
    matches = [row for row in master.coupling_rows if row.row_name == source_row_name]
    if len(matches) != 1:
        raise ScopfError("Commitment capacity cut source row is missing or duplicated")
    coupling = matches[0]
    coefficients = np.asarray(coupling.generator_coefficients, dtype=np.float64)
    curves = [master.costs[int(row)] for row in source_rows]
    pmin = np.asarray([curve.pmin_mw for curve in curves], dtype=np.float64)
    pmax = np.asarray([curve.pmax_mw for curve in curves], dtype=np.float64)
    if (
        coefficients.shape != (source_rows.size,)
        or np.any(~np.isfinite(coefficients))
        or np.any(~np.isfinite(pmin))
        or np.any(~np.isfinite(pmax))
        or np.any(pmin > pmax)
    ):
        raise ScopfError("Commitment capacity cut found invalid row or generator bounds")

    minimum_activity = np.minimum(coefficients * pmin, coefficients * pmax)
    maximum_activity = np.maximum(coefficients * pmin, coefficients * pmax)
    row_lower, row_upper = master.canonical.row_bound_arrays()
    if source_row_side == "upper":
        source_bound = float(row_upper[coupling.row_index])
        if not isfinite(source_bound):
            raise ScopfError("Commitment capacity-cut upper side is not finite")
        cut_coefficients = minimum_activity / base_mva
        raw_rhs = source_bound / base_mva
    else:
        source_bound = float(row_lower[coupling.row_index])
        if not isfinite(source_bound):
            raise ScopfError("Commitment capacity-cut lower side is not finite")
        cut_coefficients = -maximum_activity / base_mva
        raw_rhs = -source_bound / base_mva

    cut_coefficients = np.where(
        cut_coefficients == 0.0, 0.0, cut_coefficients
    ).astype(np.float64, copy=False)
    source_commitment = np.asarray(cut_coefficients > 0.0, dtype=np.int8)
    raw_violation = fsum(
        [-raw_rhs]
        + [
            float(cut_coefficients[position])
            for position in np.flatnonzero(source_commitment)
        ]
    )
    if raw_violation <= safety_margin_pu:
        raise ScopfError(
            "Commitment capacity-cut side has no binary violation beyond its margin"
        )

    rounding_margin = 32.0 * np.finfo(np.float64).eps * max(
        1.0,
        abs(raw_rhs),
        float(np.sum(np.abs(cut_coefficients))),
    )
    outward_relaxation = float(safety_margin_pu + rounding_margin)
    rhs = float(raw_rhs + outward_relaxation)
    source_violation = fsum(
        [-rhs]
        + [
            float(cut_coefficients[position])
            for position in np.flatnonzero(source_commitment)
        ]
    )
    if source_violation <= 0.0:
        raise ScopfError("Commitment capacity-cut margin consumed its source violation")

    source_sha = hashlib.sha256(source_commitment.tobytes()).hexdigest()
    identity_bytes = (
        b"conditional_dispatch_row_capacity_cut_v1\0"
        + source_row_name.encode("utf-8")
        + b"\0"
        + source_row_side.encode("ascii")
        + np.asarray([rhs], dtype=np.float64).tobytes()
        + cut_coefficients.tobytes()
    )
    cut = CommitmentCapacityCut(
        cut_id="rc_" + hashlib.sha256(identity_bytes).hexdigest()[:24],
        coefficients=cut_coefficients,
        rhs=rhs,
        source_commitment_sha256=source_sha,
        conservative_source_violation_pu=source_violation,
        source_row_name=source_row_name,
        source_row_side=source_row_side,
        source_row_bound_pu=source_bound / base_mva,
        outward_rhs_relaxation_pu=outward_relaxation,
    )
    cut.validate(source_rows.size)
    return cut, {
        "derivation": (
            "direct_conditional_pmin_pmax_row_activity_envelope_"
            "analytic_maximizer_v1"
        ),
        "source_row_name": source_row_name,
        "source_row_side": source_row_side,
        "source_row_kind": coupling.kind,
        "source_row_bound_pu": source_bound / base_mva,
        "raw_source_violation_pu": raw_violation,
        "conservative_source_violation_pu": source_violation,
        "outward_rhs_relaxation_pu": outward_relaxation,
        "analytic_source_commitment_policy": (
            "positive_projected_coefficient_online_else_off_v1"
        ),
        "analytic_source_commitment_generator_rows": (
            source_rows[source_commitment == 1] + 1
        ).tolist(),
        "nonzero_coefficient_count": int(np.count_nonzero(cut_coefficients)),
        "cut": cut.as_dict(source_rows + 1),
    }


def derive_commitment_feasibility_cut(
    *,
    master: ReducedMaster,
    phase_model: Any,
    phase_certificate: dict[str, Any],
    source_commitment: npt.ArrayLike,
    replay_tolerance_pu: float,
    coefficient_zero_tolerance: float = 0.0,
) -> tuple[CommitmentFeasibilityCut, dict[str, Any]]:
    """Lift one replayed fixed-u Phase-I dual into a global binary cut.

    For a Phase-I upper row ``A p - baseMVA * rho <= b`` with multiplier
    ``y <= 0``, the box-dual contribution of generator ``g`` when online is
    ``min(r_g * PMIN_g, r_g * PMAX_g)``, where ``r = -A.T @ y``.  It is zero
    when the generator is offline.  Consequently the dual lower bound is
    affine in the binary commitment, and every commitment admitting a
    zero-violation dispatch must make that affine expression nonpositive.
    """

    if not isfinite(replay_tolerance_pu) or replay_tolerance_pu < 0.0:
        raise ScopfError("Commitment-cut replay tolerance must be nonnegative")
    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64)
    binary = np.asarray(source_commitment, dtype=np.int8)
    if binary.shape != (source_rows.size,) or np.any((binary != 0) & (binary != 1)):
        raise ScopfError("Commitment-cut derivation requires an exact binary commitment")
    replayed = replay_phase_one_certificate(phase_model, phase_certificate)
    if not bool(replayed["prune_certified"]):
        raise ScopfError("A non-pruning Phase-I dual cannot create a feasibility cut")

    by_source_side = _phase_dual_by_source_side(phase_model, phase_certificate)
    coupling_by_name = {row.row_name: row for row in master.coupling_rows}
    if len(coupling_by_name) != len(master.coupling_rows):
        raise ScopfError("Reduced master contains duplicate coupling-row names")
    row_lower, row_upper = master.canonical.row_bound_arrays()
    reduced_dispatch = np.zeros(source_rows.size, dtype=np.float64)
    constant_terms: list[float] = []
    multiplier_sum = 0.0
    used_sides: list[dict[str, Any]] = []
    for (row_name, side), multiplier in sorted(by_source_side.items()):
        coupling = coupling_by_name.get(row_name)
        if coupling is None:
            raise ScopfError(f"Phase-I cut row {row_name!r} is not a source coupling row")
        row = int(coupling.row_index)
        if side == "upper":
            bound = float(row_upper[row])
            coefficients = np.asarray(coupling.generator_coefficients, dtype=np.float64)
        else:
            bound = -float(row_lower[row])
            coefficients = -np.asarray(coupling.generator_coefficients, dtype=np.float64)
        if not isfinite(bound):
            raise ScopfError("Phase-I cut references a nonfinite source-row side")
        constant_terms.append(multiplier * bound)
        reduced_dispatch -= multiplier * coefficients
        multiplier_sum += multiplier
        if multiplier != 0.0:
            used_sides.append(
                {
                    "row_name": row_name,
                    "side": side,
                    "canonical_row_dual": multiplier,
                }
            )

    curves = [master.costs[int(row)] for row in source_rows]
    pmin = np.asarray([curve.pmin_mw for curve in curves], dtype=np.float64)
    pmax = np.asarray([curve.pmax_mw for curve in curves], dtype=np.float64)
    if np.any(~np.isfinite(pmin)) or np.any(~np.isfinite(pmax)) or np.any(pmin > pmax):
        raise ScopfError("Commitment-cut derivation found invalid source PMIN/PMAX")
    online_values = np.minimum(reduced_dispatch * pmin, reduced_dispatch * pmax)
    # Signed zero is mathematically immaterial but changes byte hashes across
    # sparse/BLAS implementations.  Canonicalize it before the stable cut id
    # and serialized coefficient hash are formed.
    online_values = np.where(online_values == 0.0, 0.0, online_values).astype(
        np.float64, copy=False
    )
    violation_upper = float(phase_model.column_upper[-1])
    base_mva = float(-phase_model.matrix_csr()[0, -1])
    if not isfinite(base_mva) or base_mva <= 0.0:
        raise ScopfError("Phase-I cut could not recover positive base MVA")
    rho_reduced_cost = 1.0 + base_mva * multiplier_sum
    rho_box_term = min(0.0, rho_reduced_cost * violation_upper)
    raw_constant = fsum(constant_terms + [rho_box_term])
    safety_margin = float(phase_certificate["safety_margin_pu"])
    conservative_constant = raw_constant - safety_margin
    raw_at_source = fsum([raw_constant] + [float(value) for value in online_values * binary])
    conservative_at_source = raw_at_source - safety_margin
    replayed_raw = float(replayed["raw_lower_bound_pu"])
    replayed_conservative = float(replayed["conservative_lower_bound_pu"])
    raw_difference = abs(raw_at_source - replayed_raw)
    conservative_difference = abs(conservative_at_source - replayed_conservative)
    if max(raw_difference, conservative_difference) > replay_tolerance_pu:
        raise ScopfError("Lifted Phase-I commitment cut does not reproduce the source certificate")

    rhs = -conservative_constant
    online_values, rhs, cleaned_source_violation, cleanup_audit = (
        _relax_commitment_cut_coefficient_dust(
            coefficients=online_values,
            rhs=rhs,
            source_commitment=binary,
            source_violation_pu=conservative_at_source,
            requested_zero_tolerance=coefficient_zero_tolerance,
        )
    )
    source_sha = hashlib.sha256(binary.tobytes()).hexdigest()
    identity_bytes = (
        source_sha.encode("ascii")
        + np.asarray([rhs], dtype=np.float64).tobytes()
        + online_values.tobytes()
    )
    cut = CommitmentFeasibilityCut(
        cut_id="fc_" + hashlib.sha256(identity_bytes).hexdigest()[:24],
        coefficients=online_values,
        rhs=rhs,
        source_commitment_sha256=source_sha,
        conservative_source_violation_pu=cleaned_source_violation,
    )
    cut.validate(source_rows.size)
    if abs(cut.violation(binary) - cleaned_source_violation) > replay_tolerance_pu:
        raise ScopfError("Commitment-cut source violation failed its affine replay")
    return cut, {
        "derivation": "phase_one_box_dual_conditional_pmin_pmax_lift_v1",
        "phase_one_replay": replayed,
        "raw_source_replay_difference_pu": raw_difference,
        "conservative_source_replay_difference_pu": conservative_difference,
        "rho_reduced_cost": rho_reduced_cost,
        "rho_box_term": rho_box_term,
        "nonzero_phase_row_dual_count": len(used_sides),
        "nonzero_phase_row_duals": used_sides,
        "coefficient_cleanup": cleanup_audit,
        "cut": cut.as_dict(source_rows),
    }


def generate_commitment_cut_repairs(
    *,
    cut: CommitmentFeasibilityCut | CommitmentCapacityCut,
    commitment: npt.ArrayLike,
    fixed_off: npt.ArrayLike,
    fixed_on: npt.ArrayLike,
    pmin_mw: npt.ArrayLike,
    pmax_mw: npt.ArrayLike,
    demand_mw: float,
    economic_on_values: npt.ArrayLike,
    maximum_repairs: int,
) -> list[tuple[IntArray, dict[str, Any]]]:
    """Generate deterministic binary neighbors that satisfy one certified cut."""

    binary = np.asarray(commitment, dtype=np.int8)
    off_mask = np.asarray(fixed_off, dtype=bool)
    on_mask = np.asarray(fixed_on, dtype=bool)
    pmin = np.asarray(pmin_mw, dtype=np.float64)
    pmax = np.asarray(pmax_mw, dtype=np.float64)
    economics = np.asarray(economic_on_values, dtype=np.float64)
    count = binary.size
    cut.validate(count)
    if maximum_repairs < 1:
        raise ScopfError("Commitment-cut repair count must be positive")
    if not (
        off_mask.shape
        == on_mask.shape
        == pmin.shape
        == pmax.shape
        == economics.shape
        == binary.shape
    ):
        raise ScopfError("Commitment-cut repair arrays have inconsistent shapes")
    if np.any((binary != 0) & (binary != 1)) or np.any(off_mask & on_mask):
        raise ScopfError("Commitment-cut repair received invalid binary masks")
    base_violation = cut.violation(binary)
    if base_violation <= 0.0:
        return []

    positions = np.flatnonzero(~(off_mask | on_mask))
    flip_delta = cut.coefficients * (1 - 2 * binary)
    beneficial = positions[flip_delta[positions] < 0.0]
    if beneficial.size == 0:
        return []

    def capacity(candidate: IntArray) -> tuple[float, float, float]:
        minimum = float(pmin @ candidate)
        maximum = float(pmax @ candidate)
        shortfall = max(0.0, minimum - demand_mw, demand_mw - maximum)
        return minimum, maximum, float(shortfall)

    orders = [
        sorted(
            (int(position) for position in beneficial),
            key=lambda position: (
                float(flip_delta[position]),
                float(economics[position] if binary[position] == 0 else -economics[position]),
                position,
            ),
        ),
        sorted(
            (int(position) for position in beneficial),
            key=lambda position: (
                float(economics[position] if binary[position] == 0 else -economics[position]),
                float(flip_delta[position]),
                position,
            ),
        ),
        sorted(
            (int(position) for position in beneficial),
            key=lambda position: (
                float(flip_delta[position])
                / max(
                    1e-12,
                    abs(
                        float(
                            economics[position] if binary[position] == 0 else -economics[position]
                        )
                    ),
                ),
                position,
            ),
        ),
    ]
    repairs: list[tuple[IntArray, dict[str, Any]]] = []
    seen: set[str] = set()
    for order_index, order in enumerate(orders, start=1):
        candidate = binary.copy()
        flipped: list[int] = []
        for position in order:
            candidate[position] = np.int8(1 - candidate[position])
            flipped.append(position)
            violation = cut.violation(candidate)
            minimum, maximum, shortfall = capacity(candidate)
            if violation > 0.0 or shortfall > 1e-9:
                continue
            digest = hashlib.sha256(candidate.tobytes()).hexdigest()
            if digest in seen:
                break
            seen.add(digest)
            repairs.append(
                (
                    candidate.copy(),
                    {
                        "policy": "certified_phase_one_cut_greedy_repair_v1",
                        "cut_id": cut.cut_id,
                        "order": order_index,
                        "source_cut_violation_pu": base_violation,
                        "repaired_cut_violation_pu": violation,
                        "hamming_distance": len(flipped),
                        "flipped_generator_positions": list(flipped),
                        "minimum_dispatch_mw": minimum,
                        "maximum_dispatch_mw": maximum,
                        "capacity_shortfall_mw": shortfall,
                        "candidate_commitment_sha256": digest,
                        "candidate_only_not_feasibility_proof": True,
                    },
                )
            )
            break
        if len(repairs) >= maximum_repairs:
            break
    return repairs
