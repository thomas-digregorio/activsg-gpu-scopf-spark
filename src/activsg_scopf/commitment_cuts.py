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
        canonical_coefficients = np.where(
            self.coefficients == 0.0, 0.0, self.coefficients
        ).astype(np.float64, copy=False)
        nonzero = np.flatnonzero(canonical_coefficients != 0.0)
        return {
            "certificate_kind": "phase_one_binary_benders_feasibility_cut_v1",
            "cut_id": self.cut_id,
            "source_commitment_sha256": self.source_commitment_sha256,
            "rhs": self.rhs,
            "conservative_source_violation_pu": (
                self.conservative_source_violation_pu
            ),
            "generator_coefficient_count": int(rows.size),
            "nonzero_generator_coefficient_count": int(nonzero.size),
            "generator_coefficients": [
                {
                    "source_row": int(rows[position]) + 1,
                    "coefficient_pu": float(canonical_coefficients[position]),
                }
                for position in nonzero
            ],
            "coefficient_sha256": hashlib.sha256(
                canonical_coefficients.tobytes()
            ).hexdigest(),
            "validity": (
                "necessary_for_zero_violation_dispatch_with_exact_conditional_"
                "source_pmin_pmax"
            ),
            "exact_source_pmin_pmax_changed": False,
        }


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
            "coefficient_sha256": hashlib.sha256(
                self.coefficients.tobytes()
            ).hexdigest(),
            "validity": "integer_sum_disjunction_over_source_binary_commitments",
            "exact_source_pmin_pmax_changed": False,
        }


type CommitmentUpperCut = CommitmentFeasibilityCut | CommitmentCardinalityCut


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
    rhs = float(integer_threshold) if branch_side == "at_most" else -float(
        integer_threshold
    )
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
        conservative_source_violation_pu=float(
            record["conservative_source_violation_pu"]
        ),
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


def commitment_upper_cut_from_record(
    record: dict[str, Any], generator_source_rows: npt.ArrayLike
) -> CommitmentUpperCut:
    """Rebuild either supported replayable commitment upper inequality."""

    kind = str(record.get("certificate_kind", ""))
    if kind == "phase_one_binary_benders_feasibility_cut_v1":
        return commitment_feasibility_cut_from_record(
            record, generator_source_rows
        )
    if kind == "binary_commitment_cardinality_branch_v1":
        return commitment_cardinality_cut_from_record(
            record, generator_source_rows
        )
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
        if len(parts) != 3 or parts[0] != "phase1" or parts[1] not in {
            "lower",
            "upper",
        }:
            raise ScopfError(f"Malformed semantic Phase-I row key: {semantic!r}")
        key = (parts[2], parts[1])
        if key in by_source_side:
            raise ScopfError("Phase-I cut derivation found a duplicate source row side")
        by_source_side[key] = min(float(value), 0.0)
    return by_source_side


def derive_commitment_feasibility_cut(
    *,
    master: ReducedMaster,
    phase_model: Any,
    phase_certificate: dict[str, Any],
    source_commitment: npt.ArrayLike,
    replay_tolerance_pu: float,
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
            raise ScopfError(
                f"Phase-I cut row {row_name!r} is not a source coupling row"
            )
        row = int(coupling.row_index)
        if side == "upper":
            bound = float(row_upper[row])
            coefficients = np.asarray(
                coupling.generator_coefficients, dtype=np.float64
            )
        else:
            bound = -float(row_lower[row])
            coefficients = -np.asarray(
                coupling.generator_coefficients, dtype=np.float64
            )
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
        raise ScopfError(
            "Lifted Phase-I commitment cut does not reproduce the source certificate"
        )

    rhs = -conservative_constant
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
        conservative_source_violation_pu=conservative_at_source,
    )
    cut.validate(source_rows.size)
    if abs(cut.violation(binary) - conservative_at_source) > replay_tolerance_pu:
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
        "cut": cut.as_dict(source_rows),
    }


def generate_commitment_cut_repairs(
    *,
    cut: CommitmentFeasibilityCut,
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
                            economics[position]
                            if binary[position] == 0
                            else -economics[position]
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
