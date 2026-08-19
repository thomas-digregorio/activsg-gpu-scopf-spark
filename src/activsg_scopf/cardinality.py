"""GPU-disjunctive branching helpers for binary commitment cardinalities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import floor
from typing import Any

import numpy as np
import numpy.typing as npt

from .commitment_cuts import (
    CommitmentCardinalityCut,
    build_commitment_cardinality_cut,
)
from .errors import ScopfError
from .reduced import ReducedMaster

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CommitmentSubset:
    subset_id: str
    positions: IntArray
    source_rows: tuple[int, ...]
    family: str
    depth: int

    def validate(self, generator_count: int) -> None:
        if self.positions.ndim != 1 or self.positions.size == 0:
            raise ScopfError("Commitment subset is empty or malformed")
        if np.any(np.diff(self.positions) <= 0):
            raise ScopfError("Commitment subset positions are not strictly ordered")
        if self.positions[0] < 0 or self.positions[-1] >= generator_count:
            raise ScopfError("Commitment subset position is out of range")
        if len(self.source_rows) != int(self.positions.size):
            raise ScopfError("Commitment subset source-row identity is inconsistent")


@dataclass(frozen=True)
class CardinalitySplit:
    subset: CommitmentSubset
    lp_sum: float
    floor_value: int
    ceil_value: int
    fractionality: float
    at_most_cut: CommitmentCardinalityCut
    at_least_cut: CommitmentCardinalityCut

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_kind": "binary_commitment_cardinality_sum_v1",
            "subset_id": self.subset.subset_id,
            "subset_family": self.subset.family,
            "subset_depth": self.subset.depth,
            "subset_source_rows": list(self.subset.source_rows),
            "subset_size": len(self.subset.source_rows),
            "parent_lp_sum": self.lp_sum,
            "floor_value": self.floor_value,
            "ceil_value": self.ceil_value,
            "fractionality": self.fractionality,
            "at_most_cut_id": self.at_most_cut.cut_id,
            "at_least_cut_id": self.at_least_cut.cut_id,
            "validity": "integer_sum_disjunction_over_source_binary_commitments",
        }


def _curve_signature(master: ReducedMaster, generator_source_row: int) -> tuple[Any, ...]:
    curve = master.costs[int(generator_source_row)]
    return (
        float(curve.pmin_mw),
        float(curve.pmax_mw),
        tuple(float(value) for value in curve.coefficients),
        tuple(float(value) for value in curve.segment_widths_mw),
        tuple(float(value) for value in curve.segment_slopes_per_mwh),
        float(curve.committed_base_cost),
    )


def exact_cost_type_groups(master: ReducedMaster) -> tuple[IntArray, ...]:
    """Return exact PMIN/PMAX/PWL type groups in stable source-row order."""

    grouped: dict[tuple[Any, ...], list[int]] = {}
    for position, source_row in enumerate(master.index.generator_source_rows):
        grouped.setdefault(_curve_signature(master, int(source_row)), []).append(position)
    return tuple(
        np.asarray(positions, dtype=np.int64)
        for _signature, positions in sorted(
            grouped.items(),
            key=lambda item: tuple(
                int(master.index.generator_source_rows[position])
                for position in item[1]
            ),
        )
    )


def _subset_identity(
    source_rows: tuple[int, ...], *, family: str, depth: int
) -> str:
    payload = json.dumps(
        {
            "family": family,
            "depth": int(depth),
            "source_rows": source_rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"cs_{family}_{hashlib.sha256(payload).hexdigest()[:20]}"


def commitment_branch_subsets(master: ReducedMaster) -> tuple[CommitmentSubset, ...]:
    """Build deterministic exact-type and system-wide laminar subset families."""

    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64) + 1
    candidates: dict[tuple[int, ...], CommitmentSubset] = {}

    def add(positions: npt.ArrayLike, *, family: str, depth: int) -> None:
        selected = np.unique(np.asarray(positions, dtype=np.int64))
        # A one-element cardinality disjunction is just ordinary binary
        # branching.  Keeping those leaves in this family made the most-
        # fractional rule silently collapse back to unit-at-a-time branching.
        # Binary branching remains an explicit completeness fallback in the
        # experiment controller after all genuine multi-unit sums are integral.
        if selected.size < 2:
            return
        rows = tuple(int(row) for row in source_rows[selected])
        key = tuple(int(position) for position in selected)
        candidate = CommitmentSubset(
            subset_id=_subset_identity(rows, family=family, depth=depth),
            positions=selected,
            source_rows=rows,
            family=family,
            depth=int(depth),
        )
        candidate.validate(source_rows.size)
        prior = candidates.get(key)
        priority = {"exact_type": 0, "exact_type_hierarchy": 1, "system_hierarchy": 2}
        if prior is None or priority[candidate.family] < priority[prior.family]:
            candidates[key] = candidate

    def add_hierarchy(positions: IntArray, *, family: str, depth: int = 0) -> None:
        add(positions, family=family, depth=depth)
        if positions.size <= 1:
            return
        midpoint = positions.size // 2
        add_hierarchy(positions[:midpoint], family=family, depth=depth + 1)
        add_hierarchy(positions[midpoint:], family=family, depth=depth + 1)

    for positions in exact_cost_type_groups(master):
        add(positions, family="exact_type", depth=0)
        if positions.size > 1:
            midpoint = positions.size // 2
            add_hierarchy(
                positions[:midpoint], family="exact_type_hierarchy", depth=1
            )
            add_hierarchy(
                positions[midpoint:], family="exact_type_hierarchy", depth=1
            )
    add_hierarchy(
        np.arange(source_rows.size, dtype=np.int64),
        family="system_hierarchy",
    )
    return tuple(
        sorted(
            candidates.values(),
            key=lambda subset: (
                {"exact_type": 0, "exact_type_hierarchy": 1, "system_hierarchy": 2}[
                    subset.family
                ],
                subset.depth,
                subset.source_rows,
            ),
        )
    )


def choose_cardinality_split(
    *,
    master: ReducedMaster,
    commitments: npt.ArrayLike,
    subsets: tuple[CommitmentSubset, ...],
    existing_cut_ids: set[str],
    integrality_tolerance: float = 1e-7,
) -> CardinalitySplit:
    """Choose the most fractional valid cardinality disjunction."""

    values = np.asarray(commitments, dtype=np.float64)
    generator_count = master.index.generator_source_rows.size
    if values.shape != (generator_count,) or not np.all(np.isfinite(values)):
        raise ScopfError("Cardinality branching received invalid commitments")
    if not np.isfinite(integrality_tolerance) or integrality_tolerance < 0.0:
        raise ScopfError("Cardinality integrality tolerance is invalid")
    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64) + 1
    ranked: list[tuple[tuple[float, int, int, str], CardinalitySplit]] = []
    family_priority = {"exact_type": 2, "exact_type_hierarchy": 1, "system_hierarchy": 0}
    for subset in subsets:
        subset.validate(generator_count)
        value = float(np.sum(values[subset.positions]))
        nearest = round(value)
        if abs(value - nearest) <= integrality_tolerance:
            continue
        lower = int(floor(value))
        upper = lower + 1
        fractionality = min(value - lower, upper - value)
        at_most = build_commitment_cardinality_cut(
            generator_source_rows=source_rows,
            subset_positions=subset.positions,
            subset_id=subset.subset_id,
            branch_side="at_most",
            integer_threshold=lower,
        )
        at_least = build_commitment_cardinality_cut(
            generator_source_rows=source_rows,
            subset_positions=subset.positions,
            subset_id=subset.subset_id,
            branch_side="at_least",
            integer_threshold=upper,
        )
        if at_most.cut_id in existing_cut_ids or at_least.cut_id in existing_cut_ids:
            continue
        split = CardinalitySplit(
            subset=subset,
            lp_sum=value,
            floor_value=lower,
            ceil_value=upper,
            fractionality=fractionality,
            at_most_cut=at_most,
            at_least_cut=at_least,
        )
        rank = (
            family_priority[subset.family],
            fractionality,
            int(subset.positions.size),
            subset.subset_id,
        )
        ranked.append((rank, split))
    if not ranked:
        raise ScopfError("No fractional commitment cardinality subset remains")
    return max(ranked, key=lambda item: item[0])[1]


def exact_type_group_rounding(
    master: ReducedMaster, commitments: npt.ArrayLike
) -> tuple[np.ndarray, dict[str, Any]]:
    """Round exact-type counts and preserve the LP's preferred bus placements."""

    values = np.asarray(commitments, dtype=np.float64)
    generator_count = master.index.generator_source_rows.size
    if values.shape != (generator_count,) or not np.all(np.isfinite(values)):
        raise ScopfError("Type-group rounding received invalid commitments")
    rounded = np.zeros(generator_count, dtype=np.float64)
    records: list[dict[str, Any]] = []
    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64) + 1
    for positions in exact_cost_type_groups(master):
        lp_sum = float(np.sum(values[positions]))
        count = min(int(positions.size), max(0, int(floor(lp_sum + 0.5))))
        order = sorted(
            (int(position) for position in positions),
            key=lambda position: (-float(values[position]), int(source_rows[position])),
        )
        selected = order[:count]
        rounded[selected] = 1.0
        records.append(
            {
                "source_rows": [int(source_rows[position]) for position in positions],
                "lp_sum": lp_sum,
                "rounded_count": count,
                "selected_source_rows": [int(source_rows[position]) for position in selected],
            }
        )
    return rounded, {
        "policy": "exact_pmin_pmax_pwl_type_count_nearest_then_lp_placement_v1",
        "type_group_count": len(records),
        "multi_unit_type_group_count": sum(
            len(record["source_rows"]) > 1 for record in records
        ),
        "fractional_type_sum_count": sum(
            abs(float(record["lp_sum"]) - round(float(record["lp_sum"]))) > 1e-7
            for record in records
        ),
        "rounded_commitment_count": int(np.count_nonzero(rounded)),
        "groups": records,
        "cpu_solution_data_used": False,
    }


def balanced_exact_type_group_rounding_candidates(
    master: ReducedMaster,
    commitments: npt.ArrayLike,
    *,
    target_offsets: tuple[int, ...] = (-1, 0, 1),
) -> tuple[tuple[np.ndarray, dict[str, Any]], ...]:
    """Globally balance exact-type counts around the aggregate LP count.

    Independent nearest-integer rounding can round many type groups upward at
    once.  This dependent rounding instead fixes a requested *global* count,
    gives the available ceil decisions to the largest fractional type sums,
    and retains the LP-preferred bus placements within each exact type.  The
    result is only a deterministic GPU-LP-derived primal candidate; downstream
    exact-PMIN Phase I remains the feasibility authority.
    """

    values = np.asarray(commitments, dtype=np.float64)
    generator_count = master.index.generator_source_rows.size
    if values.shape != (generator_count,) or not np.all(np.isfinite(values)):
        raise ScopfError("Balanced type-group rounding received invalid commitments")
    if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
        raise ScopfError("Balanced type-group rounding received out-of-box commitments")
    if not target_offsets or any(
        isinstance(offset, bool) or not isinstance(offset, int) for offset in target_offsets
    ):
        raise ScopfError("Balanced type-group target offsets must be nonempty integers")

    clipped = np.clip(values, 0.0, 1.0)
    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64) + 1
    groups = exact_cost_type_groups(master)
    group_data: list[dict[str, Any]] = []
    floor_total = 0
    for group_index, positions in enumerate(groups):
        lp_sum = float(np.sum(clipped[positions]))
        lower = min(int(positions.size), max(0, int(floor(lp_sum + 1e-12))))
        fractional_part = max(0.0, min(1.0, lp_sum - lower))
        can_round_up = lower < int(positions.size) and fractional_part > 1e-10
        order = tuple(
            sorted(
                (int(position) for position in positions),
                key=lambda position: (
                    -float(clipped[position]),
                    int(source_rows[position]),
                ),
            )
        )
        group_data.append(
            {
                "group_index": group_index,
                "positions": positions,
                "source_rows": tuple(int(source_rows[position]) for position in positions),
                "lp_sum": lp_sum,
                "floor_count": lower,
                "fractional_part": fractional_part,
                "can_round_up": can_round_up,
                "placement_order": order,
            }
        )
        floor_total += lower

    aggregate_lp_count = float(np.sum(clipped))
    center_target = int(floor(aggregate_lp_count + 0.5))
    maximum_target = floor_total + sum(bool(group["can_round_up"]) for group in group_data)
    ranked_fractional_groups = tuple(
        sorted(
            (group for group in group_data if bool(group["can_round_up"])),
            key=lambda group: (
                -float(group["fractional_part"]),
                tuple(group["source_rows"]),
            ),
        )
    )

    candidates: list[tuple[np.ndarray, dict[str, Any]]] = []
    seen_targets: set[int] = set()
    for offset in target_offsets:
        requested_target = center_target + int(offset)
        target = min(maximum_target, max(floor_total, requested_target))
        if target in seen_targets:
            continue
        seen_targets.add(target)
        ceil_group_indices = {
            int(group["group_index"])
            for group in ranked_fractional_groups[: target - floor_total]
        }
        rounded = np.zeros(generator_count, dtype=np.float64)
        records: list[dict[str, Any]] = []
        for group in group_data:
            count = int(group["floor_count"]) + int(
                int(group["group_index"]) in ceil_group_indices
            )
            selected = tuple(group["placement_order"][:count])
            rounded[np.asarray(selected, dtype=np.int64)] = 1.0
            records.append(
                {
                    "source_rows": list(group["source_rows"]),
                    "lp_sum": float(group["lp_sum"]),
                    "floor_count": int(group["floor_count"]),
                    "rounded_up": bool(int(group["group_index"]) in ceil_group_indices),
                    "rounded_count": count,
                    "selected_source_rows": [int(source_rows[position]) for position in selected],
                }
            )
        observed_count = int(np.count_nonzero(rounded))
        if observed_count != target:
            raise ScopfError("Balanced type-group rounding missed its global target")
        candidates.append(
            (
                rounded,
                {
                    "policy": (
                        "global_dependent_exact_pmin_pmax_pwl_type_count_"
                        "then_lp_placement_v1"
                    ),
                    "target_offset": int(offset),
                    "requested_global_commitment_count": requested_target,
                    "global_commitment_count": target,
                    "aggregate_lp_commitment_count": aggregate_lp_count,
                    "global_floor_count": floor_total,
                    "global_maximum_dependent_rounding_count": maximum_target,
                    "rounded_up_type_group_count": target - floor_total,
                    "type_group_count": len(group_data),
                    "groups": records,
                    "candidate_only_not_feasibility_proof": True,
                    "exact_source_pmin_pmax_retained": True,
                    "cpu_solution_data_used": False,
                },
            )
        )
    return tuple(candidates)
