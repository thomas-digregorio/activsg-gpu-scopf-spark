"""Auditable mapping of a source contingency table into a revised topology."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from .errors import ProvenanceError
from .matpower import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_X,
    F_BUS,
    RATE_A,
    SHIFT,
    T_BUS,
    TAP,
    ContingencyTable,
    MatpowerCase,
)

BRANCH_MAPPING_METHOD = "endpoint_parameter_assignment_v1"

_IDENTITY_COLUMNS = np.asarray(
    [BR_R, BR_X, BR_B, RATE_A, TAP, SHIFT, ANGMIN, ANGMAX], dtype=np.int64
)
_NORMALIZATION_FLOORS = np.asarray(
    [1e-5, 1e-5, 1e-5, 1.0, 1e-3, 0.1, 1.0, 1.0], dtype=np.float64
)
_MAXIMUM_ACCEPTED_DISTANCE = 0.5


def _endpoint_key(row: np.ndarray) -> tuple[int, int]:
    return tuple(sorted((int(row[F_BUS]), int(row[T_BUS]))))


def _endpoint_groups(branch: np.ndarray) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for source_index, row in enumerate(branch):
        groups[_endpoint_key(row)].append(source_index)
    return dict(groups)


def _same_orientation(reference: np.ndarray, target: np.ndarray) -> bool:
    return bool(
        int(reference[F_BUS]) == int(target[F_BUS])
        and int(reference[T_BUS]) == int(target[T_BUS])
    )


def _normalized_parameter_distance(
    reference: np.ndarray, target: np.ndarray
) -> float:
    reference_values = reference[_IDENTITY_COLUMNS]
    target_values = target[_IDENTITY_COLUMNS]
    scale = np.maximum.reduce(
        (np.abs(reference_values), np.abs(target_values), _NORMALIZATION_FLOORS)
    )
    return float(np.sum(np.abs(reference_values - target_values) / scale))


def _parameter_record(row: np.ndarray) -> dict[str, float]:
    return {
        "br_r_pu": float(row[BR_R]),
        "br_x_pu": float(row[BR_X]),
        "br_b_pu": float(row[BR_B]),
        "rate_a_mw": float(row[RATE_A]),
        "tap": float(row[TAP]),
        "shift_degrees": float(row[SHIFT]),
        "angle_min_degrees": float(row[ANGMIN]),
        "angle_max_degrees": float(row[ANGMAX]),
    }


def map_reference_branch_contingencies(
    reference_case: MatpowerCase,
    target_case: MatpowerCase,
    reference_table: ContingencyTable,
    *,
    method: str,
) -> ContingencyTable:
    """Map reference branch rows by endpoints and closest physical parameters.

    The public MATPOWER files have no stable circuit identifier across these
    revisions. Endpoint pairs provide the physical grouping. A deterministic
    rectangular assignment then retains the closest same-orientation circuits;
    target-only additions remain monitored but are not outage candidates.
    """

    if method != BRANCH_MAPPING_METHOD:
        raise ProvenanceError(f"Unsupported branch-contingency mapping method: {method}")
    if reference_table.mode != "source_table":
        raise ProvenanceError("Reference contingencies must come from a source table")
    if set(reference_case.bus[:, 0].astype(np.int64)) != set(
        target_case.bus[:, 0].astype(np.int64)
    ):
        raise ProvenanceError("Reference and target cases do not have identical bus IDs")

    referenced_rows = sorted(
        {
            int(change.element_row)
            for change in reference_table.changes
            if change.table == "CT_TBRCH"
        }
    )
    if not referenced_rows:
        raise ProvenanceError("Reference table has no branch contingencies to map")
    if referenced_rows[0] < 1 or referenced_rows[-1] > reference_case.branch.shape[0]:
        raise ProvenanceError("Reference contingency table names an invalid branch row")

    reference_groups = _endpoint_groups(reference_case.branch)
    target_groups = _endpoint_groups(target_case.branch)
    mapping: dict[int, int] = {}
    records: list[dict[str, object]] = []
    equivalent_ties = 0
    non_equivalent_ties = 0

    for endpoint, all_reference_indices in sorted(reference_groups.items()):
        reference_indices = all_reference_indices
        target_indices = target_groups.get(endpoint, [])
        if len(target_indices) < len(reference_indices):
            raise ProvenanceError(
                "Target topology has too few circuits for reference endpoint "
                f"{endpoint}: requires {len(reference_indices)}, found {len(target_indices)}"
            )

        physical_cost = np.empty(
            (len(reference_indices), len(target_indices)), dtype=np.float64
        )
        deterministic_cost = np.empty_like(physical_cost)
        for reference_ordinal, reference_index in enumerate(reference_indices):
            reference_row = reference_case.branch[reference_index]
            for target_ordinal, target_index in enumerate(target_indices):
                target_row = target_case.branch[target_index]
                distance = _normalized_parameter_distance(reference_row, target_row)
                if not _same_orientation(reference_row, target_row):
                    distance += 1_000.0
                physical_cost[reference_ordinal, target_ordinal] = distance
                deterministic_cost[reference_ordinal, target_ordinal] = (
                    distance
                    + 1e-9 * abs(reference_ordinal - target_ordinal)
                    + 1e-12 * target_ordinal
                )

        assigned_reference, assigned_target = linear_sum_assignment(deterministic_cost)
        for reference_ordinal, target_ordinal in zip(
            assigned_reference, assigned_target, strict=True
        ):
            reference_index = reference_indices[int(reference_ordinal)]
            target_index = target_indices[int(target_ordinal)]
            reference_row = reference_case.branch[reference_index]
            target_row = target_case.branch[target_index]
            distance = float(physical_cost[reference_ordinal, target_ordinal])
            if not _same_orientation(reference_row, target_row):
                raise ProvenanceError(
                    "Branch mapping required a reversed circuit orientation at reference "
                    f"row {reference_index + 1}"
                )
            if distance > _MAXIMUM_ACCEPTED_DISTANCE:
                raise ProvenanceError(
                    "Branch mapping exceeded the accepted physical-parameter distance at "
                    f"reference row {reference_index + 1}: {distance}"
                )

            tied_ordinals = np.flatnonzero(
                np.isclose(
                    physical_cost[reference_ordinal],
                    distance,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            tie_kind = "none"
            if tied_ordinals.size > 1:
                signatures = {
                    (
                        bool(_same_orientation(reference_row, target_case.branch[index])),
                        *target_case.branch[index, _IDENTITY_COLUMNS].tolist(),
                    )
                    for index in (target_indices[int(value)] for value in tied_ordinals)
                }
                if len(signatures) == 1:
                    tie_kind = "equivalent_parallel_circuits"
                    equivalent_ties += 1
                else:
                    tie_kind = "non_equivalent_candidates"
                    non_equivalent_ties += 1

            mapping[reference_index + 1] = target_index + 1
            exact = bool(
                _same_orientation(reference_row, target_row)
                and np.array_equal(
                    reference_row[_IDENTITY_COLUMNS], target_row[_IDENTITY_COLUMNS]
                )
            )
            records.append(
                {
                    "reference_branch_source_row": reference_index + 1,
                    "target_branch_source_row": target_index + 1,
                    "from_bus": int(target_row[F_BUS]),
                    "to_bus": int(target_row[T_BUS]),
                    "exact_parameter_match": exact,
                    "normalized_parameter_distance": distance,
                    "assignment_tie": tie_kind,
                    "reference_parameters": _parameter_record(reference_row),
                    "target_parameters": _parameter_record(target_row),
                }
            )

    expected_reference_rows = set(range(1, reference_case.branch.shape[0] + 1))
    if set(mapping) != expected_reference_rows:
        missing = sorted(expected_reference_rows - set(mapping))
        raise ProvenanceError(f"Reference branch rows were not mapped: {missing[:10]}")
    if len(set(mapping.values())) != len(mapping):
        raise ProvenanceError("Branch-contingency mapping is not one-to-one")
    if non_equivalent_ties:
        raise ProvenanceError(
            "Branch-contingency mapping has non-equivalent minimum-cost ties"
        )

    mapped_changes = tuple(
        replace(change, element_row=mapping[change.element_row])
        if change.table == "CT_TBRCH"
        else change
        for change in reference_table.changes
    )
    mapped_target_rows = set(mapping.values())
    unreferenced_reference_rows = sorted(expected_reference_rows - set(referenced_rows))
    unreferenced_records = [
        {
            "reference_branch_source_row": reference_row,
            "target_branch_source_row": mapping[reference_row],
        }
        for reference_row in unreferenced_reference_rows
    ]
    target_only_records = [
        {
            "target_branch_source_row": source_index + 1,
            "from_bus": int(row[F_BUS]),
            "to_bus": int(row[T_BUS]),
        }
        for source_index, row in enumerate(target_case.branch)
        if source_index + 1 not in mapped_target_rows
    ]
    exact_matches = sum(bool(record["exact_parameter_match"]) for record in records)
    maximum_distance = max(
        float(record["normalized_parameter_distance"]) for record in records
    )
    mapping_records_sha256 = hashlib.sha256(
        json.dumps(
            records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    derivation: dict[str, object] = {
        "mapping_method": method,
        "reference_case_file": reference_case.source_path.name,
        "reference_case_sha256": reference_case.sha256,
        "reference_contingency_file": (
            None
            if reference_table.source_path is None
            else reference_table.source_path.name
        ),
        "reference_contingency_sha256": reference_table.sha256,
        "target_case_file": target_case.source_path.name,
        "target_case_sha256": target_case.sha256,
        "reference_change_rows": len(reference_table.changes),
        "reference_branch_change_rows": sum(
            change.table == "CT_TBRCH" for change in reference_table.changes
        ),
        "reference_unique_branch_contingencies": len(referenced_rows),
        "reference_branches_without_branch_contingency": len(
            unreferenced_reference_rows
        ),
        "reference_branches_without_branch_contingency_records": (
            unreferenced_records
        ),
        "reference_generator_change_rows_deferred_unmapped": sum(
            change.table == "CT_TGEN" for change in reference_table.changes
        ),
        "mapped_unique_reference_branches": len(mapping),
        "mapped_unique_target_branches": len(mapped_target_rows),
        "exact_parameter_matches": exact_matches,
        "updated_parameter_matches": len(records) - exact_matches,
        "equivalent_parallel_assignment_ties": equivalent_ties,
        "non_equivalent_assignment_ties": non_equivalent_ties,
        "maximum_normalized_parameter_distance": maximum_distance,
        "branch_mapping_records_sha256": mapping_records_sha256,
        "target_only_branches_not_outage_candidates": len(target_only_records),
        "target_only_branches_remain_monitored": True,
        "updated_parameter_mapping_records": [
            record for record in records if not record["exact_parameter_match"]
        ],
        "target_only_branch_records": target_only_records,
    }
    return ContingencyTable(
        source_path=reference_table.source_path,
        sha256=reference_table.sha256,
        changes=mapped_changes,
        mode="mapped_reference_branch_table",
        derivation=derivation,
    )
