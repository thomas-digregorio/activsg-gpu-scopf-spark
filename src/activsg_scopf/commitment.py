"""Deterministic unit-commitment snapshots and round-to-round comparisons."""

from __future__ import annotations

import hashlib
from typing import Any


def _binary_commitment(record: dict[str, Any]) -> int:
    return int(float(record["commitment"]) > 0.5)


def commitment_snapshot(
    solution: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """Serialize a full source-row commitment and its changes from one prior round."""

    generators = solution["generators"]
    records = [
        {
            "source_id": str(record["source_id"]),
            "source_row": int(record["source_row"]),
            "bus": int(record["bus"]),
            "commitment": _binary_commitment(record),
        }
        for record in generators
    ]
    vector = [record["commitment"] for record in records]
    fingerprint = hashlib.sha256(bytes(vector)).hexdigest()
    commitment_count = sum(vector)
    if previous is None:
        return {
            "policy": "full_source_row_commitment_delta_v1",
            "generator_count": len(records),
            "commitment_count": commitment_count,
            "commitment_fingerprint_sha256": fingerprint,
            "stable_from_previous_round": None,
            "hamming_distance_from_previous_round": None,
            "hamming_similarity_from_previous_round": None,
            "active_set_jaccard_from_previous_round": None,
            "off_to_on_from_previous_round": None,
            "on_to_off_from_previous_round": None,
            "changes_from_previous_round": [],
            "generator_commitments": records,
        }
    previous_records = previous["generator_commitments"]
    if len(previous_records) != len(records):
        raise ValueError("Commitment snapshots use different generator counts")
    changes: list[dict[str, Any]] = []
    intersection = 0
    union = 0
    for before, after in zip(previous_records, records, strict=True):
        if before["source_id"] != after["source_id"]:
            raise ValueError("Commitment snapshot source identity changed")
        old = int(before["commitment"])
        new = int(after["commitment"])
        intersection += int(old == 1 and new == 1)
        union += int(old == 1 or new == 1)
        if old != new:
            changes.append(
                {
                    "source_id": after["source_id"],
                    "source_row": after["source_row"],
                    "bus": after["bus"],
                    "from_commitment": old,
                    "to_commitment": new,
                }
            )
    hamming_distance = len(changes)
    generator_count = len(records)
    return {
        "policy": "full_source_row_commitment_delta_v1",
        "generator_count": generator_count,
        "commitment_count": commitment_count,
        "commitment_fingerprint_sha256": fingerprint,
        "stable_from_previous_round": hamming_distance == 0,
        "hamming_distance_from_previous_round": hamming_distance,
        "hamming_similarity_from_previous_round": (
            1.0 if not generator_count else 1.0 - hamming_distance / generator_count
        ),
        "active_set_jaccard_from_previous_round": (
            1.0 if not union else intersection / union
        ),
        "off_to_on_from_previous_round": sum(
            change["to_commitment"] == 1 for change in changes
        ),
        "on_to_off_from_previous_round": sum(
            change["to_commitment"] == 0 for change in changes
        ),
        "changes_from_previous_round": changes,
        "generator_commitments": records,
    }
