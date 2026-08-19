"""Immutable source manifest and source-row identity reporting."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .matpower import GEN_STATUS, PMAX, PMIN, ContingencyTable, MatpowerCase


def generator_source_id(source_row_zero_based: int) -> str:
    return f"gen-row-{source_row_zero_based + 1:04d}"


def branch_source_id(source_row_zero_based: int) -> str:
    return f"branch-row-{source_row_zero_based + 1:04d}"


def build_source_manifest(
    case: MatpowerCase,
    contingencies: ContingencyTable,
    *,
    source_archive_path: Path | None = None,
    source_archive_sha256: str | None = None,
) -> dict[str, Any]:
    status = case.gen[:, GEN_STATUS] > 0
    table_counts = Counter(change.table for change in contingencies.changes)
    source_identity: dict[str, Any] = {
        "case_file": case.source_path.name,
        "case_sha256": case.sha256,
        "contingency_mode": contingencies.mode,
        "row_identity_convention": {
            "generator": "gen-row-NNNN is the immutable one-based mpc.gen source row",
            "branch": "branch-row-NNNN is the immutable one-based mpc.branch source row",
            "contingency_change": (
                "source_row is the immutable one-based chgtab row"
                if contingencies.mode
                in {"source_table", "mapped_reference_branch_table"}
                else "source_row equals the immutable one-based mpc.branch source row"
            ),
        },
    }
    if contingencies.source_path is not None:
        source_identity.update(
            {
                "contingency_file": contingencies.source_path.name,
                "contingency_sha256": contingencies.sha256,
            }
        )
    else:
        source_identity.update(
            {
                "contingency_file": None,
                "contingency_sha256": None,
                "contingency_derivation": (
                    "one deterministic outage candidate per immutable branch row; "
                    "catalog retains only in-service non-islanding branches"
                ),
            }
        )
    if contingencies.derivation is not None:
        source_identity["contingency_derivation"] = contingencies.derivation
    if source_archive_path is not None:
        source_identity.update(
            {
                "source_archive_file": source_archive_path.name,
                "source_archive_sha256": source_archive_sha256,
            }
        )
    return {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "case_name": case.case_name,
        "synthetic_system": True,
        "source_identity": source_identity,
        "dimensions": {
            "buses": int(case.bus.shape[0]),
            "generators": int(case.gen.shape[0]),
            "source_online_generators": int(np.count_nonzero(status)),
            "source_offline_generators": int(np.count_nonzero(~status)),
            "branches": int(case.branch.shape[0]),
            "gencost_rows": int(case.gencost.shape[0]),
            "contingency_change_rows": len(contingencies.changes),
            "contingency_table_counts": dict(sorted(table_counts.items())),
        },
        "source_online_capacity_mw": {
            "pmin_sum": float(case.gen[status, PMIN].sum()),
            "pmax_sum": float(case.gen[status, PMAX].sum()),
        },
        "generators": [
            {
                "source_id": generator_source_id(i),
                "source_row": i + 1,
                "bus": int(row[0]),
                "source_status": int(row[GEN_STATUS]),
                "pmin_mw": float(row[PMIN]),
                "pmax_mw": float(row[PMAX]),
            }
            for i, row in enumerate(case.gen)
        ],
    }


def write_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    """Write JSON with a same-directory replace; caller must guard the path."""

    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
