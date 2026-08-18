"""Rebuild one serialized v4 Phase-I leaf and report row-identity drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from activsg_scopf.config import load_config
from activsg_scopf.lagrangian_experiment import _masks_from_record
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.phase_one import (
    build_phase_one_model,
    phase_one_semantic_row_key,
    replay_phase_one_certificate,
)
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    fix_commitments,
    security_pair_from_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--region", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    records = payload["pruned_regions"]
    record = next(
        item
        for item in records
        if args.region is None or item["region_id"] == args.region
    )
    case = read_matpower_case(
        config.case_path, expected_sha256=config.raw["raw_inputs"]["case_sha256"]
    )
    table = read_contingency_table(
        config.contingency_path,
        expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
    )
    network = build_network(case)
    catalog = build_contingency_catalog(
        case,
        network,
        table,
        validation_columns=int(config.model["lodf_validation_columns"]),
        validation_tolerance_pu=float(config.model["lodf_validation_tolerance_pu"]),
        chunk_columns=int(config.model["lodf_build_chunk_columns"]),
    )
    source_rows = np.flatnonzero(case.gen[:, GEN_STATUS] > 0).astype(np.int64) + 1
    masks = _masks_from_record(record, source_rows)
    master = build_reduced_master(
        case,
        network,
        segments=10,
        coefficient_zero_tolerance=float(config.model["reduced_coefficient_zero_tolerance"]),
    )
    pairs = tuple(
        security_pair_from_record(
            item,
            catalog,
            lodf_absolute_tolerance=float(
                config.model["serialized_lodf_replay_tolerance"]
            ),
        )
        for item in record["security_pairs"]
    )
    add_reduced_security_pairs(
        master,
        network,
        pairs,
        expected_representative_by_pair_id=record["security_row_equivalence"][
            "representative_by_pair_id"
        ],
        equivalence_replay_tolerance=float(
            config.model["security_equivalence_replay_tolerance"]
        ),
    )
    fix_commitments(master, masks.fixed_off, masks.fixed_on)
    phase = build_phase_one_model(
        master.canonical,
        base_mva=float(case.base_mva),
        maximum_violation_pu=float(config.runtime["phase_one_maximum_violation_pu"]),
    )
    serialized = [
        item["row_name"]
        for item in record["phase_one_certificate"]["canonical_row_duals"]
    ]
    rebuilt = list(phase.row_names)
    serialized_set = set(serialized)
    rebuilt_set = set(rebuilt)
    serialized_semantic = {
        item.get(
            "semantic_row_key",
            phase_one_semantic_row_key(item["row_name"]),
        )
        for item in record["phase_one_certificate"]["canonical_row_duals"]
    }
    rebuilt_semantic = {phase_one_semantic_row_key(name) for name in rebuilt}
    first_order_mismatch = next(
        (
            {
                "position": position,
                "serialized": left,
                "rebuilt": right,
            }
            for position, (left, right) in enumerate(
                zip(serialized, rebuilt, strict=False)
            )
            if left != right
        ),
        None,
    )
    print(
        json.dumps(
            {
                "region_id": record["region_id"],
                "serialized_count": len(serialized),
                "rebuilt_count": len(rebuilt),
                "serialized_unique_count": len(serialized_set),
                "rebuilt_unique_count": len(rebuilt_set),
                "missing_from_rebuild": sorted(serialized_set - rebuilt_set)[:50],
                "extra_in_rebuild": sorted(rebuilt_set - serialized_set)[:50],
                "missing_semantic_rows": sorted(
                    serialized_semantic - rebuilt_semantic
                ),
                "extra_semantic_rows": sorted(
                    rebuilt_semantic - serialized_semantic
                ),
                "first_order_mismatch": first_order_mismatch,
                "security_representative_map_matches": (
                    record["security_row_equivalence"][
                        "representative_by_pair_id"
                    ]
                    == master.security_pair_representative_by_id
                ),
                "security_equivalence_classes_match": (
                    record["security_row_equivalence"]["equivalence_classes"]
                    == master.security_pair_equivalence_classes
                ),
                "phase_one_replay": replay_phase_one_certificate(
                    phase, record["phase_one_certificate"]
                ),
                "recorded_cleanup_audit": record["coefficient_cleanup_audit"],
                "rebuilt_cleanup_audit": master.coefficient_cleanup_audit,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
