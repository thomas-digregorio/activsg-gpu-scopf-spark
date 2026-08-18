#!/usr/bin/env python3
"""Replay the preserved v8 root dual with the v9 compact identity contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from activsg_scopf.config import load_config
from activsg_scopf.lagrangian_experiment import verify_lagrangian_certificate_payload
from activsg_scopf.matpower import read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    security_pair_from_record,
)


def main() -> None:
    root = Path("/workspace")
    config = load_config(root / "configs" / "activsg2000-gpu-lagrangian-v9.json")
    source = json.loads(
        (
            root
            / "results"
            / "experiments"
            / "activsg2000-gpu-lagrangian-v8-dgx-spark.json"
        ).read_text(encoding="utf-8")
    )
    payload = copy.deepcopy(source)
    case = read_matpower_case(
        config.case_path,
        expected_sha256=config.raw["raw_inputs"]["case_sha256"],
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
    for record in payload["frontier_regions"]:
        master = build_reduced_master(
            case,
            network,
            segments=10,
            coefficient_zero_tolerance=float(
                config.model["reduced_coefficient_zero_tolerance"]
            ),
        )
        pairs = tuple(
            security_pair_from_record(
                pair_record,
                catalog,
                lodf_absolute_tolerance=float(
                    config.model["serialized_lodf_replay_tolerance"]
                ),
            )
            for pair_record in record["security_pairs"]
        )
        add_reduced_security_pairs(
            master,
            network,
            pairs,
            expected_representative_by_pair_id=record[
                "security_row_equivalence"
            ]["representative_by_pair_id"],
            equivalence_replay_tolerance=float(
                config.model["security_equivalence_replay_tolerance"]
            ),
        )
        certificate = record["lagrangian_certificate"]
        names = sorted(row.row_name for row in master.coupling_rows)
        certificate["serialization"] = (
            "sparse_nonzero_dual_order_independent_identity_v3"
        )
        certificate["coupling_row_name_set_sha256"] = hashlib.sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest()
        certificate["derived_generator_vectors_are_recomputed_not_hash_gated"] = True
        for key in (
            "coupling_row_name_sha256",
            "effective_dispatch_coefficient_sha256",
            "on_subproblem_value_sha256",
            "minimizing_commitment_sha256",
        ):
            certificate.pop(key, None)
    replay = verify_lagrangian_certificate_payload(config, payload)
    print("V9_ROOT_REPLAY=" + json.dumps(replay, sort_keys=True))


if __name__ == "__main__":
    main()
