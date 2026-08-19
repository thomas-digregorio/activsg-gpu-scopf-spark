"""Exercise the v26 shared-master frontier replay on one real root certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activsg_scopf.config import load_config
from activsg_scopf.lagrangian_experiment import verify_lagrangian_certificate_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    source = json.loads(args.result.read_text(encoding="utf-8"))
    root = dict(source["solved_region_history"][0])
    if root["region_id"] != "r":
        raise RuntimeError("Shared-master preflight did not find the root certificate")
    evidence_id = "v26_shared_master_preflight_v1"
    shared = {
        "evidence_id": evidence_id,
        "policy": "immutable_shared_network_master_for_proof_only_children_v1",
        "coefficient_cleanup_audit": root.pop("coefficient_cleanup_audit"),
        "security_row_equivalence": root.pop("security_row_equivalence"),
        "security_pairs": root.pop("security_pairs"),
    }
    shared["logical_security_pair_count"] = len(shared["security_pairs"])
    root["shared_master_evidence_id"] = evidence_id
    payload = {
        "commitment_feasibility_cuts": source["commitment_feasibility_cuts"],
        "commitment_capacity_cuts": source["commitment_capacity_cuts"],
        "commitment_cover_cuts": source["commitment_cover_cuts"],
        "frontier_regions": [root],
        "pruned_regions": [],
        "disjunctive_splits": [],
        "bound": root["lagrangian_certificate"]["conservative_lower_bound"],
        "shared_master_evidence": shared,
    }
    replay = verify_lagrangian_certificate_payload(config, payload)
    if not replay["passed"]:
        raise RuntimeError("Shared-master preflight replay did not pass")
    print(
        "V26_SHARED_FRONTIER_PREFLIGHT="
        + json.dumps(
            {
                key: replay[key]
                for key in (
                    "passed",
                    "frontier_region_count",
                    "active_master_cache_builds",
                    "global_bound_difference_dollars",
                    "maximum_region_replay_difference_dollars",
                    "elapsed_seconds",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
