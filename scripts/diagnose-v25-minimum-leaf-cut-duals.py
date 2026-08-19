"""Replay and polish the weakest v25 frontier leaf without CPU solution data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from activsg_scopf.commitment_cuts import (
    CommitmentCardinalityCut,
    commitment_upper_cut_from_record,
)
from activsg_scopf.config import load_config
from activsg_scopf.lagrangian import (
    evaluate_lagrangian_bound,
    optimize_commitment_cut_duals_coordinate_cupy,
    replay_lagrangian_certificate,
)
from activsg_scopf.lagrangian_experiment import (
    _masks_from_record,
    _security_row_raw_violation_envelope_mw,
    validate_lagrangian_experiment_config,
)
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    security_pair_from_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_lagrangian_experiment_config(config)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    if payload.get("benchmark_id") != config.benchmark_id:
        raise RuntimeError("Diagnostic result/config benchmark identity differs")
    if payload.get("bound_status") != "independently_replayed_current_frontier":
        raise RuntimeError("Diagnostic requires an independently replayed frontier")

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
    source_rows = np.flatnonzero(case.gen[:, GEN_STATUS] > 0).astype(np.int64) + 1
    record = min(
        payload["frontier_regions"],
        key=lambda item: (
            float(item["lagrangian_certificate"]["conservative_lower_bound"]),
            str(item["region_id"]),
        ),
    )
    masks = _masks_from_record(record, source_rows)
    pairs = tuple(
        security_pair_from_record(
            pair_record,
            catalog,
            lodf_absolute_tolerance=float(
                config.model.get("serialized_lodf_replay_tolerance", 0.0)
            ),
        )
        for pair_record in record["security_pairs"]
    )
    master = build_reduced_master(
        case,
        network,
        segments=int(config.model["pwl_segments"]),
        coefficient_zero_tolerance=float(
            config.model.get("reduced_coefficient_zero_tolerance", 1e-14)
        ),
        security_row_maximum_raw_violation_envelope_mw=(
            _security_row_raw_violation_envelope_mw(
                config,
                base_mva=float(case.base_mva),
            )
        ),
    )
    add_reduced_security_pairs(master, network, pairs)
    cuts = tuple(
        commitment_upper_cut_from_record(cut_record, source_rows)
        for cut_record in record.get("commitment_upper_cuts", [])
    )
    cut_by_id = {cut.cut_id: cut for cut in cuts}
    replayed = replay_lagrangian_certificate(
        master,
        record["lagrangian_certificate"],
        masks,
        commitment_cuts_by_id=cut_by_id,
    )
    row_by_name = {
        row.row_name: row.row_index for row in master.coupling_rows
    }
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    for row_name, value in replayed.coupling_duals:
        row_dual[row_by_name[row_name]] = value
    supplied_by_id = dict(replayed.commitment_cut_duals)
    hard_ids = set(replayed.hard_cardinality_cut_ids)
    nonhard_cuts = tuple(cut for cut in cuts if cut.cut_id not in hard_ids)
    hard_cuts = tuple(
        cut
        for cut in cuts
        if cut.cut_id in hard_ids and isinstance(cut, CommitmentCardinalityCut)
    )
    nonhard_initial = np.asarray(
        [supplied_by_id.get(cut.cut_id, 0.0) for cut in nonhard_cuts],
        dtype=np.float64,
    )
    polished_nonhard, audit = optimize_commitment_cut_duals_coordinate_cupy(
        master,
        row_dual,
        masks,
        commitment_cuts=nonhard_cuts,
        initial_commitment_cut_dual=nonhard_initial,
        cycles=args.cycles,
    )
    polished_by_id = {
        cut.cut_id: float(value)
        for cut, value in zip(nonhard_cuts, polished_nonhard, strict=True)
    }
    full_polished = np.asarray(
        [polished_by_id.get(cut.cut_id, 0.0) for cut in cuts],
        dtype=np.float64,
    )
    polished = evaluate_lagrangian_bound(
        master,
        row_dual,
        masks,
        safety_margin_dollars=float(
            config.raw["benchmark"]["certificate_safety_margin_dollars"]
        ),
        commitment_cuts=cuts,
        commitment_cut_dual=full_polished,
        hard_cardinality_cuts=hard_cuts,
    )
    if polished.conservative_lower_bound + 1e-9 < replayed.conservative_lower_bound:
        raise RuntimeError("Coordinate proposal weakened the exact hard-cardinality bound")
    print(
        json.dumps(
            {
                "benchmark_id": config.benchmark_id,
                "region_id": record["region_id"],
                "cycles": args.cycles,
                "nonhard_cut_count": len(nonhard_cuts),
                "hard_cardinality_cut_count": len(hard_cuts),
                "initial_nonzero_nonhard_cut_duals": int(
                    np.count_nonzero(nonhard_initial)
                ),
                "polished_nonzero_nonhard_cut_duals": int(
                    np.count_nonzero(polished_nonhard)
                ),
                "initial_exact_bound": replayed.conservative_lower_bound,
                "polished_exact_bound": polished.conservative_lower_bound,
                "exact_bound_lift_dollars": (
                    polished.conservative_lower_bound
                    - replayed.conservative_lower_bound
                ),
                "coordinate_search_separable_bound_lift_dollars": audit[
                    "improvement_dollars"
                ],
                "cycle_raw_lower_bounds": audit["cycle_raw_lower_bounds"].tolist(),
                "cpu_solution_data_used": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
