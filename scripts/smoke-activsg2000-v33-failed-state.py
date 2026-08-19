#!/usr/bin/env python3
"""Replay the exact v32 failed proof state through the v33 GPU path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from activsg_scopf.cardinality import (
    choose_cardinality_split,
    commitment_branch_subsets,
)
from activsg_scopf.commitment_cuts import commitment_upper_cut_from_record
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.lagrangian import replay_lagrangian_certificate
from activsg_scopf.lagrangian_experiment import (
    ACTIVSG2000_V32_EXPERIMENT_ID,
    SolvedRegion,
    _certificate_dual_arrays_for_direct_cut_replay,
    _masks_from_record,
    _prepare_region_master,
    _solve_v32_proof_only_hard_cardinality_pair,
    validate_lagrangian_experiment_config,
)
from activsg_scopf.matpower import (
    GEN_STATUS,
    read_contingency_table,
    read_matpower_case,
)
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import security_pair_from_record
from activsg_scopf.solvers.cuopt_lp import ContinuousSolveResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/configs/activsg2000-gpu-lagrangian-v33.json"),
    )
    parser.add_argument(
        "--failed-config",
        type=Path,
        default=Path("/workspace/configs/activsg2000-gpu-lagrangian-v32.json"),
    )
    parser.add_argument(
        "--failed-result",
        type=Path,
        default=Path(
            "/workspace/results/experiments/"
            "activsg2000-gpu-lagrangian-v32-dgx-spark.json"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    failed_config = load_config(args.failed_config)
    validate_lagrangian_experiment_config(config)
    validate_lagrangian_experiment_config(failed_config)
    if failed_config.benchmark_id != ACTIVSG2000_V32_EXPERIMENT_ID:
        raise RuntimeError("Failed-state smoke requires the registered v32 config")
    if (
        config.raw["raw_inputs"] != failed_config.raw["raw_inputs"]
        or config.model != failed_config.model
    ):
        raise RuntimeError("V33 changed the raw inputs or mathematical model")

    payload = json.loads(args.failed_result.read_text(encoding="utf-8"))
    if (
        payload.get("benchmark_id") != ACTIVSG2000_V32_EXPERIMENT_ID
        or payload.get("status") != "failed_exception"
        or "below-threshold finite_column_bound" not in str(payload.get("error"))
    ):
        raise RuntimeError("Input result is not the preserved v32 bound failure")
    records = payload.get("frontier_regions", [])
    if len(records) != 1:
        raise RuntimeError("V32 failed-state smoke expected exactly one root region")
    record = records[0]

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
    masks = _masks_from_record(record, source_rows)
    if np.any(masks.fixed_off) or np.any(masks.fixed_on):
        raise RuntimeError("Preserved v32 frontier is not the root region")
    pair_records = record.get("security_pairs")
    if pair_records is None:
        pair_records = payload["shared_master_evidence"]["security_pairs"]
    pairs = tuple(
        security_pair_from_record(
            pair_record,
            catalog,
            lodf_absolute_tolerance=float(
                config.model.get("serialized_lodf_replay_tolerance", 0.0)
            ),
        )
        for pair_record in pair_records
    )
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=pairs,
        commitment_cuts=(),
    )
    cuts = tuple(
        commitment_upper_cut_from_record(cut_record, source_rows)
        for cut_record in record.get("commitment_upper_cuts", [])
    )
    cut_by_id = {cut.cut_id: cut for cut in cuts}
    if len(cut_by_id) != len(cuts):
        raise RuntimeError("Preserved v32 root contains duplicate commitment cuts")
    evaluation = replay_lagrangian_certificate(
        master,
        record["lagrangian_certificate"],
        masks,
        commitment_cuts_by_id=cut_by_id,
    )
    replay_difference = abs(
        evaluation.conservative_lower_bound
        - float(record["lagrangian_certificate"]["conservative_lower_bound"])
    )
    if replay_difference > float(
        config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]
    ):
        raise RuntimeError("Preserved v32 root certificate did not replay")
    row_dual, _cut_dual = _certificate_dual_arrays_for_direct_cut_replay(
        master,
        evaluation,
        cuts,
    )

    beam_candidates = payload["alternative_primal_candidate_beam"][
        "balanced_type_rounding_candidates"
    ]
    if not beam_candidates:
        raise RuntimeError("V32 result omitted its GPU-root rounding evidence")
    group_records = beam_candidates[0]["groups"]
    position_by_source_row = {
        int(row): position for position, row in enumerate(source_rows)
    }
    reconstructed_commitment = np.zeros(source_rows.size, dtype=np.float64)
    covered = np.zeros(source_rows.size, dtype=bool)
    for group in group_records:
        positions = np.asarray(
            [position_by_source_row[int(row)] for row in group["source_rows"]],
            dtype=np.int64,
        )
        value = float(group["lp_sum"]) / float(positions.size)
        if value < -1e-12 or value > 1.0 + 1e-12 or np.any(covered[positions]):
            raise RuntimeError("V32 exact-type group evidence is inconsistent")
        reconstructed_commitment[positions] = value
        covered[positions] = True
    if not np.all(covered):
        raise RuntimeError("V32 exact-type group evidence omitted an online generator")

    maximum_support = int(
        config.runtime["proof_only_hard_cardinality_maximum_support_size"]
    )
    subsets = tuple(
        subset
        for subset in commitment_branch_subsets(master)
        if int(subset.positions.size) <= maximum_support
    )
    split = choose_cardinality_split(
        master=master,
        commitments=reconstructed_commitment,
        subsets=subsets,
        existing_cut_ids={cut.cut_id for cut in cuts},
    )
    parent = SolvedRegion(
        region_id="r",
        masks=masks,
        master=master,
        solve=ContinuousSolveResult(
            status="PreservedV32CertificateReplay",
            optimal=False,
            primal_objective=None,
            dual_objective=None,
            values=None,
            native_primal=None,
            native_row_dual=None,
            solve_time_seconds=0.0,
            statistics={
                "error_status": "Success",
                "child_phase_one_solved": False,
            },
        ),
        canonical_row_dual=row_dual,
        lagrangian=evaluation,
        commitment=reconstructed_commitment,
        security_pairs=pairs,
        rounds=[],
        final_screen=record["final_screen"],
        gpu_lagrangian={"wall_time_seconds": 0.0},
        commitment_cuts=cuts,
        commitment_cut_row_by_id={},
    )
    children, audit = _solve_v32_proof_only_hard_cardinality_pair(
        child_specs=(
            ("r_s001_0", masks, cuts + (split.at_most_cut,)),
            ("r_s001_1", masks, cuts + (split.at_least_cut,)),
        ),
        parent=parent,
        case=case,
        config=config,
        deadline=Deadline(60.0, 0.0, 0.0),
    )

    floor = float(config.runtime["centered_dual_search_coefficient_zero_tolerance"])
    model_audit = audit["model"]
    child_search_audits = {
        child_id: child.gpu_lagrangian["passes"][0]["search_model"]
        for child_id, child in children.items()
    }
    snapped_count = sum(
        int(item["post_normalization_snapped_epigraph_bound_count"])
        for item in child_search_audits.values()
    )
    if snapped_count < 1:
        raise RuntimeError("Failed-state smoke did not reproduce the v32 asymmetry")
    if model_audit["block_count"] != 2:
        raise RuntimeError("Failed-state smoke did not build two sibling blocks")
    if model_audit["matrix_nonzero_minimum_absolute"] < floor:
        raise RuntimeError("V33 retained a below-floor matrix coefficient")
    if model_audit["objective_nonzero_minimum_absolute"] < floor:
        raise RuntimeError("V33 retained a below-floor objective coefficient")
    if model_audit["finite_column_bound_maximum_absolute"] > 1.0:
        raise RuntimeError("V33 retained a finite column bound above one")
    for child_id, child_audit in child_search_audits.items():
        minimum_bound = child_audit["finite_column_bound_minimum_nonzero_absolute"]
        if minimum_bound is not None and minimum_bound < floor:
            raise RuntimeError(f"V33 child {child_id} retained a below-floor bound")
    if any(
        child.lagrangian.conservative_lower_bound
        < parent.lagrangian.conservative_lower_bound
        for child in children.values()
    ):
        raise RuntimeError("V33 child certificate regressed below its parent")

    print(
        json.dumps(
            {
                "status": "pass",
                "case_name": config.case_name,
                "failed_result_benchmark_id": payload["benchmark_id"],
                "replacement_benchmark_id": config.benchmark_id,
                "root_certificate_replay_difference_dollars": replay_difference,
                "root_bound": evaluation.conservative_lower_bound,
                "split": split.as_dict(),
                "post_normalization_snapped_epigraph_bound_count": snapped_count,
                "child_search_models": child_search_audits,
                "block_model": model_audit,
                "solver_status": audit["solve"]["status"],
                "solver_wall_time_seconds": audit["solver_wall_time_seconds"],
                "child_bounds": {
                    child_id: child.lagrangian.conservative_lower_bound
                    for child_id, child in children.items()
                },
                "exact_replay_differences_dollars": {
                    child_id: child.gpu_lagrangian[
                        "final_gpu_host_replay_difference_dollars"
                    ]
                    for child_id, child in children.items()
                },
                "cpu_solution_data_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
