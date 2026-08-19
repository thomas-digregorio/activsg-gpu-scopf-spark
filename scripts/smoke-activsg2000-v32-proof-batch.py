#!/usr/bin/env python3
"""DGX-only real-case smoke for v32 proof-model conditioning and batching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from activsg_scopf.cardinality import exact_cost_type_groups
from activsg_scopf.commitment_cuts import build_commitment_cardinality_cut
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.lagrangian import RegionMasks, evaluate_lagrangian_bound
from activsg_scopf.lagrangian_experiment import (
    SolvedRegion,
    _prepare_region_master,
    _solve_v32_proof_only_hard_cardinality_pair,
    validate_lagrangian_experiment_config,
)
from activsg_scopf.matpower import read_matpower_case
from activsg_scopf.network import build_network
from activsg_scopf.solvers.cuopt_lp import ContinuousSolveResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/configs/activsg2000-gpu-lagrangian-v32.json"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    validate_lagrangian_experiment_config(config)
    case = read_matpower_case(
        config.case_path,
        expected_sha256=config.raw["raw_inputs"]["case_sha256"],
    )
    network = build_network(case)
    generator_count = int(np.count_nonzero(case.gen[:, 7] > 0.0))
    masks = RegionMasks.root(generator_count)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
        commitment_cuts=(),
    )
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    parent_evaluation = evaluate_lagrangian_bound(
        master,
        row_dual,
        masks,
        safety_margin_dollars=float(
            config.raw["benchmark"]["certificate_safety_margin_dollars"]
        ),
    )
    parent = SolvedRegion(
        region_id="smoke_root",
        masks=masks,
        master=master,
        solve=ContinuousSolveResult(
            status="SmokeCenter",
            optimal=False,
            primal_objective=None,
            dual_objective=None,
            values=None,
            native_primal=None,
            native_row_dual=None,
            solve_time_seconds=0.0,
            statistics={"error_status": "Success"},
        ),
        canonical_row_dual=row_dual,
        lagrangian=parent_evaluation,
        commitment=np.full(generator_count, 0.5, dtype=np.float64),
        security_pairs=(),
        rounds=[],
        final_screen={"new_violated_pairs": 0, "maximum_violation_pu": 0.0},
        gpu_lagrangian={"wall_time_seconds": 0.0},
    )
    eligible_groups = tuple(
        group for group in exact_cost_type_groups(master) if 2 <= group.size <= 8
    )
    if not eligible_groups:
        raise RuntimeError("ACTIVSg2000 has no bounded exact-type smoke group")
    support = eligible_groups[0]
    threshold = int(support.size // 2)
    source_rows = master.index.generator_source_rows + 1
    at_most = build_commitment_cardinality_cut(
        generator_source_rows=source_rows,
        subset_positions=support,
        subset_id="v32_real_case_smoke",
        branch_side="at_most",
        integer_threshold=threshold,
    )
    at_least = build_commitment_cardinality_cut(
        generator_source_rows=source_rows,
        subset_positions=support,
        subset_id="v32_real_case_smoke",
        branch_side="at_least",
        integer_threshold=threshold + 1,
    )
    children, audit = _solve_v32_proof_only_hard_cardinality_pair(
        child_specs=(
            ("smoke_child_0", masks, (at_most,)),
            ("smoke_child_1", masks, (at_least,)),
        ),
        parent=parent,
        case=case,
        config=config,
        deadline=Deadline(60.0, 0.0, 0.0),
    )
    floor = float(config.runtime["centered_dual_search_coefficient_zero_tolerance"])
    model_audit = audit["model"]
    if model_audit["block_count"] != 2:
        raise RuntimeError("v32 smoke did not create exactly two independent blocks")
    if model_audit["matrix_nonzero_minimum_absolute"] < floor:
        raise RuntimeError("v32 smoke retained a below-floor matrix coefficient")
    if model_audit["objective_nonzero_minimum_absolute"] < floor:
        raise RuntimeError("v32 smoke retained a below-floor objective coefficient")
    if model_audit["finite_column_bound_maximum_absolute"] > 1.0:
        raise RuntimeError("v32 smoke retained a finite column bound above one")
    if any(
        child.lagrangian.conservative_lower_bound
        < parent.lagrangian.conservative_lower_bound
        for child in children.values()
    ):
        raise RuntimeError("v32 smoke child regressed below its parent certificate")
    print(
        json.dumps(
            {
                "status": "pass",
                "case_name": config.case_name,
                "generator_count": generator_count,
                "hard_cardinality_support_source_rows": source_rows[support].tolist(),
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
