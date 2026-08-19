#!/usr/bin/env python3
"""Test centered GPU dual search from the pre-cover v12 root to v13 covers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from activsg_scopf.commitment_cuts import (
    add_commitment_upper_cuts,
    commitment_upper_cut_from_record,
)
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.lagrangian import replay_lagrangian_certificate
from activsg_scopf.lagrangian_experiment import (
    _certificate_dual_arrays,
    _masks_from_record,
    _run_centered_dual_search,
)
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.provenance import write_json_atomic
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    security_pair_from_record,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    started = time.perf_counter()
    root = Path("/workspace")
    config = load_config(root / "configs" / "activsg2000-gpu-lagrangian-v13.json")
    config.runtime.update(
        {
            "centered_dual_coupling_trust_radii": [10.0, 100.0],
            "centered_dual_commitment_cut_trust_radii": [1_000.0, 10_000.0],
            "centered_dual_seconds_per_pass": [3.0, 3.0],
            "centered_dual_maximum_new_coupling_rows": 384,
            "centered_dual_search_coefficient_zero_tolerance": 1e-8,
            "centered_dual_optimality_tolerance": 1e-8,
            "centered_dual_primal_feasibility_tolerance": 1e-7,
            "centered_dual_certificate_residual_tolerance": 1e-7,
        }
    )
    v12_path = (
        root
        / "results"
        / "experiments"
        / "activsg2000-gpu-lagrangian-v12-dgx-spark.json"
    )
    v13_path = (
        root
        / "results"
        / "experiments"
        / "activsg2000-gpu-lagrangian-v13-dgx-spark.json"
    )
    v12 = _read_json(v12_path)
    v13 = _read_json(v13_path)
    if not bool(v12.get("lagrangian_verification", {}).get("passed")) or not bool(
        v13.get("lagrangian_verification", {}).get("passed")
    ):
        raise RuntimeError("v12/v13 fixtures lack independent Lagrangian replay")
    v12_root = next(
        record for record in v12["solved_region_history"] if record["region_id"] == "r"
    )
    v13_root = next(
        record for record in v13["solved_region_history"] if record["region_id"] == "r"
    )

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
    masks = _masks_from_record(v12_root, source_rows)

    source_master = build_reduced_master(
        case,
        network,
        segments=int(config.model["pwl_segments"]),
        coefficient_zero_tolerance=float(
            config.model["reduced_coefficient_zero_tolerance"]
        ),
    )
    source_pairs = tuple(
        security_pair_from_record(
            record,
            catalog,
            lodf_absolute_tolerance=float(
                config.model["serialized_lodf_replay_tolerance"]
            ),
        )
        for record in v12_root["security_pairs"]
    )
    add_reduced_security_pairs(source_master, network, source_pairs)
    source_cuts = tuple(
        commitment_upper_cut_from_record(record, source_rows)
        for record in v12_root["commitment_upper_cuts"]
    )
    add_commitment_upper_cuts(source_master, source_cuts)
    inherited = replay_lagrangian_certificate(
        source_master,
        v12_root["lagrangian_certificate"],
        masks,
        commitment_cuts_by_id={cut.cut_id: cut for cut in source_cuts},
    )

    target_master = build_reduced_master(
        case,
        network,
        segments=int(config.model["pwl_segments"]),
        coefficient_zero_tolerance=float(
            config.model["reduced_coefficient_zero_tolerance"]
        ),
    )
    target_pairs = tuple(
        security_pair_from_record(
            record,
            catalog,
            lodf_absolute_tolerance=float(
                config.model["serialized_lodf_replay_tolerance"]
            ),
        )
        for record in v13_root["security_pairs"]
    )
    add_reduced_security_pairs(target_master, network, target_pairs)
    target_cuts = tuple(
        commitment_upper_cut_from_record(record, source_rows)
        for record in v13_root["commitment_upper_cuts"]
    )
    add_commitment_upper_cuts(target_master, target_cuts)
    initial_row_dual, initial_cut_dual = _certificate_dual_arrays(
        target_master, inherited, target_cuts
    )
    searched = _run_centered_dual_search(
        master=target_master,
        masks=masks,
        initial_row_dual=initial_row_dual,
        initial_commitment_cut_dual=initial_cut_dual,
        commitment_cuts=target_cuts,
        case=case,
        config=config,
        deadline=Deadline(30.0, 0.0, 0.0),
        stage="v20_precover_root",
    )
    serialized = searched.evaluation.as_dict(source_rows, compact=True)
    replayed = replay_lagrangian_certificate(
        target_master,
        serialized,
        masks,
        commitment_cuts_by_id={cut.cut_id: cut for cut in target_cuts},
    )
    replay_difference = abs(
        replayed.conservative_lower_bound
        - searched.evaluation.conservative_lower_bound
    )
    if replay_difference > float(
        config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]
    ):
        raise RuntimeError("v20 centered cover certificate failed serialized replay")

    v13_reference_bound = float(
        v13["analytic_capacity_cover_policy"]["bound_after"]
    )
    output = {
        "diagnostic_id": "activsg2000-v20-centered-cover",
        "status": "passed",
        "source_gpu_v12_result": str(v12_path.relative_to(root)),
        "source_gpu_v13_result": str(v13_path.relative_to(root)),
        "source_gpu_lagrangian_replays_passed": True,
        "source_case_sha256": config.raw["raw_inputs"]["case_sha256"],
        "source_contingency_sha256": config.raw["raw_inputs"][
            "contingency_sha256"
        ],
        "cpu_problem_solution_data_used": False,
        "initial_precover_bound": inherited.conservative_lower_bound,
        "centered_search_bound": searched.evaluation.conservative_lower_bound,
        "centered_search_lift_dollars": (
            searched.evaluation.conservative_lower_bound
            - inherited.conservative_lower_bound
        ),
        "v13_full_lp_reference_bound_for_diagnostic_only": v13_reference_bound,
        "difference_from_v13_full_lp_reference_dollars": (
            searched.evaluation.conservative_lower_bound - v13_reference_bound
        ),
        "serialized_certificate_replay_difference_dollars": replay_difference,
        "search": searched.audit,
        "search_lp_solution_used_as_bound": False,
        "exact_nonsmoothed_host_replay_is_bound_authority": True,
        "exact_source_pmin_pmax_changed": False,
        "mathematical_original_integer_feasible_set_changed": False,
        "total_wall_time_seconds": time.perf_counter() - started,
    }
    output_path = (
        root
        / "results"
        / "diagnostics"
        / "activsg2000-v20-centered-cover.json"
    )
    write_json_atomic(output, output_path)
    print(json.dumps(output, default=lambda value: value.tolist()))


if __name__ == "__main__":
    main()
