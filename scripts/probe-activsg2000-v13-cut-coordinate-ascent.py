#!/usr/bin/env python3
"""Measure exact GPU commitment-cut coordinate ascent on v12 bottleneck leaves.

The preserved v12 result is a development fixture only.  The probe rebuilds
each selected leaf from immutable raw inputs and independently replays its
certificate before changing any multiplier.  No CPU commitment, dispatch,
objective, or bound is read or used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from activsg_scopf.commitment_cuts import commitment_upper_cut_from_record
from activsg_scopf.config import load_config
from activsg_scopf.lagrangian import (
    evaluate_lagrangian_bound,
    optimize_commitment_cut_duals_coordinate_cupy,
    replay_lagrangian_certificate,
)
from activsg_scopf.lagrangian_experiment import (
    _certificate_dual_arrays,
    _masks_from_record,
    _prepare_region_master,
    _validate_prepared_region_master,
)
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import security_pair_from_record


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    started = time.perf_counter()
    root = Path("/workspace")
    config = load_config(root / "configs" / "activsg2000-gpu-lagrangian-v12.json")
    prior = _read_json(
        root / "results" / "experiments" / "activsg2000-gpu-lagrangian-v12-dgx-spark.json"
    )
    if prior["frozen_identity"]["tag"] != "experiment-2000-gpu-lagrangian-v12":
        raise RuntimeError("The v13 probe development fixture identity changed")
    if prior["source_manifest"]["source_identity"]["case_sha256"] != config.raw[
        "raw_inputs"
    ]["case_sha256"]:
        raise RuntimeError("The v13 probe development fixture case hash changed")

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
    root_bound = float(prior["root_lagrangian_verification"]["recorded_global_lower_bound"])
    bottlenecks = [
        record
        for record in prior["frontier_regions"]
        if abs(float(record["lagrangian_certificate"]["conservative_lower_bound"]) - root_bound)
        <= 1e-6
    ]
    if len(bottlenecks) != 2:
        raise RuntimeError("The v12 bottleneck-leaf count changed")

    records: list[dict[str, Any]] = []
    strengthened_by_id: dict[str, float] = {}
    cycles = 32
    for record in sorted(bottlenecks, key=lambda item: str(item["region_id"])):
        masks = _masks_from_record(record, source_rows)
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
        cuts = tuple(
            commitment_upper_cut_from_record(cut_record, source_rows)
            for cut_record in record["commitment_upper_cuts"]
        )
        master = _prepare_region_master(
            case=case,
            network=network,
            config=config,
            masks=masks,
            initial_pairs=pairs,
            commitment_cuts=cuts,
        )
        _validate_prepared_region_master(master, masks, pairs, cuts)
        initial = replay_lagrangian_certificate(
            master,
            record["lagrangian_certificate"],
            masks,
            commitment_cuts_by_id={cut.cut_id: cut for cut in cuts},
        )
        row_dual, cut_dual = _certificate_dual_arrays(master, initial, cuts)
        gpu_started = time.perf_counter()
        optimized_cut_dual, audit = optimize_commitment_cut_duals_coordinate_cupy(
            master,
            row_dual,
            masks,
            commitment_cuts=cuts,
            initial_commitment_cut_dual=cut_dual,
            cycles=cycles,
        )
        gpu_wall = time.perf_counter() - gpu_started
        replayed = evaluate_lagrangian_bound(
            master,
            row_dual,
            masks,
            safety_margin_dollars=float(initial.safety_margin_dollars),
            commitment_cuts=cuts,
            commitment_cut_dual=optimized_cut_dual,
        )
        replay_difference = abs(
            float(audit["best_raw_lower_bound"]) - float(replayed.raw_lower_bound)
        )
        if replay_difference > float(
            config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]
        ):
            raise RuntimeError("GPU coordinate-ascent certificate failed exact replay")
        if replayed.conservative_lower_bound + 1e-6 < initial.conservative_lower_bound:
            raise RuntimeError("GPU coordinate ascent weakened a bottleneck certificate")
        initial_violations = sum(
            float(cut.coefficients @ initial.minimizing_commitment - cut.rhs) > 1e-9
            for cut in cuts
        )
        final_violations = sum(
            float(cut.coefficients @ replayed.minimizing_commitment - cut.rhs) > 1e-9
            for cut in cuts
        )
        strengthened_by_id[str(record["region_id"])] = replayed.conservative_lower_bound
        records.append(
            {
                "region_id": str(record["region_id"]),
                "cut_count": len(cuts),
                "cycles": cycles,
                "gpu_wall_time_seconds": gpu_wall,
                "initial_bound": initial.conservative_lower_bound,
                "final_bound": replayed.conservative_lower_bound,
                "improvement_dollars": (
                    replayed.conservative_lower_bound - initial.conservative_lower_bound
                ),
                "nonzero_initial_cut_duals": int(np.count_nonzero(cut_dual)),
                "nonzero_final_cut_duals": int(np.count_nonzero(optimized_cut_dual)),
                "initial_minimizer_violated_cut_count": initial_violations,
                "final_minimizer_violated_cut_count": final_violations,
                "initial_minimizer_commitment_count": int(
                    np.count_nonzero(initial.minimizing_commitment)
                ),
                "final_minimizer_commitment_count": int(
                    np.count_nonzero(replayed.minimizing_commitment)
                ),
                "cycle_raw_lower_bounds": np.asarray(
                    audit["cycle_raw_lower_bounds"], dtype=np.float64
                ).tolist(),
                "cpu_replay_difference_dollars": replay_difference,
                "host_transfer_during_coordinates": audit[
                    "host_transfer_during_coordinates"
                ],
            }
        )

    candidate_frontier_bounds = [
        strengthened_by_id.get(
            str(record["region_id"]),
            float(record["lagrangian_certificate"]["conservative_lower_bound"]),
        )
        for record in prior["frontier_regions"]
    ]
    candidate_global_bound = min(candidate_frontier_bounds)
    gpu_incumbent = float(prior["objective"])
    candidate_gap = (gpu_incumbent - candidate_global_bound) / abs(gpu_incumbent)
    requested_gap = float(config.model["mip_relative_gap_tolerance"])
    required_bound = gpu_incumbent * (1.0 - requested_gap)
    required_lift = max(0.0, required_bound - root_bound)
    achieved_lift = max(0.0, candidate_global_bound - root_bound)
    # A few cents of exact coordinate improvement are numerically real but do
    # not justify a full registered run when the certificate needs thousands
    # of dollars of lift.  Treat at least one percent of the required lift as
    # the component-probe gate for a material algorithmic improvement.
    material_lift_threshold = 0.01 * required_lift
    print(
        "V13_CUT_COORDINATE_ASCENT_PROBE="
        + json.dumps(
            {
                "passed": True,
                "development_component_only": True,
                "registered_future_experiment_reads_prior_result": False,
                "cpu_problem_solution_data_used": False,
                "fixture": "preserved_gpu_v12_bottleneck_leaves_only",
                "root_bound": root_bound,
                "candidate_global_bound": candidate_global_bound,
                "candidate_relative_gap": candidate_gap,
                "requested_relative_gap": requested_gap,
                "required_bound": required_bound,
                "required_lift_dollars": required_lift,
                "achieved_lift_dollars": achieved_lift,
                "achieved_fraction_of_required_lift": (
                    1.0 if required_lift == 0.0 else achieved_lift / required_lift
                ),
                "material_lift_threshold_dollars": material_lift_threshold,
                "material_global_improvement": (
                    required_lift == 0.0 or achieved_lift >= material_lift_threshold
                ),
                "regions": records,
                "total_probe_wall_time_seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
