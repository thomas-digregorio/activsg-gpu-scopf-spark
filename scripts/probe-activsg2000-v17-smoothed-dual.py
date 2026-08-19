#!/usr/bin/env python3
"""Probe exact-bound-tracked smooth GPU dual ascent on v13 bottleneck leaves.

This development diagnostic reads only the immutable ACTIVSg2000 inputs and
the independently replayed v13 GPU certificate.  It does not use a laptop
commitment, dispatch, objective, or lower bound.  Smoothing proposes dual
multipliers, but every reported bound is the original nonsmoothed separable
binary-generator Lagrangian value and is independently replayed on the host.
"""

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
from activsg_scopf.lagrangian import (
    evaluate_lagrangian_bound,
    optimize_lagrangian_bound_cupy_smoothed,
    replay_lagrangian_certificate,
)
from activsg_scopf.lagrangian_experiment import (
    _certificate_dual_arrays,
    _lagrangian_precondition_scales,
    _masks_from_record,
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
    prior_path = (
        root
        / "results"
        / "experiments"
        / "activsg2000-gpu-lagrangian-v13-dgx-spark.json"
    )
    prior = _read_json(prior_path)
    if not bool(prior.get("lagrangian_verification", {}).get("passed")):
        raise RuntimeError("v13 diagnostic fixture lacks independent Lagrangian replay")
    if prior["source_manifest"]["source_identity"]["case_sha256"] != config.raw[
        "raw_inputs"
    ]["case_sha256"]:
        raise RuntimeError("v13 diagnostic fixture case identity changed")

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
    frontier = list(prior["frontier_regions"])
    minimum_prior_bound = min(
        float(record["lagrangian_certificate"]["conservative_lower_bound"])
        for record in frontier
    )
    bottlenecks = sorted(
        (
            record
            for record in frontier
            if abs(
                float(record["lagrangian_certificate"]["conservative_lower_bound"])
                - minimum_prior_bound
            )
            <= 1e-6
        ),
        key=lambda record: str(record["region_id"]),
    )
    if not bottlenecks:
        raise RuntimeError("v13 frontier has no bottleneck leaf")

    temperatures = (100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1)
    iterations = 512
    learning_rate = 10.0
    records: list[dict[str, Any]] = []
    for frontier_record in bottlenecks:
        region_started = time.perf_counter()
        region_id = str(frontier_record["region_id"])
        masks = _masks_from_record(frontier_record, source_rows)
        master = build_reduced_master(
            case,
            network,
            segments=int(config.model["pwl_segments"]),
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
            for pair_record in frontier_record["security_pairs"]
        )
        equivalence = frontier_record.get("security_row_equivalence", {})
        add_reduced_security_pairs(
            master,
            network,
            pairs,
            expected_representative_by_pair_id=equivalence.get(
                "representative_by_pair_id"
            ),
            equivalence_replay_tolerance=float(
                config.model["security_equivalence_replay_tolerance"]
            ),
        )
        commitment_cuts = tuple(
            commitment_upper_cut_from_record(cut_record, source_rows)
            for cut_record in frontier_record.get("commitment_upper_cuts", [])
        )
        add_commitment_upper_cuts(master, commitment_cuts)
        inherited = replay_lagrangian_certificate(
            master,
            frontier_record["lagrangian_certificate"],
            masks,
            commitment_cuts_by_id={cut.cut_id: cut for cut in commitment_cuts},
        )
        inherited_row_dual, inherited_cut_dual = _certificate_dual_arrays(
            master, inherited, commitment_cuts
        )
        coupling_scales, cut_scales = _lagrangian_precondition_scales(
            master=master,
            commitment_cuts=commitment_cuts,
            config=config,
            base_mva=float(case.base_mva),
        )
        polished_row_dual, gpu = optimize_lagrangian_bound_cupy_smoothed(
            master,
            inherited_row_dual,
            masks,
            temperatures_dollars=temperatures,
            iterations_per_temperature=iterations,
            learning_rate=learning_rate,
            commitment_cuts=commitment_cuts,
            initial_commitment_cut_dual=inherited_cut_dual,
            coupling_row_scales=coupling_scales,
            commitment_cut_scales=cut_scales,
        )
        polished_cut_dual = np.asarray(
            gpu["best_commitment_cut_dual"], dtype=np.float64
        )
        polished = evaluate_lagrangian_bound(
            master,
            polished_row_dual,
            masks,
            safety_margin_dollars=float(
                config.raw["benchmark"]["certificate_safety_margin_dollars"]
            ),
            commitment_cuts=commitment_cuts,
            commitment_cut_dual=polished_cut_dual,
        )
        gpu_replay_difference = abs(
            float(gpu["best_raw_lower_bound"]) - polished.raw_lower_bound
        )
        if gpu_replay_difference > float(
            config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]
        ):
            raise RuntimeError(
                f"Smoothed GPU exact bound failed host replay for {region_id}"
            )
        serialized = polished.as_dict(source_rows, compact=True)
        replayed = replay_lagrangian_certificate(
            master,
            serialized,
            masks,
            commitment_cuts_by_id={cut.cut_id: cut for cut in commitment_cuts},
        )
        serialized_replay_difference = abs(
            replayed.conservative_lower_bound - polished.conservative_lower_bound
        )
        if serialized_replay_difference > float(
            config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]
        ):
            raise RuntimeError(
                f"Smoothed GPU serialized certificate failed replay for {region_id}"
            )
        if polished.conservative_lower_bound + 1e-6 < inherited.conservative_lower_bound:
            raise RuntimeError(f"Smoothed GPU optimizer weakened {region_id}")
        records.append(
            {
                "region_id": region_id,
                "prior_conservative_lower_bound": inherited.conservative_lower_bound,
                "polished_conservative_lower_bound": polished.conservative_lower_bound,
                "improvement_dollars": (
                    polished.conservative_lower_bound
                    - inherited.conservative_lower_bound
                ),
                "gpu_raw_bound_host_replay_difference_dollars": (
                    gpu_replay_difference
                ),
                "serialized_certificate_replay_difference_dollars": (
                    serialized_replay_difference
                ),
                "security_pair_count": len(pairs),
                "commitment_cut_count": len(commitment_cuts),
                "wall_time_seconds": time.perf_counter() - region_started,
                "gpu_optimizer": gpu,
            }
        )

    output = {
        "diagnostic_id": "activsg2000-v17-smoothed-dual-probe",
        "status": "passed",
        "source_case_sha256": config.raw["raw_inputs"]["case_sha256"],
        "source_contingency_sha256": config.raw["raw_inputs"]["contingency_sha256"],
        "source_gpu_result": str(prior_path.relative_to(root)),
        "source_gpu_result_status": prior["status"],
        "source_gpu_lagrangian_replay_passed": True,
        "cpu_problem_solution_data_used": False,
        "smoothing_used_as_certificate": False,
        "exact_nonsmoothed_bound_scored_every_iteration": True,
        "exact_source_pmin_pmax_changed": False,
        "mathematical_original_integer_feasible_set_changed": False,
        "temperatures_dollars": list(temperatures),
        "iterations_per_temperature": iterations,
        "learning_rate": learning_rate,
        "bottleneck_region_count": len(bottlenecks),
        "prior_global_bound": minimum_prior_bound,
        "polished_bottleneck_bound": min(
            float(record["polished_conservative_lower_bound"])
            for record in records
        ),
        "minimum_bottleneck_improvement_dollars": min(
            float(record["improvement_dollars"]) for record in records
        ),
        "regions": records,
        "total_wall_time_seconds": time.perf_counter() - started,
    }
    output_path = (
        root
        / "results"
        / "diagnostics"
        / "activsg2000-v17-smoothed-dual-probe.json"
    )
    write_json_atomic(output, output_path)
    print(json.dumps({"output": str(output_path), **output}, default=str))


if __name__ == "__main__":
    main()
