#!/usr/bin/env python3
"""Replay the five v10 GPU cuts through v11's real-size preconditioner.

This is a bounded component check only.  The registered v11 experiment never
reads the v10 result and constructs all primal and dual state from scratch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from activsg_scopf.commitment_cuts import (
    add_commitment_upper_cuts,
    commitment_upper_cut_from_record,
)
from activsg_scopf.config import load_config
from activsg_scopf.lagrangian import (
    RegionMasks,
    evaluate_lagrangian_bound,
    optimize_lagrangian_bound_cupy,
    replay_lagrangian_certificate,
)
from activsg_scopf.matpower import read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    security_pair_from_record,
)
from activsg_scopf.solvers.cuopt import native_scaling_vectors


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "activsg2000-gpu-lagrangian-v11.json")
    prior = json.loads(
        (
            root
            / "results"
            / "experiments"
            / "activsg2000-gpu-lagrangian-v10-dgx-spark.json"
        ).read_text(encoding="utf-8")
    )
    root_record = next(
        record
        for record in prior["solved_region_history"]
        if record["region_id"] == "r"
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
            record,
            catalog,
            lodf_absolute_tolerance=float(
                config.model["serialized_lodf_replay_tolerance"]
            ),
        )
        for record in root_record["security_pairs"]
    )
    add_reduced_security_pairs(master, network, pairs)
    source_rows = master.index.generator_source_rows + 1
    cuts = tuple(
        commitment_upper_cut_from_record(record["cut"], source_rows)
        for record in prior["commitment_feasibility_cuts"]
    )
    row_by_cut = add_commitment_upper_cuts(master, cuts)
    region = RegionMasks.root(source_rows.size)
    inherited = replay_lagrangian_certificate(
        master,
        root_record["lagrangian_certificate"],
        region,
        commitment_cuts_by_id={cut.cut_id: cut for cut in cuts},
    )
    row_by_name = {
        name: row for row, name in enumerate(master.canonical.row_names)
    }
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    for name, value in inherited.coupling_duals:
        row_dual[row_by_name[name]] = value
    _column_scale, row_scale = native_scaling_vectors(
        master.canonical,
        mode=str(
            config.raw["platforms"]["dgx_spark"]["native_scaling_mode"]
        ),
        base_mva=float(case.base_mva),
    )
    coupling_scales = np.asarray(
        [
            row_scale[row.row_index]
            for row in sorted(master.coupling_rows, key=lambda item: item.row_name)
        ],
        dtype=np.float64,
    )
    cut_scales = np.asarray(
        [row_scale[row_by_cut[cut.cut_id]] for cut in cuts], dtype=np.float64
    )
    polished_dual, gpu = optimize_lagrangian_bound_cupy(
        master,
        row_dual,
        region,
        relaxation_primal_objective=float(prior["objective"]),
        iterations=int(config.runtime["phase_lagrangian_gpu_iterations"]),
        polyak_fraction=float(
            config.raw["platforms"]["dgx_spark"]["lagrangian_polyak_fraction"]
        ),
        commitment_cuts=cuts,
        initial_commitment_cut_dual=np.zeros(len(cuts), dtype=np.float64),
        coupling_row_scales=coupling_scales,
        commitment_cut_scales=cut_scales,
    )
    replayed = evaluate_lagrangian_bound(
        master,
        polished_dual,
        region,
        safety_margin_dollars=float(
            config.raw["benchmark"]["certificate_safety_margin_dollars"]
        ),
        commitment_cuts=cuts,
        commitment_cut_dual=np.asarray(gpu["best_commitment_cut_dual"]),
    )
    replay_difference = abs(
        float(gpu["best_raw_lower_bound"]) - replayed.raw_lower_bound
    )
    if replay_difference > float(
        config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]
    ):
        raise RuntimeError("Real-size preconditioned certificate failed replay")
    if (
        replayed.conservative_lower_bound
        < inherited.conservative_lower_bound
    ):
        raise RuntimeError("Real-size preconditioned certificate regressed")
    print(
        json.dumps(
            {
                "status": "passed",
                "case_name": config.case_name,
                "generator_count": int(source_rows.size),
                "security_pair_count": len(pairs),
                "global_feasibility_cut_count": len(cuts),
                "inherited_conservative_lower_bound": (
                    inherited.conservative_lower_bound
                ),
                "preconditioned_conservative_lower_bound": (
                    replayed.conservative_lower_bound
                ),
                "improvement_dollars": (
                    replayed.conservative_lower_bound
                    - inherited.conservative_lower_bound
                ),
                "gpu_cpu_replay_difference_dollars": replay_difference,
                "diagonal_preconditioning": gpu["diagonal_preconditioning"],
                "registered_v11_reads_prior_result": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
