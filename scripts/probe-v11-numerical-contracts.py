#!/usr/bin/env python3
"""Tiny DGX-only checks for v11 scaling, cut replay, and complete MIP starts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.commitment_cuts import (
    add_commitment_upper_cuts,
    build_commitment_cardinality_cut,
)
from activsg_scopf.lagrangian import (
    RegionMasks,
    evaluate_lagrangian_bound,
    optimize_lagrangian_bound_cupy,
)
from activsg_scopf.network import build_network
from activsg_scopf.reduced import build_reduced_master
from activsg_scopf.solvers.cuopt import (
    FULL_MIP_START,
    native_scaling_vectors,
    solve_cuopt,
)
from tests.helpers import triangle_case


def main() -> None:
    case, _ = triangle_case()
    master = build_reduced_master(case, build_network(case))
    source_rows = master.index.generator_source_rows
    cut = build_commitment_cardinality_cut(
        generator_source_rows=source_rows + 1,
        subset_positions=np.arange(source_rows.size, dtype=np.int64),
        subset_id="probe_all_online_generators",
        branch_side="at_least",
        integer_threshold=1,
    )
    row_by_cut = add_commitment_upper_cuts(master, (cut,))
    _column_scale, row_scale = native_scaling_vectors(
        master.canonical,
        mode="power_system_equilibrated_safe_v3",
        base_mva=float(case.base_mva),
    )
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    coupling_scales = np.asarray(
        [
            row_scale[row.row_index]
            for row in sorted(master.coupling_rows, key=lambda item: item.row_name)
        ],
        dtype=np.float64,
    )
    cut_scales = np.asarray([row_scale[row_by_cut[cut.cut_id]]], dtype=np.float64)
    polished, gpu = optimize_lagrangian_bound_cupy(
        master,
        row_dual,
        RegionMasks.root(source_rows.size),
        relaxation_primal_objective=1_000_000.0,
        iterations=64,
        polyak_fraction=0.5,
        commitment_cuts=(cut,),
        initial_commitment_cut_dual=np.zeros(1, dtype=np.float64),
        coupling_row_scales=coupling_scales,
        commitment_cut_scales=cut_scales,
    )
    replay = evaluate_lagrangian_bound(
        master,
        polished,
        RegionMasks.root(source_rows.size),
        safety_margin_dollars=0.0,
        commitment_cuts=(cut,),
        commitment_cut_dual=np.asarray(gpu["best_commitment_cut_dual"]),
    )
    replay_difference = abs(float(gpu["best_raw_lower_bound"]) - replay.raw_lower_bound)
    if replay_difference > 1e-8:
        raise RuntimeError("Preconditioned GPU Lagrangian certificate did not replay")

    mip = CanonicalMILP()
    u = mip.add_variable("u", objective=1.0, lower=0.0, upper=1.0, integer=True)
    x = mip.add_variable("x", lower=0.0, upper=1.0)
    mip.add_row("link", {x: 1.0, u: -1.0}, lower=0.0, upper=0.0)
    complete_start = np.asarray([1.0, 1.0], dtype=np.float64)
    solved = solve_cuopt(
        mip,
        time_limit_seconds=2.0,
        mip_relative_gap=1e-3,
        mip_start_values=complete_start,
        mip_start_mode=FULL_MIP_START,
        native_scaling_mode="power_system_equilibrated_safe_v3",
        native_base_mva=100.0,
        log_to_console=True,
        mip_certificate_residual_tolerance=1e-6,
        mip_heuristics_only=True,
    )
    contract = solved.statistics["mip_start_native_contract"]
    if not bool(contract["contract_passed"]) or bool(
        contract.get("native_rejection_detected", True)
    ):
        raise RuntimeError("Complete cuOpt MIP-start contract failed")
    print(
        json.dumps(
            {
                "status": "passed",
                "gpu_lagrangian_replay_difference_dollars": replay_difference,
                "preconditioning": gpu["diagonal_preconditioning"],
                "mip_start_submitted_canonical_columns": contract[
                    "submitted_canonical_columns"
                ],
                "mip_start_submitted_native_columns": contract[
                    "submitted_native_columns"
                ],
                "mip_start_native_log_contract_passed": contract[
                    "native_log_contract_passed"
                ],
                "mip_status": solved.status,
                "mip_has_incumbent": solved.has_incumbent,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
