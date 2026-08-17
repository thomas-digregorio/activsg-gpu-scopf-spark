"""DGX-only tiny-fixture smoke for PDLP and the resident Lagrangian loop."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from activsg_scopf.config import load_config
from activsg_scopf.lagrangian import (
    RegionMasks,
    canonical_row_duals,
    evaluate_lagrangian_bound,
    optimize_lagrangian_bound_cupy,
)
from activsg_scopf.lagrangian_experiment import (
    _load_cpu_comparison,
    validate_lagrangian_experiment_config,
)
from activsg_scopf.matpower import read_matpower_case
from activsg_scopf.network import build_network
from activsg_scopf.phase_one import (
    build_phase_one_model,
    phase_one_certificate,
    replay_phase_one_certificate,
)
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    fix_commitments,
)
from activsg_scopf.screening import SecurityPair
from activsg_scopf.solvers.cuopt import native_scaling_vectors
from activsg_scopf.solvers.cuopt_lp import solve_cuopt_continuous_pdlp
from tests.helpers import triangle_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/configs/activsg500-gpu-lagrangian-v2.json"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    registration = validate_lagrangian_experiment_config(config)
    comparison = _load_cpu_comparison(config, registration)
    real_case = read_matpower_case(
        config.case_path,
        expected_sha256=config.raw["raw_inputs"]["case_sha256"],
    )
    real_master = build_reduced_master(
        real_case,
        build_network(real_case),
        coefficient_zero_tolerance=float(
            config.model["reduced_coefficient_zero_tolerance"]
        ),
    )
    column_scale, row_scale = native_scaling_vectors(
        real_master.canonical,
        mode=str(config.raw["platforms"]["dgx_spark"]["native_scaling_mode"]),
        base_mva=real_case.base_mva,
    )
    real_native_matrix = (
        real_master.canonical.matrix_csr()
        .multiply(column_scale)
        .multiply(row_scale[:, None])
        .tocsr()
    )
    real_nonzero = np.abs(real_native_matrix.data[real_native_matrix.data != 0.0])
    real_minimum_nonzero = float(np.min(real_nonzero))
    cleanup = real_master.coefficient_cleanup_audit
    if real_minimum_nonzero < 1e-8:
        raise RuntimeError(
            f"ACTIVSg500 cleaned native coefficient is still too small: "
            f"{real_minimum_nonzero}"
        )
    if not cleanup["solver_rows_are_relaxations_of_original_rows"]:
        raise RuntimeError("ACTIVSg500 coefficient cleanup lost its relaxation proof")
    case, _ = triangle_case()
    network = build_network(case)
    master = build_reduced_master(case, network)
    solved = solve_cuopt_continuous_pdlp(
        master.canonical,
        time_limit_seconds=10.0,
        optimality_tolerance=1e-8,
        primal_feasibility_tolerance=1e-6,
        certificate_residual_tolerance=1e-7,
        native_scaling_mode="power_system_per_unit_v1",
        native_base_mva=case.base_mva,
        log_to_console=True,
        per_constraint_residual=True,
        presolve=0,
    )
    if (
        solved.native_row_dual is None
        or solved.native_primal is None
        or solved.primal_objective is None
    ):
        raise RuntimeError("Tiny PDLP solve did not return primal/dual vectors")
    dispatch_column = master.index.dispatch_by_generator[0]
    master.canonical.add_row(
        "smoke_appended_security_row", {dispatch_column: 1.0}, upper=100.0
    )
    resolved = solve_cuopt_continuous_pdlp(
        master.canonical,
        time_limit_seconds=10.0,
        optimality_tolerance=1e-8,
        primal_feasibility_tolerance=1e-6,
        certificate_residual_tolerance=1e-7,
        native_scaling_mode="power_system_per_unit_v1",
        native_base_mva=case.base_mva,
        log_to_console=True,
        per_constraint_residual=True,
        presolve=0,
        initial_native_primal=solved.native_primal,
        initial_native_row_dual=solved.native_row_dual,
    )
    if resolved.native_row_dual is None or resolved.primal_objective is None:
        raise RuntimeError("Warm-started tiny PDLP re-solve did not return vectors")
    warm_start = resolved.statistics["warm_start"]
    if warm_start.get("initial_dual_zero_extended_count") != 1:
        raise RuntimeError(f"PDLP row-dual warm start was not extended once: {warm_start}")
    row_dual = canonical_row_duals(
        master,
        resolved.native_row_dual,
        native_scaling_mode="power_system_per_unit_v1",
        base_mva=case.base_mva,
    )
    polished, gpu = optimize_lagrangian_bound_cupy(
        master,
        row_dual,
        RegionMasks.root(1),
        relaxation_primal_objective=resolved.primal_objective,
        iterations=32,
        polyak_fraction=0.5,
    )
    replay = evaluate_lagrangian_bound(
        master,
        polished,
        RegionMasks.root(1),
        safety_margin_dollars=0.01,
    )
    difference = abs(float(gpu["best_raw_lower_bound"]) - replay.raw_lower_bound)
    if difference > 1e-6:
        raise RuntimeError(f"GPU/CPU tiny-certificate replay difference is {difference}")

    duplicate_master = build_reduced_master(case, network)
    first_pair = SecurityPair(11, 1, "upper", 0, 0, 1, 0.25)
    alias_pair = SecurityPair(12, 2, "upper", 1, 0, 1, 0.25)
    rows_before = duplicate_master.canonical.num_rows
    add_reduced_security_pairs(
        duplicate_master, network, (first_pair, alias_pair)
    )
    if duplicate_master.canonical.num_rows != rows_before + 1:
        raise RuntimeError("Exact duplicate security rows were not coalesced")
    if duplicate_master.security_pair_ids != {first_pair.pair_id, alias_pair.pair_id}:
        raise RuntimeError("Security-row coalescing lost a source pair identity")

    infeasible_master = build_reduced_master(case, network)
    fix_commitments(
        infeasible_master,
        np.asarray([True]),
        np.asarray([False]),
    )
    phase_model = build_phase_one_model(
        infeasible_master.canonical,
        base_mva=case.base_mva,
        maximum_violation_pu=1e6,
    )
    phase_solve = solve_cuopt_continuous_pdlp(
        phase_model,
        time_limit_seconds=10.0,
        optimality_tolerance=1e-8,
        primal_feasibility_tolerance=1e-6,
        certificate_residual_tolerance=1e-7,
        native_scaling_mode="power_system_per_unit_v1",
        native_base_mva=case.base_mva,
        log_to_console=True,
        per_constraint_residual=True,
        presolve=0,
    )
    if phase_solve.native_row_dual is None:
        raise RuntimeError("Tiny Phase-I solve did not return a row dual")
    _, phase_row_scale = native_scaling_vectors(
        phase_model,
        mode="power_system_per_unit_v1",
        base_mva=case.base_mva,
    )
    phase_certificate = phase_one_certificate(
        phase_model,
        phase_solve.native_row_dual * phase_row_scale,
        safety_margin_pu=1e-8,
        infeasibility_threshold_pu=1e-6,
    )
    phase_replay = replay_phase_one_certificate(
        phase_model, phase_certificate
    )
    if not phase_replay["prune_certified"]:
        raise RuntimeError(f"Tiny Phase-I prune was not certified: {phase_replay}")
    print(
        {
            "status": resolved.status,
            "objective": resolved.primal_objective,
            "solved_by_pdlp": resolved.statistics["solved_by_pdlp"],
            "native_integer_columns": resolved.statistics["native_integer_columns"],
            "dual_certificate_passed": resolved.statistics["dual_certificate"]["passed"],
            "warm_start_dual_zero_extended_count": warm_start[
                "initial_dual_zero_extended_count"
            ],
            "gpu_raw_lower_bound": gpu["best_raw_lower_bound"],
            "cpu_raw_lower_bound": replay.raw_lower_bound,
            "gpu_cpu_difference": difference,
            "device_state_persistent": gpu["device_state_persistent_across_iterations"],
            "cpu_comparison_objective": comparison["objective"],
            "cpu_comparison_canonical_hash": comparison["canonical_json_sha256"],
            "activsg500_native_minimum_nonzero": real_minimum_nonzero,
            "activsg500_cleanup": cleanup,
            "exact_duplicate_security_rows_removed": 1,
            "phase_one_conservative_lower_bound_pu": phase_replay[
                "conservative_lower_bound_pu"
            ],
            "phase_one_box_derived_upper_bound_pu": phase_model.column_upper[-1],
            "phase_one_prune_certified": phase_replay["prune_certified"],
        }
    )


if __name__ == "__main__":
    main()
