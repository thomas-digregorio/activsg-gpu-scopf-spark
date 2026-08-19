import json
from pathlib import Path

import numpy as np
import pytest

from activsg_scopf import lagrangian_experiment as experiment_module
from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.commitment_cuts import build_commitment_cardinality_cut
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.errors import PrimalCandidateRejected, ScopfError
from activsg_scopf.lagrangian import RegionMasks
from activsg_scopf.lagrangian_experiment import (
    ACTIVSG2000_ALL_EXPERIMENT_IDS,
    ACTIVSG2000_BEST_FIRST_REPAIR_EXPERIMENT_IDS,
    ACTIVSG2000_DIVERSIFIED_PRIMAL_SEARCH_EXPERIMENT_IDS,
    ACTIVSG2000_EXPERIMENT_ID,
    ACTIVSG2000_EXPERIMENT_ID_SEQUENCE,
    ACTIVSG2000_FIXED_COMMITMENT_COST_PROJECTION_EXPERIMENT_IDS,
    ACTIVSG2000_FIXED_COMMITMENT_FEASIBILITY_EXPERIMENT_IDS,
    ACTIVSG2000_GPU_PRIMAL_HEURISTIC_EXPERIMENT_IDS,
    ACTIVSG2000_PHASE_ONE_CHILD_EXPERIMENT_IDS,
    ACTIVSG2000_PHASE_ONE_FIRST_EXPERIMENT_IDS,
    ACTIVSG2000_V4_EXPERIMENT_ID,
    ACTIVSG2000_V5_EXPERIMENT_ID,
    ACTIVSG2000_V6_EXPERIMENT_ID,
    ACTIVSG2000_V7_EXPERIMENT_ID,
    ACTIVSG2000_V8_EXPERIMENT_ID,
    ACTIVSG2000_V9_EXPERIMENT_ID,
    ACTIVSG2000_V10_EXPERIMENT_ID,
    ACTIVSG2000_V11_EXPERIMENT_ID,
    ACTIVSG2000_V12_EXPERIMENT_ID,
    ACTIVSG2000_V13_EXPERIMENT_ID,
    ACTIVSG2000_V14_CENTERED_DUAL_FIX,
    ACTIVSG2000_V14_EXPERIMENT_ID,
    ACTIVSG2000_V14_RUNTIME,
    ACTIVSG2000_V15_EXPERIMENT_ID,
    ACTIVSG2000_V15_MINIMIZER_CUT_FIX,
    ACTIVSG2000_V15_RUNTIME,
    ACTIVSG2000_V16_BENDERS_COVER_FIX,
    ACTIVSG2000_V16_EXPERIMENT_ID,
    ACTIVSG2000_V16_RUNTIME,
    ACTIVSG2000_V17_EXPERIMENT_ID,
    ACTIVSG2000_V17_PRIMAL_DIVERSIFICATION_FIX,
    ACTIVSG2000_V17_RUNTIME,
    ACTIVSG2000_V18_EXPERIMENT_ID,
    ACTIVSG2000_V18_PRIMAL_SEARCH_FIX,
    ACTIVSG2000_V18_RUNTIME,
    ACTIVSG2000_V19_EXPERIMENT_ID,
    ACTIVSG2000_V19_NUMERICAL_EXTENDED_COVER_FIX,
    ACTIVSG2000_V19_RUNTIME,
    ACTIVSG2000_V20_EXPERIMENT_ID,
    ACTIVSG2000_V20_RUNTIME,
    ACTIVSG2000_V20_SECURITY_ROW_NUMERICAL_FIX,
    ACTIVSG2000_V21_CANDIDATE_PIPELINE_REGISTRATION_FIX,
    ACTIVSG2000_V21_EXPERIMENT_ID,
    ACTIVSG2000_V21_RUNTIME,
    ACTIVSG2000_V22_EXPERIMENT_ID,
    ACTIVSG2000_V22_LOWER_BOUND_THROUGHPUT_FIX,
    ACTIVSG2000_V22_RUNTIME,
    ACTIVSG2000_V23_EXPERIMENT_ID,
    ACTIVSG2000_V23_NUMERICAL_HARD_CARDINALITY_FIX,
    ACTIVSG2000_V23_RUNTIME,
    ACTIVSG2000_V24_CUPY_LEXSORT_FIX,
    ACTIVSG2000_V24_EXPERIMENT_ID,
    ACTIVSG2000_V24_RUNTIME,
    ACTIVSG2000_V25_EXPERIMENT_ID,
    ACTIVSG2000_V25_GPU_DUAL_SEARCH_FIX,
    ACTIVSG2000_V25_RUNTIME,
    ACTIVSG2000_V26_EXPERIMENT_ID,
    ACTIVSG2000_V26_NUMERICAL_PROOF_THROUGHPUT_FIX,
    ACTIVSG2000_V26_RUNTIME,
    ACTIVSG2000_V27_EXPERIMENT_ID,
    ACTIVSG2000_V27_MIP_START_NUMERICAL_FIX,
    ACTIVSG2000_V27_RUNTIME,
    ACTIVSG2000_V28_ARGMIN_REPLAY_NUMERICAL_FIX,
    ACTIVSG2000_V28_EXPERIMENT_ID,
    ACTIVSG2000_V28_RUNTIME,
    ACTIVSG2000_V29_EXPERIMENT_ID,
    ACTIVSG2000_V29_PROOF_EVIDENCE_SERIALIZATION_FIX,
    ACTIVSG2000_V29_RUNTIME,
    ACTIVSG2000_V30_EXPERIMENT_ID,
    ACTIVSG2000_V30_FULL_COUPLING_DUAL_FIX,
    ACTIVSG2000_V30_RUNTIME,
    ACTIVSG2000_V31_EXPERIMENT_ID,
    ACTIVSG2000_V31_PRIMAL_AND_PROOF_THROUGHPUT_FIX,
    ACTIVSG2000_V31_RUNTIME,
    ACTIVSG2000_V32_EXPERIMENT_ID,
    ACTIVSG2000_V32_NUMERICAL_AND_BATCH_THROUGHPUT_FIX,
    ACTIVSG2000_V32_RUNTIME,
    EXPERIMENT_ID,
    EXPERIMENT_TAG,
    PrimalCandidatePolicy,
    RegionAttemptRejected,
    _activsg2000_solver_path_registration,
    _gpu_lagrangian_wall_time,
    _load_cpu_comparison,
    _map_phase_one_dual_to_source_native,
    _prepare_region_master,
    _refresh_region_with_bounded_cost_dual_search,
    _region_pmin_pmax_capacity_gate,
    _relative_gap,
    _replay_cleanup_audit_comparison,
    _run_phase_one_attempt,
    _select_analytic_capacity_cover_cuts,
    _solve_region,
    _validate_prepared_region_master,
    validate_lagrangian_experiment_config,
)
from activsg_scopf.matpower import GEN_STATUS
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import CouplingRow, build_reduced_master
from activsg_scopf.screening import (
    ContingencyScreener,
    ScreenResult,
    SecurityPair,
)
from activsg_scopf.solvers.cuopt import native_scaling_vectors
from activsg_scopf.solvers.cuopt_lp import ContinuousSolveResult

from .helpers import triangle_case

ROOT = Path(__file__).resolve().parents[1]


def test_registered_activsg500_lagrangian_config_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v1.json")
    registration = validate_lagrangian_experiment_config(config)
    assert config.benchmark_id == EXPERIMENT_ID
    assert config.raw["benchmark"]["required_git_tag"] == EXPERIMENT_TAG
    assert config.runtime["deadline_seconds"] == 600.0
    assert config.model["mip_relative_gap_tolerance"] == 1e-3
    assert registration["profile"]["integer_solver"] == "none"
    assert registration["profile"]["branch_and_bound"] is False
    assert registration["profile"]["console_logging"] is True
    comparison = _load_cpu_comparison(config, registration)
    assert comparison["status"] == "optimal_verified"
    assert comparison["objective"] == pytest.approx(79410.651432182)
    assert comparison["total_wall_time_seconds"] == pytest.approx(2.702380099988659)


def test_registered_v2_bugfix_config_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v2.json")
    registration = validate_lagrangian_experiment_config(config)
    assert config.benchmark_id == "activsg500-gpu-lagrangian-v2"
    assert registration["benchmark"]["required_git_tag"] == ("experiment-500-gpu-lagrangian-v2")
    assert config.model["reduced_coefficient_zero_tolerance"] == 1e-14
    assert registration["benchmark"]["bugfix_change"]["pdlp_primal_gate"] == (
        "never_screen_or_add_rows_from_primal_infeasible_vector"
    )


def test_registered_v3_controller_config_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v3.json")
    registration = validate_lagrangian_experiment_config(config)
    assert config.benchmark_id == "activsg500-gpu-lagrangian-v3"
    assert registration["benchmark"]["required_git_tag"] == ("experiment-500-gpu-lagrangian-v3")
    assert registration["benchmark"]["controller_change"]["candidate_budget"] == (
        "15_seconds_total_with_5_second_pdlp_slices"
    )
    policy = PrimalCandidatePolicy.from_config(config)
    assert policy.total_seconds == 15.0
    assert policy.maximum_round_seconds == 5.0
    assert policy.stagnation_window_rounds == 2
    assert policy.dual_divergence_multiple == 1e6
    config.raw["runtime"]["maximum_primal_candidate_seconds"] = 16.0
    with pytest.raises(ScopfError, match="candidate policy changed"):
        validate_lagrangian_experiment_config(config)


def test_registered_v4_phase_one_controller_config_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v4.json")
    registration = validate_lagrangian_experiment_config(config)
    assert config.benchmark_id == "activsg500-gpu-lagrangian-v4"
    assert registration["benchmark"]["required_git_tag"] == ("experiment-500-gpu-lagrangian-v4")
    assert registration["benchmark"]["controller_change"]["infeasible_leaf_gate"] == (
        "replayable_gpu_phase_one_box_dual_certificate"
    )
    assert config.model["serialized_lodf_replay_tolerance"] == 1e-12
    candidate = PrimalCandidatePolicy.from_config(config)
    region = PrimalCandidatePolicy.from_config(config, scope="disjunctive_region")
    assert candidate.cold_restart_attempts == 1
    assert region.total_seconds == 15.0
    assert region.cold_restart_attempts == 1
    config.raw["runtime"]["phase_one_time_limit_seconds"] = 16.0
    with pytest.raises(ScopfError, match="bounded-region policy changed"):
        validate_lagrangian_experiment_config(config)


def test_registered_v5_replay_bugfix_config_is_fail_closed() -> None:
    v4 = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v4.json")
    v5 = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v5.json")
    registration = validate_lagrangian_experiment_config(v5)

    assert v5.benchmark_id == "activsg500-gpu-lagrangian-v5"
    assert registration["benchmark"]["required_git_tag"] == ("experiment-500-gpu-lagrangian-v5")
    assert v5.raw["raw_inputs"] == v4.raw["raw_inputs"]
    assert v5.model == v4.model
    assert v5.runtime == v4.runtime
    assert v5.raw["platforms"] == v4.raw["platforms"]
    assert registration["benchmark"]["bugfix_change"]["phase_one_row_identity"] == (
        "semantic_source_row_and_side_order_independent_v1"
    )
    assert registration["benchmark"]["bugfix_change"]["gap_bookkeeping"] == (
        "refresh_at_every_frontier_checkpoint"
    )
    v5.raw["benchmark"]["bugfix_change"]["gap_bookkeeping"] = "changed"
    with pytest.raises(ScopfError, match="v5 bugfix identity changed"):
        validate_lagrangian_experiment_config(v5)


def test_registered_v6_phase_one_first_config_is_fail_closed() -> None:
    v5 = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v5.json")
    v6 = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v6.json")
    registration = validate_lagrangian_experiment_config(v6)

    assert v6.benchmark_id == "activsg500-gpu-lagrangian-v6"
    assert registration["benchmark"]["required_git_tag"] == ("experiment-500-gpu-lagrangian-v6")
    assert v6.raw["raw_inputs"] == v5.raw["raw_inputs"]
    assert v6.model == v5.model
    assert v6.raw["platforms"] == v5.raw["platforms"]
    v6_runtime_without_precheck = dict(v6.runtime)
    assert v6_runtime_without_precheck.pop("precheck_phase_one_time_limit_seconds") == 2.0
    assert v6_runtime_without_precheck == v5.runtime
    assert registration["benchmark"]["controller_change"]["prune_authority"] == (
        "positive_independently_replayable_phase_one_dual_only"
    )
    v6.raw["runtime"]["precheck_phase_one_time_limit_seconds"] = 3.0
    with pytest.raises(ScopfError, match="short Phase-I budget changed"):
        validate_lagrangian_experiment_config(v6)


def test_registered_v7_source_row_mapping_bugfix_is_fail_closed() -> None:
    v6 = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v6.json")
    v7 = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v7.json")
    registration = validate_lagrangian_experiment_config(v7)

    assert v7.benchmark_id == "activsg500-gpu-lagrangian-v7"
    assert registration["benchmark"]["required_git_tag"] == ("experiment-500-gpu-lagrangian-v7")
    assert v7.raw["raw_inputs"] == v6.raw["raw_inputs"]
    assert v7.model == v6.model
    assert v7.runtime == v6.runtime
    assert v7.raw["platforms"] == v6.raw["platforms"]
    assert registration["benchmark"]["bugfix_change"] == {
        "comparison_baseline": "activsg500-gpu-lagrangian-v6",
        "prepared_region_commitment_column_mapping": (
            "generator_source_row_key_to_canonical_commitment_column_v1"
        ),
        "failed_v6_run_preserved": True,
    }
    v7.raw["benchmark"]["bugfix_change"]["failed_v6_run_preserved"] = False
    with pytest.raises(ScopfError, match="v7 bugfix identity changed"):
        validate_lagrangian_experiment_config(v7)


def test_registered_activsg2000_lagrangian_config_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v1.json")
    registration = validate_lagrangian_experiment_config(config)

    assert config.benchmark_id == ACTIVSG2000_EXPERIMENT_ID
    assert config.case_name == "ACTIVSg2000"
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v1")
    assert config.runtime["deadline_seconds"] == 1800.0
    assert config.runtime["precheck_phase_one_time_limit_seconds"] == 10.0
    assert config.model["mip_relative_gap_tolerance"] == 1e-3
    assert config.model["reduced_coefficient_zero_tolerance"] == 1e-9
    assert registration["profile"]["integer_solver"] == "none"
    assert registration["profile"]["branch_and_bound"] is False
    comparison = _load_cpu_comparison(config, registration)
    assert comparison["status"] == "optimal_verified"
    assert comparison["objective"] == pytest.approx(1133479.3855011363)
    assert comparison["total_wall_time_seconds"] == pytest.approx(1007.3702709000063)

    config.raw["runtime"]["deadline_seconds"] = 1801.0
    with pytest.raises(ScopfError, match="deadline changed"):
        validate_lagrangian_experiment_config(config)


def test_registered_activsg2000_v4_utilization_config_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v4.json")
    registration = validate_lagrangian_experiment_config(config)

    assert config.benchmark_id == ACTIVSG2000_V4_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v4")
    assert config.raw["platforms"]["dgx_spark"]["native_scaling_mode"] == (
        "power_system_equilibrated_v2"
    )
    assert config.runtime["maximum_primal_repairs"] == 12
    assert config.runtime["parallel_child_solver_contexts"] == 2
    assert registration["benchmark"]["utilization_change"]["exact_source_pmin_changed"] is False
    config.raw["runtime"]["network_repair_pair_search_limit"] = 31
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(config)


def test_registered_activsg2000_v5_numerical_fix_is_fail_closed() -> None:
    v4 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v4.json")
    v5 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v5.json")
    registration = validate_lagrangian_experiment_config(v5)

    assert v5.benchmark_id == ACTIVSG2000_V5_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v5")
    assert v5.raw["raw_inputs"] == v4.raw["raw_inputs"]
    assert v5.model == v4.model
    assert v5.runtime["root_canonical_residual_refinement_attempts"] == 1
    assert v5.runtime["root_canonical_residual_refinement_optimality_tolerance"] == 1e-10
    assert v5.raw["platforms"]["dgx_spark"]["native_scaling_mode"] == (
        "power_system_equilibrated_safe_v3"
    )
    assert registration["benchmark"]["numerical_fix"]["failed_v4_run_preserved"] is True
    v5.raw["runtime"]["root_canonical_residual_refinement_attempts"] = 2
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v5)


def test_registered_activsg2000_v6_numerical_runtime_fix_is_fail_closed() -> None:
    v5 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v5.json")
    v6 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v6.json")
    registration = validate_lagrangian_experiment_config(v6)

    assert v6.benchmark_id == ACTIVSG2000_V6_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v6")
    assert v6.raw["raw_inputs"] == v5.raw["raw_inputs"]
    assert v6.model == v5.model
    assert v6.runtime["deadline_seconds"] == 990.0
    assert v6.runtime["gpu_primal_heuristics_seconds"] == 105.0
    assert v6.runtime["gpu_primal_seed_seconds"] == 75.0
    assert v6.raw["platforms"]["dgx_spark"]["pdlp_solver_mode_native"] == 1
    assert v6.raw["platforms"]["dgx_spark"]["save_best_primal_so_far"] is True
    fix = registration["benchmark"]["numerical_and_runtime_fix"]
    assert fix["primal_generator"] == (
        "reduced_gpu_heuristics_then_sparse_full_gpu_heuristics_with_complete_gpu_feasible_start_v3"
    )
    assert fix["partial_mip_start_policy"] == (
        "never_submit_unextended_commitment_as_native_full_assignment"
    )
    v6.raw["runtime"]["maximum_failed_split_attempts"] = 63
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v6)


def test_registered_activsg2000_v7_numerical_robustness_is_fail_closed() -> None:
    v6 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v6.json")
    v7 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v7.json")
    registration = validate_lagrangian_experiment_config(v7)

    assert v7.benchmark_id == ACTIVSG2000_V7_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v7")
    assert v7.raw["raw_inputs"] == v6.raw["raw_inputs"]
    assert v7.model == v6.model
    assert v7.runtime == v6.runtime
    v7_profile = dict(v7.raw["platforms"]["dgx_spark"])
    assert v7_profile == v6.raw["platforms"]["dgx_spark"]
    fix = registration["benchmark"]["numerical_robustness_fix"]
    assert fix["same_shape_pdlp_continuation"] == ("optimal_complete_state_else_raw_primal_dual_v3")
    assert fix["cost_polish_initial_dual_policy"] == (
        "verified_phase_one_primal_only_because_phase_one_dual_has_different_objective"
    )
    assert fix["cost_polish_formulation"] == (
        "exact_fixed_commitment_convex_pwl_epigraph_without_fixed_u_or_segment_columns_v1"
    )
    assert fix["cost_polish_solver"] == (
        "single_cuopt_gpu_pdlp_slice_then_cupy_balance_and_feasible_segment_v2"
    )
    assert fix["barrier_policy"] == (
        "disabled_after_first_newton_step_nan_factorization_diagnostic"
    )
    assert fix["pricing_policy"] == ("fail_closed_if_repair_breaks_primal_dual_complementarity")
    assert fix["repair_residual_budget_fraction"] == 0.5
    v7.raw["benchmark"]["numerical_robustness_fix"]["failed_v6_run_preserved"] = False
    with pytest.raises(ScopfError, match="numerical-robustness identity changed"):
        validate_lagrangian_experiment_config(v7)


def test_registered_activsg2000_v8_certificate_controller_is_fail_closed() -> None:
    v7 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v7.json")
    v8 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v8.json")
    registration = validate_lagrangian_experiment_config(v8)

    assert v8.benchmark_id == ACTIVSG2000_V8_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v8")
    assert v8.raw["raw_inputs"] == v7.raw["raw_inputs"]
    assert v8.model == v7.model
    assert v8.runtime["maximum_primal_repairs"] == 64
    assert v8.runtime["maximum_frontier_regions"] == 192
    assert v8.runtime["frontier_replay_after_every_split"] is False
    assert v8.runtime["compact_lagrangian_certificates"] is True
    assert v8.runtime["fixed_commitment_secure_seed_tolerance_fraction"] == 0.25
    fix = registration["benchmark"]["certificate_controller_fix"]
    assert fix["exact_source_pmin_changed"] is False
    assert fix["gpu_cut_lower_bound_used"] is False
    assert fix["secure_seed_numerical_margin"] == (
        "quarter_tolerance_phase_one_then_expanded_row_reprojection_v2"
    )
    assert fix["cpu_commitment_dispatch_objective_or_bound_seeded"] is False
    v8.raw["runtime"]["maximum_primal_repairs"] = 63
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v8)


def test_registered_activsg2000_v9_compact_replay_fix_is_fail_closed() -> None:
    v8 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v8.json")
    v9 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v9.json")
    registration = validate_lagrangian_experiment_config(v9)

    assert v9.benchmark_id == ACTIVSG2000_V9_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v9")
    assert v9.raw["raw_inputs"] == v8.raw["raw_inputs"]
    assert v9.model == v8.model
    assert v9.runtime == v8.runtime
    assert v9.raw["platforms"] == v8.raw["platforms"]
    controller = registration["benchmark"]["certificate_controller_fix"]
    assert controller["lagrangian_certificate_serialization"] == (
        "sparse_nonzero_dual_order_independent_identity_v3"
    )
    fix = registration["benchmark"]["compact_replay_fix"]
    assert fix["failed_v8_run_preserved"] is True
    assert fix["derived_fp64_vector_hash_gate_removed"] is True
    assert fix["exact_source_pmin_changed"] is False
    assert fix["cpu_commitment_dispatch_objective_or_bound_seeded"] is False
    v9.raw["benchmark"]["compact_replay_fix"]["failed_v8_run_preserved"] = False
    with pytest.raises(ScopfError, match="compact-replay identity changed"):
        validate_lagrangian_experiment_config(v9)


def test_registered_activsg2000_v10_cardinality_refinement_is_fail_closed() -> None:
    v9 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v9.json")
    v10 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v10.json")
    registration = validate_lagrangian_experiment_config(v10)

    assert v10.benchmark_id == ACTIVSG2000_V10_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v10")
    assert v10.raw["raw_inputs"] == v9.raw["raw_inputs"]
    assert v10.model == v9.model
    assert v10.runtime["maximum_frontier_regions"] == 256
    assert v10.runtime["phase_lagrangian_gpu_iterations"] == 512
    assert registration["benchmark"]["cardinality_refinement"]["child_phase_one"] == (
        "disabled_after_zero_of_83_v9_prunes"
    )
    assert (
        registration["benchmark"]["cardinality_refinement"]["minimum_cardinality_subset_size"] == 2
    )
    assert registration["benchmark"]["cardinality_refinement"]["binary_fallback"] == (
        "only_after_no_fractional_multi_unit_sum_remains"
    )
    assert registration["benchmark"]["cardinality_refinement"]["child_warm_start"] == (
        "row_name_mapped_parent_dual_only_because_parent_primal_violates_"
        "the_new_cardinality_branch_v2"
    )
    assert (
        registration["benchmark"]["cardinality_refinement"][
            "cpu_commitment_dispatch_objective_or_bound_seeded"
        ]
        is False
    )
    v10.raw["benchmark"]["cardinality_refinement"][
        "mathematical_original_integer_optimum_changed"
    ] = True
    with pytest.raises(ScopfError, match="cardinality identity changed"):
        validate_lagrangian_experiment_config(v10)


def test_registered_activsg2000_v11_numerical_runtime_fix_is_fail_closed() -> None:
    v10 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v10.json")
    v11 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v11.json")
    registration = validate_lagrangian_experiment_config(v11)

    assert v11.benchmark_id == ACTIVSG2000_V11_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v11")
    assert v11.raw["raw_inputs"] == v10.raw["raw_inputs"]
    assert v11.model == v10.model
    assert v11.runtime["parallel_child_solver_contexts"] == 1
    assert v11.runtime["gpu_primal_seed_seconds"] == 1.0
    assert v11.runtime["minimum_refinement_launch_seconds"] == 195.0
    assert v11.runtime["lagrangian_diagonal_preconditioning"] is True
    assert (
        registration["benchmark"]["certificate_controller_fix"]["gpu_cut_lower_bound_used"] is True
    )
    fix = registration["benchmark"]["numerical_runtime_fix"]
    assert fix["failed_v10_run_preserved"] is True
    assert fix["cpu_commitment_dispatch_objective_or_bound_seeded"] is False
    v11.raw["runtime"]["parallel_child_solver_contexts"] = 2
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v11)


def test_registered_activsg2000_v12_numerical_throughput_fix_is_fail_closed() -> None:
    v11 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v11.json")
    v12 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v12.json")
    registration = validate_lagrangian_experiment_config(v12)

    assert v12.benchmark_id == ACTIVSG2000_V12_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v12")
    assert v12.raw["raw_inputs"] == v11.raw["raw_inputs"]
    assert v12.model == v11.model
    assert v12.runtime["minimum_refinement_launch_seconds"] == 99.0
    assert v12.runtime["phase_one_precheck_optimality_tolerance"] == 1e-10
    assert v12.runtime["feasibility_cut_coefficient_zero_tolerance"] == 1e-8
    assert v12.runtime["maximum_child_phase_one_rounds"] == 3
    assert v12.runtime["child_cost_dual_seed_seconds"] == 12.0
    fix = registration["benchmark"]["numerical_throughput_fix"]
    assert fix["v11_result_preserved"] is True
    assert fix["exact_source_pmin_changed"] is False
    assert fix["mathematical_original_integer_optimum_changed"] is False
    assert fix["cpu_commitment_dispatch_objective_or_bound_seeded"] is False
    v12.raw["runtime"]["phase_one_precheck_optimality_tolerance"] = 1e-8
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v12)


def test_registered_activsg2000_v13_capacity_cover_fix_is_fail_closed() -> None:
    v12 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v12.json")
    v13 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v13.json")
    registration = validate_lagrangian_experiment_config(v13)

    assert v13.benchmark_id == ACTIVSG2000_V13_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v13"
    )
    assert v13.raw["raw_inputs"] == v12.raw["raw_inputs"]
    assert v13.model["reduced_coefficient_zero_tolerance"] == 2e-6
    assert v13.model["mip_relative_gap_tolerance"] == 1e-3
    assert v13.runtime["analytic_capacity_cover_rounds"] == 1
    assert v13.runtime["analytic_capacity_cover_maximum_cuts"] == 16
    fix = registration["benchmark"]["numerical_cover_fix"]
    assert fix["v12_result_preserved"] is True
    assert fix["exact_source_pmin_changed"] is False
    assert fix["mathematical_original_integer_optimum_changed"] is False
    assert fix["cpu_commitment_dispatch_objective_or_bound_seeded"] is False
    v13.raw["runtime"]["analytic_capacity_cover_rounds"] = 2
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v13)


def test_registered_activsg2000_v14_centered_dual_fix_is_fail_closed() -> None:
    v13 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v13.json")
    v14 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v14.json")
    registration = validate_lagrangian_experiment_config(v14)

    assert v14.benchmark_id == ACTIVSG2000_V14_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v14"
    )
    assert v14.raw["raw_inputs"] == v13.raw["raw_inputs"]
    assert v14.model == v13.model
    assert v14.runtime == ACTIVSG2000_V14_RUNTIME
    assert v14.runtime["deadline_seconds"] == 900.0
    assert v14.runtime["gpu_primal_heuristics_seconds"] == 15.0
    assert v14.runtime["minimum_refinement_launch_seconds"] == 81.0
    assert v14.runtime["centered_dual_child_maximum_passes"] == 1
    assert v14.runtime["centered_dual_cover_maximum_passes"] == 2
    assert registration["benchmark"]["centered_dual_fix"] == (
        ACTIVSG2000_V14_CENTERED_DUAL_FIX
    )
    assert registration["benchmark"]["centered_dual_fix"][
        "cpu_commitment_dispatch_objective_or_bound_seeded"
    ] is False
    v14.raw["runtime"]["centered_dual_coupling_trust_radii"][0] = 11.0
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v14)


def test_registered_activsg2000_v15_minimizer_cut_fix_is_fail_closed() -> None:
    v14 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v14.json")
    v15 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v15.json")
    registration = validate_lagrangian_experiment_config(v15)

    assert v15.benchmark_id == ACTIVSG2000_V15_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v15"
    )
    assert v15.raw["raw_inputs"] == v14.raw["raw_inputs"]
    assert v15.model == v14.model
    assert v15.runtime == ACTIVSG2000_V15_RUNTIME
    assert v15.runtime["deadline_seconds"] == 900.0
    assert v15.runtime["maximum_primal_candidate_seconds"] == 30.0
    assert v15.runtime["maximum_primal_candidate_round_seconds"] == 15.0
    assert v15.runtime["always_run_gpu_primal_heuristics"] is False
    assert v15.runtime["minimizer_feasibility_cut_enabled"] is True
    assert v15.runtime["minimizer_feasibility_cut_maximum_iterations"] == 24
    assert registration["benchmark"]["minimizer_cut_fix"] == (
        ACTIVSG2000_V15_MINIMIZER_CUT_FIX
    )
    assert registration["benchmark"]["minimizer_cut_fix"][
        "cpu_commitment_dispatch_objective_or_bound_seeded"
    ] is False
    v15.raw["runtime"]["minimizer_feasibility_cut_maximum_iterations"] = 25
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v15)


def test_registered_activsg2000_v16_benders_cover_fix_is_fail_closed() -> None:
    v15 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v15.json")
    v16 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v16.json")
    registration = validate_lagrangian_experiment_config(v16)

    assert v16.benchmark_id == ACTIVSG2000_V16_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v16"
    )
    assert v16.raw["raw_inputs"] == v15.raw["raw_inputs"]
    assert v16.model == v15.model
    assert v16.runtime == ACTIVSG2000_V16_RUNTIME
    assert v16.runtime["deadline_seconds"] == 900.0
    assert v16.runtime["benders_feasibility_cover_enabled"] is True
    assert registration["benchmark"]["benders_cover_fix"] == (
        ACTIVSG2000_V16_BENDERS_COVER_FIX
    )
    assert registration["benchmark"]["benders_cover_fix"][
        "cover_selection_uses_cpu_solution_data"
    ] is False
    v16.raw["runtime"]["benders_feasibility_cover_enabled"] = False
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v16)


def test_registered_activsg2000_v17_primal_diversification_is_fail_closed() -> None:
    v16 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v16.json")
    v17 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v17.json")
    registration = validate_lagrangian_experiment_config(v17)

    assert v17.benchmark_id == ACTIVSG2000_V17_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v17"
    )
    assert v17.raw["raw_inputs"] == v16.raw["raw_inputs"]
    assert v17.model == v16.model
    assert v17.runtime == ACTIVSG2000_V17_RUNTIME
    assert v17.runtime["deadline_seconds"] == 900.0
    assert v17.runtime["alternative_primal_candidate_maximum_attempts"] == 18
    assert v17.runtime["alternative_primal_candidate_wall_seconds"] == 120.0
    assert v17.runtime[
        "alternative_primal_candidate_minimum_remaining_solver_seconds"
    ] == 420.0
    assert registration["benchmark"]["primal_diversification_fix"] == (
        ACTIVSG2000_V17_PRIMAL_DIVERSIFICATION_FIX
    )
    assert registration["benchmark"]["primal_diversification_fix"][
        "candidate_selection_uses_cpu_solution_data"
    ] is False
    v17.raw["runtime"]["alternative_primal_candidate_maximum_attempts"] = 19
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v17)


def test_registered_activsg2000_v18_primal_search_is_fail_closed() -> None:
    v17 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v17.json")
    v18 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v18.json")
    registration = validate_lagrangian_experiment_config(v18)

    assert v18.benchmark_id == ACTIVSG2000_V18_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v18"
    )
    assert v18.raw["raw_inputs"] == v17.raw["raw_inputs"]
    assert v18.model == v17.model
    assert v18.runtime == ACTIVSG2000_V18_RUNTIME
    assert v18.runtime["deadline_seconds"] == 900.0
    assert v18.runtime["alternative_primal_candidate_maximum_attempts"] == 44
    assert v18.runtime["alternative_primal_candidate_wall_seconds"] == 180.0
    assert v18.runtime[
        "alternative_primal_candidate_minimum_remaining_solver_seconds"
    ] == 300.0
    assert v18.runtime["balanced_type_rounding_target_offsets"] == [-1, 0, 1]
    assert registration["benchmark"]["primal_diversification_fix"] == (
        ACTIVSG2000_V17_PRIMAL_DIVERSIFICATION_FIX
    )
    assert registration["benchmark"]["primal_search_fix"] == (
        ACTIVSG2000_V18_PRIMAL_SEARCH_FIX
    )
    assert registration["benchmark"]["primal_search_fix"][
        "candidate_selection_uses_cpu_solution_data"
    ] is False
    v18.raw["runtime"]["balanced_type_rounding_target_offsets"] = [-2, 0, 1]
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v18)


def test_registered_activsg2000_v19_numerical_extended_cover_fix_is_fail_closed() -> None:
    v18 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v18.json")
    v19 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v19.json")
    registration = validate_lagrangian_experiment_config(v19)

    assert v19.benchmark_id == ACTIVSG2000_V19_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v19"
    )
    assert v19.raw["raw_inputs"] == v18.raw["raw_inputs"]
    assert v19.runtime == ACTIVSG2000_V19_RUNTIME
    assert v19.runtime["deadline_seconds"] == 900.0
    assert v19.model["reduced_coefficient_zero_tolerance"] == 1e-5
    assert v19.runtime["feasibility_cut_coefficient_zero_tolerance"] == 1e-5
    assert v19.runtime["centered_dual_search_coefficient_zero_tolerance"] == 1e-5
    assert v19.runtime["centered_dual_search_objective_zero_tolerance"] == 1e-5
    assert v19.runtime["extended_cover_enabled"] is True
    assert v19.runtime["benders_parent_master_policy"] == "cover_only_when_available"
    assert registration["benchmark"]["numerical_extended_cover_fix"] == (
        ACTIVSG2000_V19_NUMERICAL_EXTENDED_COVER_FIX
    )
    assert registration["benchmark"]["numerical_extended_cover_fix"][
        "cpu_solution_data_used"
    ] is False
    v19.raw["runtime"]["extended_cover_enabled"] = False
    with pytest.raises(ScopfError, match="runtime policy changed"):
        validate_lagrangian_experiment_config(v19)


def test_registered_activsg2000_v20_security_row_numerical_fix_is_fail_closed() -> None:
    v19 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v19.json")
    v20 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v20.json")
    registration = validate_lagrangian_experiment_config(v20)

    assert v20.benchmark_id == ACTIVSG2000_V20_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v20"
    )
    assert v20.raw["raw_inputs"] == v19.raw["raw_inputs"]
    assert v20.runtime == ACTIVSG2000_V20_RUNTIME
    assert v20.model["reduced_coefficient_zero_tolerance"] == 1e-5
    assert v20.model["security_row_maximum_raw_violation_envelope_pu"] == 5e-6
    assert (
        v20.model["security_row_maximum_raw_violation_envelope_pu"]
        + v20.model["model_residual_tolerance_pu"]
        < v20.model["security_violation_tolerance_pu"]
    )
    assert registration["benchmark"]["security_row_numerical_fix"] == (
        ACTIVSG2000_V20_SECURITY_ROW_NUMERICAL_FIX
    )
    assert registration["benchmark"]["security_row_numerical_fix"][
        "cpu_solution_data_used"
    ] is False
    v20.raw["model"]["security_row_maximum_raw_violation_envelope_pu"] = 9e-6
    with pytest.raises(ScopfError, match="security-row envelope"):
        validate_lagrangian_experiment_config(v20)


def test_registered_activsg2000_v21_candidate_pipeline_is_fail_closed() -> None:
    v20 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v20.json")
    v21 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v21.json")
    registration = validate_lagrangian_experiment_config(v21)

    assert v21.benchmark_id == ACTIVSG2000_V21_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v21"
    )
    assert v21.raw["raw_inputs"] == v20.raw["raw_inputs"]
    assert v21.model == v20.model
    assert v21.runtime == ACTIVSG2000_V21_RUNTIME
    assert registration["benchmark"]["candidate_pipeline_registration_fix"] == (
        ACTIVSG2000_V21_CANDIDATE_PIPELINE_REGISTRATION_FIX
    )
    assert ACTIVSG2000_V21_EXPERIMENT_ID in ACTIVSG2000_EXPERIMENT_ID_SEQUENCE
    assert set(ACTIVSG2000_EXPERIMENT_ID_SEQUENCE) == set(
        ACTIVSG2000_ALL_EXPERIMENT_IDS
    )
    for feature_ids in (
        ACTIVSG2000_FIXED_COMMITMENT_FEASIBILITY_EXPERIMENT_IDS,
        ACTIVSG2000_FIXED_COMMITMENT_COST_PROJECTION_EXPERIMENT_IDS,
        ACTIVSG2000_PHASE_ONE_CHILD_EXPERIMENT_IDS,
        ACTIVSG2000_PHASE_ONE_FIRST_EXPERIMENT_IDS,
        ACTIVSG2000_DIVERSIFIED_PRIMAL_SEARCH_EXPERIMENT_IDS,
        ACTIVSG2000_BEST_FIRST_REPAIR_EXPERIMENT_IDS,
        ACTIVSG2000_GPU_PRIMAL_HEURISTIC_EXPERIMENT_IDS,
    ):
        assert ACTIVSG2000_V21_EXPERIMENT_ID in feature_ids
    assert all(
        _activsg2000_solver_path_registration(ACTIVSG2000_V21_EXPERIMENT_ID).values()
    )
    v21.raw["benchmark"]["candidate_pipeline_registration_fix"][
        "v20_candidate_policy_was_null"
    ] = False
    with pytest.raises(ScopfError, match="candidate-pipeline identity changed"):
        validate_lagrangian_experiment_config(v21)


def test_registered_activsg2000_v22_lower_bound_throughput_is_fail_closed() -> None:
    v21 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v21.json")
    v22 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v22.json")
    registration = validate_lagrangian_experiment_config(v22)

    assert v22.benchmark_id == ACTIVSG2000_V22_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v22"
    )
    assert v22.raw["raw_inputs"] == v21.raw["raw_inputs"]
    assert v22.model == v21.model
    assert v22.runtime == ACTIVSG2000_V22_RUNTIME
    assert v22.runtime["intermediate_raw_input_replays"] is False
    assert v22.runtime["precheck_phase_one_time_limit_seconds"] == 5.0
    assert v22.runtime["maximum_child_phase_one_rounds"] == 2
    assert v22.runtime["child_cost_dual_seed_seconds"] == 30.0
    assert v22.runtime["minimum_refinement_launch_seconds"] == 95.0
    assert v22.runtime["post_cut_root_cost_dual_seconds"] == 90.0
    assert registration["benchmark"]["lower_bound_throughput_fix"] == (
        ACTIVSG2000_V22_LOWER_BOUND_THROUGHPUT_FIX
    )
    assert ACTIVSG2000_V22_EXPERIMENT_ID in ACTIVSG2000_EXPERIMENT_ID_SEQUENCE
    assert all(
        _activsg2000_solver_path_registration(ACTIVSG2000_V22_EXPERIMENT_ID).values()
    )
    v22.raw["benchmark"]["lower_bound_throughput_fix"][
        "independent_raw_input_replay_policy"
    ] = "changed"
    with pytest.raises(ScopfError, match="lower-bound throughput identity changed"):
        validate_lagrangian_experiment_config(v22)


def test_registered_activsg2000_v23_numerical_hard_cardinality_is_fail_closed() -> None:
    v22 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v22.json")
    v23 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v23.json")
    registration = validate_lagrangian_experiment_config(v23)

    assert v23.benchmark_id == ACTIVSG2000_V23_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v23"
    )
    assert v23.raw["raw_inputs"] == v22.raw["raw_inputs"]
    assert v23.model == v22.model
    assert v23.runtime == ACTIVSG2000_V23_RUNTIME
    assert v23.runtime["minimum_refinement_launch_seconds"] == 35.0
    assert v23.runtime["child_cost_dual_seed_seconds"] == 0.0
    assert v23.runtime["post_cut_root_cost_dual_seconds"] == 0.0
    assert v23.raw["platforms"]["dgx_spark"]["native_scaling_mode"] == (
        "power_system_equilibrated_certified_v4"
    )
    assert registration["benchmark"]["numerical_hard_cardinality_fix"] == (
        ACTIVSG2000_V23_NUMERICAL_HARD_CARDINALITY_FIX
    )
    assert ACTIVSG2000_V23_EXPERIMENT_ID in ACTIVSG2000_EXPERIMENT_ID_SEQUENCE
    assert all(
        _activsg2000_solver_path_registration(ACTIVSG2000_V23_EXPERIMENT_ID).values()
    )
    v23.raw["benchmark"]["numerical_hard_cardinality_fix"][
        "child_cost_pdlp_removed"
    ] = False
    with pytest.raises(ScopfError, match="numerical/hard-cardinality identity changed"):
        validate_lagrangian_experiment_config(v23)


def test_registered_activsg2000_v24_cupy_lexsort_fix_is_fail_closed() -> None:
    v23 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v23.json")
    v24 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v24.json")
    registration = validate_lagrangian_experiment_config(v24)

    assert v24.benchmark_id == ACTIVSG2000_V24_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v24"
    )
    assert v24.raw["raw_inputs"] == v23.raw["raw_inputs"]
    assert v24.model == v23.model
    assert v24.runtime == ACTIVSG2000_V24_RUNTIME
    assert registration["benchmark"]["numerical_hard_cardinality_fix"] == (
        ACTIVSG2000_V23_NUMERICAL_HARD_CARDINALITY_FIX
    )
    assert registration["benchmark"]["cupy_lexsort_fix"] == (
        ACTIVSG2000_V24_CUPY_LEXSORT_FIX
    )
    assert ACTIVSG2000_V24_EXPERIMENT_ID in ACTIVSG2000_EXPERIMENT_ID_SEQUENCE
    assert all(
        _activsg2000_solver_path_registration(ACTIVSG2000_V24_EXPERIMENT_ID).values()
    )
    v24.raw["benchmark"]["cupy_lexsort_fix"]["gpu_host_exact_commitment_replay_required"] = (
        False
    )
    with pytest.raises(ScopfError, match="CuPy-lexsort identity changed"):
        validate_lagrangian_experiment_config(v24)


def test_registered_activsg2000_v25_gpu_dual_search_is_fail_closed() -> None:
    v24 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v24.json")
    v25 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v25.json")
    registration = validate_lagrangian_experiment_config(v25)

    assert v25.benchmark_id == ACTIVSG2000_V25_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v25"
    )
    assert v25.raw["raw_inputs"] == v24.raw["raw_inputs"]
    assert v25.model == v24.model
    assert v25.runtime == ACTIVSG2000_V25_RUNTIME
    assert v25.runtime["minimum_refinement_launch_seconds"] == 50.0
    assert v25.runtime["hard_cardinality_adam_split_budget_seconds"] == 12.0
    assert registration["benchmark"]["cupy_lexsort_fix"] == (
        ACTIVSG2000_V24_CUPY_LEXSORT_FIX
    )
    assert registration["benchmark"]["gpu_dual_search_fix"] == (
        ACTIVSG2000_V25_GPU_DUAL_SEARCH_FIX
    )
    assert ACTIVSG2000_V25_EXPERIMENT_ID in ACTIVSG2000_EXPERIMENT_ID_SEQUENCE
    assert all(
        _activsg2000_solver_path_registration(ACTIVSG2000_V25_EXPERIMENT_ID).values()
    )
    v25.raw["benchmark"]["gpu_dual_search_fix"][
        "nonfinite_update_policy"
    ] = "changed"
    with pytest.raises(ScopfError, match="v25 dual-search identity changed"):
        validate_lagrangian_experiment_config(v25)


def test_registered_activsg2000_v26_proof_only_path_is_fail_closed() -> None:
    v25 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v25.json")
    v26 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v26.json")
    registration = validate_lagrangian_experiment_config(v26)

    assert v26.benchmark_id == ACTIVSG2000_V26_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v26"
    )
    assert v26.raw["raw_inputs"] == v25.raw["raw_inputs"]
    assert v26.model == v25.model
    assert v26.runtime == ACTIVSG2000_V26_RUNTIME
    assert v26.runtime["minimum_refinement_launch_seconds"] == 19.0
    assert v26.runtime["alternative_primal_candidate_maximum_attempts"] == 0
    assert v26.runtime["always_run_gpu_primal_heuristics"] is True
    assert registration["benchmark"]["numerical_proof_throughput_fix"] == (
        ACTIVSG2000_V26_NUMERICAL_PROOF_THROUGHPUT_FIX
    )
    assert ACTIVSG2000_V26_EXPERIMENT_ID in ACTIVSG2000_EXPERIMENT_ID_SEQUENCE
    assert all(
        _activsg2000_solver_path_registration(ACTIVSG2000_V26_EXPERIMENT_ID).values()
    )
    v26.raw["benchmark"]["numerical_proof_throughput_fix"][
        "below_floor_policy"
    ] = "changed"
    with pytest.raises(ScopfError, match="v26 numerical/proof-throughput"):
        validate_lagrangian_experiment_config(v26)


def test_registered_activsg2000_v27_mip_start_fix_is_fail_closed() -> None:
    v26 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v26.json")
    v27 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v27.json")
    registration = validate_lagrangian_experiment_config(v27)

    assert v27.benchmark_id == ACTIVSG2000_V27_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v27"
    )
    assert v27.raw["raw_inputs"] == v26.raw["raw_inputs"]
    assert v27.model == v26.model
    assert v27.runtime == ACTIVSG2000_V27_RUNTIME
    assert v27.runtime["gpu_primal_heuristics_seconds"] == 60.0
    assert v27.runtime["full_mip_start_polish_seconds"] == 20.0
    assert registration["benchmark"]["numerical_proof_throughput_fix"] == (
        ACTIVSG2000_V26_NUMERICAL_PROOF_THROUGHPUT_FIX
    )
    assert registration["benchmark"]["mip_start_numerical_fix"] == (
        ACTIVSG2000_V27_MIP_START_NUMERICAL_FIX
    )
    assert ACTIVSG2000_V27_EXPERIMENT_ID in ACTIVSG2000_EXPERIMENT_ID_SEQUENCE
    assert all(
        _activsg2000_solver_path_registration(ACTIVSG2000_V27_EXPERIMENT_ID).values()
    )
    v27.raw["benchmark"]["mip_start_numerical_fix"][
        "start_polish_conditioning"
    ] = "changed"
    with pytest.raises(ScopfError, match="v27 MIP-start numerical-fix"):
        validate_lagrangian_experiment_config(v27)


def test_registered_activsg2000_v28_argmin_replay_fix_is_fail_closed() -> None:
    v27 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v27.json")
    v28 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v28.json")
    registration = validate_lagrangian_experiment_config(v28)

    assert v28.benchmark_id == ACTIVSG2000_V28_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v28"
    )
    assert v28.raw["raw_inputs"] == v27.raw["raw_inputs"]
    assert v28.model == v27.model
    assert v28.runtime == ACTIVSG2000_V28_RUNTIME == ACTIVSG2000_V27_RUNTIME
    assert registration["benchmark"]["mip_start_numerical_fix"] == (
        ACTIVSG2000_V27_MIP_START_NUMERICAL_FIX
    )
    assert registration["benchmark"]["argmin_replay_numerical_fix"] == (
        ACTIVSG2000_V28_ARGMIN_REPLAY_NUMERICAL_FIX
    )
    assert ACTIVSG2000_V28_EXPERIMENT_ID in ACTIVSG2000_EXPERIMENT_ID_SEQUENCE
    assert all(
        _activsg2000_solver_path_registration(ACTIVSG2000_V28_EXPERIMENT_ID).values()
    )
    v28.raw["benchmark"]["argmin_replay_numerical_fix"][
        "alternate_argmin_gate"
    ] = "changed"
    with pytest.raises(ScopfError, match="v28 argmin-replay numerical-fix"):
        validate_lagrangian_experiment_config(v28)


def test_registered_activsg2000_v29_proof_serialization_fix_is_fail_closed() -> None:
    v28 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v28.json")
    v29 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v29.json")
    registration = validate_lagrangian_experiment_config(v29)

    assert v29.benchmark_id == ACTIVSG2000_V29_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v29"
    )
    assert v29.raw["raw_inputs"] == v28.raw["raw_inputs"]
    assert v29.model == v28.model
    assert v29.runtime == ACTIVSG2000_V29_RUNTIME == ACTIVSG2000_V28_RUNTIME
    assert registration["benchmark"]["argmin_replay_numerical_fix"] == (
        ACTIVSG2000_V28_ARGMIN_REPLAY_NUMERICAL_FIX
    )
    assert registration["benchmark"]["proof_evidence_serialization_fix"] == (
        ACTIVSG2000_V29_PROOF_EVIDENCE_SERIALIZATION_FIX
    )
    assert ACTIVSG2000_V29_EXPERIMENT_ID in ACTIVSG2000_EXPERIMENT_ID_SEQUENCE
    assert all(
        _activsg2000_solver_path_registration(ACTIVSG2000_V29_EXPERIMENT_ID).values()
    )
    v29.raw["benchmark"]["proof_evidence_serialization_fix"][
        "serialization_fix"
    ] = "changed"
    with pytest.raises(ScopfError, match="v29 proof-evidence serialization-fix"):
        validate_lagrangian_experiment_config(v29)


def test_registered_activsg2000_v30_full_coupling_dual_is_fail_closed() -> None:
    v29 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v29.json")
    v30 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v30.json")
    registration = validate_lagrangian_experiment_config(v30)

    assert v30.benchmark_id == ACTIVSG2000_V30_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v30"
    )
    assert v30.raw["raw_inputs"] == v29.raw["raw_inputs"]
    assert v30.model == v29.model
    assert v30.runtime == ACTIVSG2000_V30_RUNTIME
    assert v30.runtime["full_coupling_root_dual_enabled"] is True
    assert v30.runtime["full_coupling_root_dual_seconds"] == 90.0
    assert v30.runtime[
        "full_coupling_root_dual_minimum_remaining_solver_seconds"
    ] == 180.0
    assert registration["benchmark"]["proof_evidence_serialization_fix"] == (
        ACTIVSG2000_V29_PROOF_EVIDENCE_SERIALIZATION_FIX
    )
    assert registration["benchmark"]["full_coupling_dual_fix"] == (
        ACTIVSG2000_V30_FULL_COUPLING_DUAL_FIX
    )
    assert ACTIVSG2000_V30_EXPERIMENT_ID in ACTIVSG2000_EXPERIMENT_ID_SEQUENCE
    assert all(
        _activsg2000_solver_path_registration(ACTIVSG2000_V30_EXPERIMENT_ID).values()
    )
    v30.raw["benchmark"]["full_coupling_dual_fix"]["proposal_role"] = "changed"
    with pytest.raises(ScopfError, match="v30 full-coupling dual identity changed"):
        validate_lagrangian_experiment_config(v30)


def test_registered_activsg2000_v31_primal_and_proof_throughput_is_fail_closed() -> None:
    v30 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v30.json")
    v31 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v31.json")
    registration = validate_lagrangian_experiment_config(v31)

    assert v31.benchmark_id == ACTIVSG2000_V31_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v31"
    )
    assert v31.raw["raw_inputs"] == v30.raw["raw_inputs"]
    assert v31.model == v30.model
    assert v31.runtime == ACTIVSG2000_V31_RUNTIME
    assert v31.runtime["maximum_frontier_regions"] == 512
    assert v31.runtime[
        "proof_only_hard_cardinality_pdlp_seconds_per_child"
    ] == 0.5
    assert v31.runtime["proof_only_checkpoint_interval_splits"] == 16
    assert v31.runtime["alternative_primal_candidate_maximum_attempts"] == 44
    assert v31.runtime["alternative_primal_candidate_wall_seconds"] == 180.0
    assert v31.runtime["always_run_gpu_primal_heuristics"] is False
    assert "full_coupling_root_dual_enabled" not in v31.runtime
    assert registration["benchmark"]["primal_and_proof_throughput_fix"] == (
        ACTIVSG2000_V31_PRIMAL_AND_PROOF_THROUGHPUT_FIX
    )
    assert ACTIVSG2000_V31_EXPERIMENT_ID in ACTIVSG2000_EXPERIMENT_ID_SEQUENCE
    assert all(
        _activsg2000_solver_path_registration(ACTIVSG2000_V31_EXPERIMENT_ID).values()
    )
    v31.raw["benchmark"]["primal_and_proof_throughput_fix"][
        "proof_child_proposal"
    ] = "changed"
    with pytest.raises(ScopfError, match="v31 primal/proof throughput identity"):
        validate_lagrangian_experiment_config(v31)


def test_registered_activsg2000_v32_numerical_and_batch_fix_is_fail_closed() -> None:
    v31 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v31.json")
    v32 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v32.json")
    registration = validate_lagrangian_experiment_config(v32)

    assert v32.benchmark_id == ACTIVSG2000_V32_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v32")
    assert v32.raw["raw_inputs"] == v31.raw["raw_inputs"]
    assert v32.model == v31.model
    assert v32.runtime == ACTIVSG2000_V32_RUNTIME
    assert v32.runtime["centered_dual_search_coefficient_zero_tolerance"] == 1e-4
    assert v32.runtime["centered_dual_search_objective_zero_tolerance"] == 1e-4
    assert v32.runtime["proof_only_hard_cardinality_pdlp_seconds_per_batch"] == 0.75
    assert v32.runtime["alternative_primal_candidate_maximum_attempts"] == 72
    assert v32.runtime["minimizer_feasibility_cut_enabled"] is False
    assert registration["benchmark"]["numerical_and_batch_throughput_fix"] == (
        ACTIVSG2000_V32_NUMERICAL_AND_BATCH_THROUGHPUT_FIX
    )
    assert ACTIVSG2000_EXPERIMENT_ID_SEQUENCE[-1] == ACTIVSG2000_V32_EXPERIMENT_ID
    assert all(_activsg2000_solver_path_registration(ACTIVSG2000_V32_EXPERIMENT_ID).values())
    v32.raw["benchmark"]["numerical_and_batch_throughput_fix"]["proposal_batching"] = "changed"
    with pytest.raises(ScopfError, match="v32 numerical/batch throughput identity"):
        validate_lagrangian_experiment_config(v32)



def test_gpu_alternate_argmin_requires_exact_host_objective_replay() -> None:
    host = experiment_module.LagrangianEvaluation(
        raw_lower_bound=10.0,
        conservative_lower_bound=9.99,
        safety_margin_dollars=0.01,
        projected_dual_sign_violation=0.0,
        coupling_duals=(),
        effective_dispatch_coefficients=np.zeros(2),
        on_subproblem_values=np.asarray([0.0, 0.0]),
        minimizing_commitment=np.asarray([0, 0], dtype=np.int8),
    )
    audit = experiment_module._audit_gpu_minimizing_commitment_replay(
        candidate=np.asarray([1, 0], dtype=np.int8),
        host_evaluation=host,
        masks=RegionMasks.root(2),
        hard_cardinality_cuts=(),
        tolerance_dollars=1e-6,
    )

    assert audit["alternate_minimizer_accepted"] is True
    assert audit["hamming_distance_from_host_minimizer"] == 1
    assert audit["candidate_objective_gap_dollars"] == 0.0
    assert audit["host_minimizer_authoritative"] is True

    separated = experiment_module.LagrangianEvaluation(
        raw_lower_bound=10.0,
        conservative_lower_bound=9.99,
        safety_margin_dollars=0.01,
        projected_dual_sign_violation=0.0,
        coupling_duals=(),
        effective_dispatch_coefficients=np.zeros(2),
        on_subproblem_values=np.asarray([1e-4, 0.0]),
        minimizing_commitment=np.asarray([0, 0], dtype=np.int8),
    )
    with pytest.raises(ScopfError, match="exact host objective replay"):
        experiment_module._audit_gpu_minimizing_commitment_replay(
            candidate=np.asarray([1, 0], dtype=np.int8),
            host_evaluation=separated,
            masks=RegionMasks.root(2),
            hard_cardinality_cuts=(),
            tolerance_dollars=1e-6,
        )


def test_gpu_lagrangian_timing_accepts_centered_and_legacy_audits() -> None:
    assert _gpu_lagrangian_wall_time(
        {"gpu_lagrangian_evaluation": {"wall_time_seconds": 1.25}}
    ) == pytest.approx(1.25)
    assert _gpu_lagrangian_wall_time(
        {"gpu_lagrangian_evaluation": {"total_wall_time_seconds": 2.5}}
    ) == pytest.approx(2.5)
    with pytest.raises(ScopfError, match="recognized wall-time"):
        _gpu_lagrangian_wall_time({"gpu_lagrangian_evaluation": {}})


def test_analytic_capacity_cover_separator_selects_raw_row_proof() -> None:
    case, _ = triangle_case()
    case.gen[1, GEN_STATUS] = 1.0
    network = build_network(case)
    master = build_reduced_master(case, network)
    source_rows = master.index.generator_source_rows
    dispatch_columns = [
        master.index.dispatch_by_generator[int(generator)] for generator in source_rows
    ]
    row = master.canonical.add_row(
        "test_cover_capacity_row",
        {dispatch_columns[0]: 1.0, dispatch_columns[1]: 1.0},
        upper=20.0,
    )
    master.coupling_rows.append(
        CouplingRow(
            row_index=row,
            row_name="test_cover_capacity_row",
            rhs=20.0,
            generator_coefficients=np.asarray([1.0, 1.0]),
            bus_coefficients=np.zeros(network.bus_ids.size),
            kind="test_upper",
        )
    )

    covers, records, audit = _select_analytic_capacity_cover_cuts(
        master=master,
        separation_reference=np.asarray([0.8, 0.8]),
        base_mva=100.0,
        safety_margin_pu=1e-8,
        minimum_reference_violation=1e-6,
        maximum_selected_cuts=4,
        maximum_signed_cosine_similarity=0.98,
    )

    assert covers
    assert len(covers) == len(records) == audit["selected_cut_count"]
    selected = next(
        record
        for record in records
        if record["parent_cut"]["source_row_name"] == "test_cover_capacity_row"
    )
    assert selected["parent_cut"]["source_row_side"] == "upper"
    assert selected["cover_derivation"]["verification"]["passed"] is True
    assert selected["reference_violation"] > 0.0
    assert audit["cpu_problem_solution_data_used"] is False
    assert audit["original_binary_feasible_set_changed"] is False


def test_v12_cost_dual_seed_never_replaces_secure_phase_one_primal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v12.json")
    case, _table = triangle_case()
    network = build_network(case)
    masks = RegionMasks.root(1)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
        commitment_cuts=(),
    )
    values = np.zeros(master.canonical.num_columns, dtype=np.float64)
    values[master.index.commitment_by_generator[0]] = 1.0
    values[master.index.dispatch_by_generator[0]] = 62.0
    remaining = 37.0
    for column, width in zip(
        master.index.segments_by_generator[0],
        master.costs[0].segment_widths_mw,
        strict=True,
    ):
        if column is not None:
            values[column] = min(remaining, width)
            remaining -= values[column]
    assert master.canonical.max_row_violation(values) == pytest.approx(0.0)

    zero_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    parent_lagrangian = experiment_module.evaluate_lagrangian_bound(
        master,
        zero_dual,
        masks,
        safety_margin_dollars=float(
            config.raw["benchmark"]["certificate_safety_margin_dollars"]
        ),
    )
    dummy_solve = ContinuousSolveResult(
        status="Fixture",
        optimal=False,
        primal_objective=float(np.asarray(master.canonical.objective) @ values),
        dual_objective=None,
        values=values.copy(),
        native_primal=values.copy(),
        native_row_dual=zero_dual.copy(),
        solve_time_seconds=0.0,
        statistics={"error_status": "Success"},
    )
    parent = experiment_module.SolvedRegion(
        region_id="parent",
        masks=masks,
        master=master,
        solve=dummy_solve,
        canonical_row_dual=zero_dual.copy(),
        lagrangian=parent_lagrangian,
        commitment=np.asarray([1.0]),
        security_pairs=(),
        rounds=[],
        final_screen={"new_violated_pairs": 0, "maximum_violation_pu": 0.0},
        gpu_lagrangian={},
    )
    phase = experiment_module.PhaseOneAttemptResult(
        record={},
        source_native_primal=values.copy(),
        source_native_row_dual=None,
        source_values=values.copy(),
    )
    solve_calls: list[dict[str, object]] = []

    def fake_cost_solve(model, **kwargs):
        solve_calls.append(kwargs)
        wandered = np.zeros(model.num_columns, dtype=np.float64)
        return ContinuousSolveResult(
            status="TimeLimit",
            optimal=False,
            primal_objective=0.0,
            dual_objective=0.0,
            values=wandered,
            native_primal=wandered.copy(),
            native_row_dual=np.zeros(model.num_rows, dtype=np.float64),
            solve_time_seconds=0.01,
            statistics={
                "error_status": "Success",
                "warm_start": {
                    "initial_primal_submitted": True,
                    "initial_dual_submitted": True,
                },
            },
        )

    def fake_gpu_polish(inner_master, row_dual, region, **kwargs):
        evaluation = experiment_module.evaluate_lagrangian_bound(
            inner_master,
            row_dual,
            region,
            safety_margin_dollars=0.0,
            commitment_cuts=kwargs["commitment_cuts"],
            commitment_cut_dual=kwargs["initial_commitment_cut_dual"],
        )
        return np.asarray(row_dual), {
            "backend": "fixture",
            "best_raw_lower_bound": evaluation.raw_lower_bound,
            "best_minimizing_commitment": evaluation.minimizing_commitment,
            "best_commitment_cut_dual": np.asarray(
                kwargs["initial_commitment_cut_dual"], dtype=np.float64
            ),
        }

    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_cost_solve)
    monkeypatch.setattr(experiment_module, "optimize_lagrangian_bound_cupy", fake_gpu_polish)
    rounds = [{"phase_one": {"adapter_wall_time_seconds": 0.01}}]
    child = experiment_module._solve_v12_cost_dual_seeded_region(
        region_id="child",
        masks=masks,
        parent=parent,
        master=master,
        secure_phase=phase,
        security_pairs=(),
        phase_rounds=rounds,
        final_screen={"new_violated_pairs": 0, "maximum_violation_pu": 0.0},
        case=case,
        config=config,
        deadline=Deadline(30.0, 0.0, 0.0),
        commitment_cuts=(),
    )

    assert len(solve_calls) == 1
    assert np.array_equal(solve_calls[0]["initial_native_primal"], values)
    assert solve_calls[0]["initial_native_row_dual"] is not None
    assert np.array_equal(child.solve.values, values)
    assert not np.array_equal(child.solve.values, np.zeros_like(values))
    assert child.solve.statistics["cost_pdlp_returned_primal_used"] is False
    assert rounds[0]["cost_dual_seed"]["secure_phase_one_primal_preserved"] is True


@pytest.mark.parametrize(
    ("version", "uses_adam"), (("v24", False), ("v25", True))
)
def test_v24_v25_hard_cardinality_child_preserves_secure_primal_without_cost_lp(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    uses_adam: bool,
) -> None:
    config = load_config(
        ROOT / "configs" / f"activsg2000-gpu-lagrangian-{version}.json"
    )
    case, _table = triangle_case()
    case.gen[1, GEN_STATUS] = 1.0
    network = build_network(case)
    masks = RegionMasks.root(2)
    parent_master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
        commitment_cuts=(),
    )
    cut = build_commitment_cardinality_cut(
        generator_source_rows=parent_master.index.generator_source_rows,
        subset_positions=np.asarray([0, 1], dtype=np.int64),
        subset_id="fixture_both",
        branch_side="at_least",
        integer_threshold=1,
    )
    child_master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
        commitment_cuts=(cut,),
    )
    values = np.zeros(child_master.canonical.num_columns, dtype=np.float64)
    values[child_master.index.commitment_by_generator[0]] = 1.0
    values[child_master.index.dispatch_by_generator[0]] = 62.0
    remaining = 37.0
    for column, width in zip(
        child_master.index.segments_by_generator[0],
        child_master.costs[0].segment_widths_mw,
        strict=True,
    ):
        if column is not None:
            values[column] = min(remaining, width)
            remaining -= values[column]
    assert child_master.canonical.max_row_violation(values) == pytest.approx(0.0)

    parent_dual = np.zeros(parent_master.canonical.num_rows, dtype=np.float64)
    parent_lagrangian = experiment_module.evaluate_lagrangian_bound(
        parent_master,
        parent_dual,
        masks,
        safety_margin_dollars=float(
            config.raw["benchmark"]["certificate_safety_margin_dollars"]
        ),
    )
    dummy_solve = ContinuousSolveResult(
        status="Fixture",
        optimal=False,
        primal_objective=float(np.asarray(parent_master.canonical.objective) @ values),
        dual_objective=None,
        values=values.copy(),
        native_primal=values.copy(),
        native_row_dual=parent_dual.copy(),
        solve_time_seconds=0.0,
        statistics={"error_status": "Success"},
    )
    parent = experiment_module.SolvedRegion(
        region_id="parent",
        masks=masks,
        master=parent_master,
        solve=dummy_solve,
        canonical_row_dual=parent_dual,
        lagrangian=parent_lagrangian,
        commitment=np.asarray([1.0, 0.0]),
        security_pairs=(),
        rounds=[],
        final_screen={"new_violated_pairs": 0, "maximum_violation_pu": 0.0},
        gpu_lagrangian={},
    )
    phase = experiment_module.PhaseOneAttemptResult(
        record={},
        source_native_primal=values.copy(),
        source_native_row_dual=None,
        source_values=values.copy(),
    )
    gpu_calls: list[dict[str, object]] = []
    adam_calls: list[dict[str, object]] = []

    def fake_gpu_evaluation(inner_master, row_dual, region, **kwargs):
        gpu_calls.append(kwargs)
        replay = experiment_module.evaluate_lagrangian_bound(
            inner_master,
            row_dual,
            region,
            safety_margin_dollars=0.0,
            commitment_cuts=kwargs["commitment_cuts"],
            commitment_cut_dual=kwargs["commitment_cut_dual"],
            hard_cardinality_cuts=kwargs["hard_cardinality_cuts"],
        )
        return {
            "backend": "fixture",
            "raw_lower_bound": replay.raw_lower_bound,
            "minimizing_commitment": replay.minimizing_commitment,
        }

    monkeypatch.setattr(
        experiment_module,
        "evaluate_lagrangian_bound_cupy",
        fake_gpu_evaluation,
    )

    def fake_adam(inner_master, row_dual, region, **kwargs):
        adam_calls.append(kwargs)
        replay = experiment_module.evaluate_lagrangian_bound(
            inner_master,
            row_dual,
            region,
            safety_margin_dollars=0.0,
            commitment_cuts=kwargs["commitment_cuts"],
            commitment_cut_dual=kwargs["initial_commitment_cut_dual"],
            hard_cardinality_cuts=kwargs["hard_cardinality_cuts"],
        )
        return np.asarray(row_dual, dtype=np.float64).copy(), {
            "backend": "fixture_adam",
            "best_raw_lower_bound": replay.raw_lower_bound,
            "best_minimizing_commitment": replay.minimizing_commitment,
            "best_commitment_cut_dual": np.asarray(
                kwargs["initial_commitment_cut_dual"], dtype=np.float64
            ),
        }

    monkeypatch.setattr(
        experiment_module,
        "optimize_lagrangian_bound_cupy_adam",
        fake_adam,
    )
    rounds = [{"phase_one": {"adapter_wall_time_seconds": 0.01}}]
    child = experiment_module._solve_v23_hard_cardinality_region(
        region_id="child",
        masks=masks,
        parent=parent,
        master=child_master,
        secure_phase=phase,
        security_pairs=(),
        phase_rounds=rounds,
        final_screen={"new_violated_pairs": 0, "maximum_violation_pu": 0.0},
        case=case,
        config=config,
        commitment_cuts=(cut,),
    )

    assert len(gpu_calls) == 1
    assert len(adam_calls) == int(uses_adam)
    assert gpu_calls[0]["hard_cardinality_cuts"] == (cut,)
    if uses_adam:
        assert adam_calls[0]["iterations"] == 128
        assert adam_calls[0]["hard_cardinality_cuts"] == (cut,)
    assert np.array_equal(child.solve.values, values)
    assert child.solve.statistics["ordinary_cost_lp_solved"] is False
    assert child.lagrangian.hard_cardinality_cut_ids == (cut.cut_id,)
    assert rounds[0]["hard_cardinality_lagrangian"][
        "secure_phase_one_primal_preserved"
    ] is True


def test_v26_proof_only_child_uses_conditioned_proposal_and_exact_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(
        ROOT / "configs" / "activsg2000-gpu-lagrangian-v26.json"
    )
    case, _table = triangle_case()
    case.gen[1, GEN_STATUS] = 1.0
    network = build_network(case)
    masks = RegionMasks.root(2)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
        commitment_cuts=(),
    )
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    parent_lagrangian = experiment_module.evaluate_lagrangian_bound(
        master,
        row_dual,
        masks,
        safety_margin_dollars=float(
            config.raw["benchmark"]["certificate_safety_margin_dollars"]
        ),
    )
    parent = experiment_module.SolvedRegion(
        region_id="parent",
        masks=masks,
        master=master,
        solve=ContinuousSolveResult(
            status="Fixture",
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
        lagrangian=parent_lagrangian,
        commitment=np.asarray([0.5, 0.5]),
        security_pairs=(),
        rounds=[],
        final_screen={"new_violated_pairs": 0, "maximum_violation_pu": 0.0},
        gpu_lagrangian={"wall_time_seconds": 0.0},
    )
    cut = build_commitment_cardinality_cut(
        generator_source_rows=master.index.generator_source_rows + 1,
        subset_positions=np.asarray([0, 1], dtype=np.int64),
        subset_id="proof_only_fixture",
        branch_side="at_least",
        integer_threshold=1,
    )

    def fake_gpu_evaluation(inner_master, inner_dual, region, **kwargs):
        replay = experiment_module.evaluate_lagrangian_bound(
            inner_master,
            inner_dual,
            region,
            safety_margin_dollars=0.0,
            commitment_cuts=kwargs["commitment_cuts"],
            commitment_cut_dual=kwargs["commitment_cut_dual"],
            hard_cardinality_cuts=kwargs["hard_cardinality_cuts"],
        )
        return {
            "backend": "fixture",
            "raw_lower_bound": replay.raw_lower_bound,
            "minimizing_commitment": replay.minimizing_commitment,
        }

    solver_models: list[CanonicalMILP] = []

    def fake_pdlp(search_model, **_kwargs):
        solver_models.append(search_model)
        return ContinuousSolveResult(
            status="Optimal",
            optimal=True,
            primal_objective=0.0,
            dual_objective=0.0,
            values=np.zeros(search_model.num_columns, dtype=np.float64),
            native_primal=np.zeros(search_model.num_columns, dtype=np.float64),
            native_row_dual=np.zeros(search_model.num_rows, dtype=np.float64),
            solve_time_seconds=0.01,
            statistics={"error_status": "Success"},
        )

    monkeypatch.setattr(
        experiment_module,
        "evaluate_lagrangian_bound_cupy",
        fake_gpu_evaluation,
    )
    monkeypatch.setattr(
        experiment_module,
        "solve_cuopt_continuous_pdlp",
        fake_pdlp,
    )
    child = experiment_module._solve_v26_proof_only_hard_cardinality_region(
        region_id="child",
        masks=masks,
        parent=parent,
        case=case,
        config=config,
        deadline=Deadline(30.0, 0.0, 0.0),
        commitment_cuts=(cut,),
    )

    assert len(solver_models) == 1
    assert child.master is parent.master
    assert child.solve.values is None
    assert child.solve.statistics["child_phase_one_solved"] is False
    assert child.solve.statistics["ordinary_cost_lp_solved"] is False
    np.testing.assert_array_equal(child.commitment, parent.commitment)
    assert child.lagrangian.hard_cardinality_cut_ids == (cut.cut_id,)
    assert child.gpu_lagrangian["exact_host_replay_is_bound_authority"] is True
    assert child.gpu_lagrangian["search_lp_solution_used_as_bound"] is False
    record = experiment_module._region_record(
        child,
        compact_lagrangian_certificate=True,
    )
    json.dumps(record)
    nested_audit = record["constraint_generation_rounds"][0][
        "proof_only_hard_cardinality"
    ]["gpu_dual_search"]
    assert isinstance(nested_audit["best_commitment_cut_dual"], list)
    assert isinstance(nested_audit["best_minimizing_commitment"], list)


def test_v32_proof_only_siblings_share_one_block_diagonal_pdlp_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v32.json")
    case, _table = triangle_case()
    case.gen[1, GEN_STATUS] = 1.0
    network = build_network(case)
    masks = RegionMasks.root(2)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
        commitment_cuts=(),
    )
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    parent_lagrangian = experiment_module.evaluate_lagrangian_bound(
        master,
        row_dual,
        masks,
        safety_margin_dollars=float(config.raw["benchmark"]["certificate_safety_margin_dollars"]),
    )
    parent = experiment_module.SolvedRegion(
        region_id="parent",
        masks=masks,
        master=master,
        solve=ContinuousSolveResult(
            status="Fixture",
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
        lagrangian=parent_lagrangian,
        commitment=np.asarray([0.5, 0.5]),
        security_pairs=(),
        rounds=[],
        final_screen={"new_violated_pairs": 0, "maximum_violation_pu": 0.0},
        gpu_lagrangian={"wall_time_seconds": 0.0},
    )
    source_rows = master.index.generator_source_rows + 1
    at_most = build_commitment_cardinality_cut(
        generator_source_rows=source_rows,
        subset_positions=np.asarray([0, 1], dtype=np.int64),
        subset_id="proof_pair_fixture",
        branch_side="at_most",
        integer_threshold=0,
    )
    at_least = build_commitment_cardinality_cut(
        generator_source_rows=source_rows,
        subset_positions=np.asarray([0, 1], dtype=np.int64),
        subset_id="proof_pair_fixture",
        branch_side="at_least",
        integer_threshold=1,
    )

    def fake_gpu_evaluation(inner_master, inner_dual, region, **kwargs):
        replay = experiment_module.evaluate_lagrangian_bound(
            inner_master,
            inner_dual,
            region,
            safety_margin_dollars=0.0,
            commitment_cuts=kwargs["commitment_cuts"],
            commitment_cut_dual=kwargs["commitment_cut_dual"],
            hard_cardinality_cuts=kwargs["hard_cardinality_cuts"],
        )
        return {
            "backend": "fixture",
            "raw_lower_bound": replay.raw_lower_bound,
            "minimizing_commitment": replay.minimizing_commitment,
        }

    solver_calls: list[tuple[CanonicalMILP, dict[str, object]]] = []

    def fake_pdlp(search_model, **kwargs):
        solver_calls.append((search_model, kwargs))
        return ContinuousSolveResult(
            status="TimeLimit",
            optimal=False,
            primal_objective=0.0,
            dual_objective=None,
            values=np.asarray(kwargs["initial_native_primal"], dtype=np.float64),
            native_primal=np.asarray(kwargs["initial_native_primal"], dtype=np.float64),
            native_row_dual=np.zeros(search_model.num_rows, dtype=np.float64),
            solve_time_seconds=0.01,
            statistics={"error_status": "Success"},
        )

    monkeypatch.setattr(
        experiment_module,
        "evaluate_lagrangian_bound_cupy",
        fake_gpu_evaluation,
    )
    monkeypatch.setattr(
        experiment_module,
        "solve_cuopt_continuous_pdlp",
        fake_pdlp,
    )
    children, audit = experiment_module._solve_v32_proof_only_hard_cardinality_pair(
        child_specs=(
            ("child_0", masks, (at_most,)),
            ("child_1", masks, (at_least,)),
        ),
        parent=parent,
        case=case,
        config=config,
        deadline=Deadline(30.0, 0.0, 0.0),
    )

    assert len(solver_calls) == 1
    combined, solve_kwargs = solver_calls[0]
    assert audit["model"]["block_count"] == 2
    assert audit["model"]["mathematical_cross_block_coupling_added"] is False
    assert combined.num_columns == sum(block["columns"] for block in audit["model"]["blocks"])
    assert combined.max_row_violation(solve_kwargs["initial_native_primal"]) <= 1e-12
    assert audit["model"]["finite_column_bound_maximum_absolute"] <= 1.0
    assert audit["model"]["matrix_nonzero_minimum_absolute"] >= 1e-4
    assert audit["model"]["objective_nonzero_minimum_absolute"] >= 1e-4
    assert set(children) == {"child_0", "child_1"}
    assert all(
        child.gpu_lagrangian["exact_host_replay_is_bound_authority"] is True
        for child in children.values()
    )
    assert all(
        child.solve.statistics["shared_block_diagonal_solver"] is True
        for child in children.values()
    )
    assert all(
        child.lagrangian.conservative_lower_bound >= parent.lagrangian.conservative_lower_bound
        for child in children.values()
    )
    json.dumps(
        {
            child_id: experiment_module._region_record(
                child,
                compact_lagrangian_certificate=True,
            )
            for child_id, child in children.items()
        }
    )



def test_v22_strengthened_root_uses_only_exact_replayed_cost_dual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v22.json")
    case, _table = triangle_case()
    network = build_network(case)
    masks = RegionMasks.root(1)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
        commitment_cuts=(),
    )
    zero_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    certificate = experiment_module.evaluate_lagrangian_bound(
        master,
        zero_dual,
        masks,
        safety_margin_dollars=float(
            config.raw["benchmark"]["certificate_safety_margin_dollars"]
        ),
    )
    dummy_solve = ContinuousSolveResult(
        status="Fixture",
        optimal=False,
        primal_objective=1000.0,
        dual_objective=None,
        values=None,
        native_primal=None,
        native_row_dual=None,
        solve_time_seconds=0.0,
        statistics={"error_status": "Success"},
    )
    root = experiment_module.SolvedRegion(
        region_id="r",
        masks=masks,
        master=master,
        solve=dummy_solve,
        canonical_row_dual=zero_dual.copy(),
        lagrangian=certificate,
        commitment=np.asarray([0.5]),
        security_pairs=(),
        rounds=[],
        final_screen={"new_violated_pairs": 0, "maximum_violation_pu": 0.0},
        gpu_lagrangian={},
    )
    calls: list[dict[str, object]] = []

    def fake_cost_solve(model, **kwargs):
        calls.append(kwargs)
        wandered = np.full(model.num_columns, -999.0, dtype=np.float64)
        return ContinuousSolveResult(
            status="TimeLimit",
            optimal=False,
            primal_objective=-1e12,
            dual_objective=0.0,
            values=wandered,
            native_primal=wandered.copy(),
            native_row_dual=np.zeros(model.num_rows, dtype=np.float64),
            solve_time_seconds=0.01,
            statistics={
                "error_status": "Success",
                "warm_start": {
                    "initial_primal_submitted": False,
                    "initial_dual_submitted": True,
                },
            },
        )

    def fake_gpu_polish(inner_master, row_dual, region, **kwargs):
        evaluation = experiment_module.evaluate_lagrangian_bound(
            inner_master,
            row_dual,
            region,
            safety_margin_dollars=0.0,
            commitment_cuts=kwargs["commitment_cuts"],
            commitment_cut_dual=kwargs["initial_commitment_cut_dual"],
        )
        return np.asarray(row_dual), {
            "backend": "fixture",
            "best_raw_lower_bound": evaluation.raw_lower_bound,
            "best_minimizing_commitment": evaluation.minimizing_commitment,
            "best_commitment_cut_dual": np.asarray(
                kwargs["initial_commitment_cut_dual"], dtype=np.float64
            ),
        }

    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_cost_solve)
    monkeypatch.setattr(experiment_module, "optimize_lagrangian_bound_cupy", fake_gpu_polish)
    refreshed = _refresh_region_with_bounded_cost_dual_search(
        region=root,
        case=case,
        config=config,
        deadline=Deadline(200.0, 0.0, 0.0),
        dual_target_objective=1000.0,
    )

    assert len(calls) == 1
    assert calls[0].get("initial_native_primal") is None
    assert calls[0]["initial_native_row_dual"] is not None
    assert np.array_equal(refreshed.commitment, root.commitment)
    assert refreshed.solve is root.solve
    assert refreshed.gpu_lagrangian["cost_pdlp_returned_primal_used"] is False
    assert refreshed.gpu_lagrangian["cost_pdlp_native_objective_used_as_bound"] is False
    assert refreshed.lagrangian.conservative_lower_bound == pytest.approx(
        certificate.conservative_lower_bound
    )


def test_phase_one_round_limit_never_launches_an_unscreened_terminal_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v12.json")
    config.raw["runtime"]["maximum_child_phase_one_rounds"] = 2
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    masks = RegionMasks.root(1)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
        commitment_cuts=(),
    )
    values = np.zeros(master.canonical.num_columns, dtype=np.float64)
    values[master.index.commitment_by_generator[0]] = 1.0
    values[master.index.dispatch_by_generator[0]] = 62.0
    remaining = 37.0
    for column, width in zip(
        master.index.segments_by_generator[0],
        master.costs[0].segment_widths_mw,
        strict=True,
    ):
        if column is not None:
            values[column] = min(remaining, width)
            remaining -= values[column]

    def phase_result(kind: str) -> experiment_module.PhaseOneAttemptResult:
        return experiment_module.PhaseOneAttemptResult(
            record={
                "region_id": "child",
                "phase_one_attempt_kind": kind,
                "phase_one_model": {"columns": master.canonical.num_columns + 1},
                "solver_budget_seconds": 1.0,
                "adapter_wall_time_seconds": 0.01,
                "preparation_wall_time_seconds": 0.0,
                "total_attempt_wall_time_seconds": 0.01,
                "solve": {"status": "Optimal"},
                "prune_certified": False,
                "source_feasible_warm_start": {"eligible": True},
                "security_pairs": [],
            },
            source_native_primal=values.copy(),
            source_native_row_dual=None,
            source_values=values.copy(),
        )

    pairs = []
    for outage_column, monitored_active_index in ((0, 1), (1, 0)):
        outage = catalog.valid[outage_column]
        pairs.append(
            SecurityPair(
                outage.contingency_label,
                int(network.active_branch_source_rows[monitored_active_index]) + 1,
                "upper",
                outage_column,
                monitored_active_index,
                outage.active_branch_index,
                float(catalog.lodf[monitored_active_index, outage_column]),
            )
        )

    class TwoViolationScreens:
        def __init__(self) -> None:
            self.calls = 0

        def screen(self, *_args, **_kwargs):
            pair = pairs[self.calls]
            self.calls += 1
            return ScreenResult((pair,), 0.1, pair.pair_id, 1)

    phase_solve_calls = 0

    def fake_phase_solve(**kwargs):
        nonlocal phase_solve_calls
        phase_solve_calls += 1
        return phase_result(str(kwargs["attempt_kind"]))

    monkeypatch.setattr(experiment_module, "_run_phase_one_attempt", fake_phase_solve)
    screener = TwoViolationScreens()
    with pytest.raises(
        RegionAttemptRejected,
        match="exhausted Phase-I security-generation rounds",
    ):
        experiment_module._solve_phase_one_lagrangian_region(
            region_id="child",
            masks=masks,
            parent=None,  # type: ignore[arg-type]
            master=master,
            initial_pairs=(),
            initial_precheck=phase_result("pre_cost_lp"),
            case=case,
            network=network,
            config=config,
            deadline=Deadline(30.0, 0.0, 0.0),
            screener=screener,  # type: ignore[arg-type]
            checkpoint=lambda: None,
            commitment_cuts=(),
        )

    assert screener.calls == 2
    assert phase_solve_calls == 1


def test_phase_one_native_dual_maps_lower_upper_and_equality_rows() -> None:
    source = CanonicalMILP()
    x = source.add_variable("x", lower=-10.0, upper=10.0)
    source.add_row("equal", {x: 1.0}, lower=0.0, upper=0.0)
    source.add_row("ranged", {x: 1.0}, lower=-1.0, upper=1.0)
    source.add_row("lower_only", {x: 1.0}, lower=-2.0)
    source.add_row("upper_only", {x: 1.0}, upper=2.0)
    phase = experiment_module.build_phase_one_model(
        source, base_mva=100.0, maximum_violation_pu=1e6
    )
    phase_dual = np.asarray([-2.0, -3.0, -4.0, -5.0, -6.0, -7.0])

    mapped, audit = _map_phase_one_dual_to_source_native(
        phase,
        source,
        phase_dual,
        scaling_mode="power_system_equilibrated_v2",
        base_mva=100.0,
    )

    np.testing.assert_allclose(mapped, np.asarray([-0.01, 0.04, -0.05, 0.12, -0.14]))
    assert audit["mapped_phase_side_count"] == 6
    assert audit["warm_start_only"] is True
    assert audit["certificate_claimed"] is False


def test_v6_exact_pmin_pmax_capacity_gate_precedes_phase_one() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v6.json")
    case, _ = triangle_case()
    network = build_network(case)
    root_masks = RegionMasks.root(1)
    root = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=root_masks,
        initial_pairs=(),
    )
    root_gate = _region_pmin_pmax_capacity_gate(
        case=case,
        master=root,
        masks=root_masks,
        tolerance_pu=1e-6,
    )
    assert root_gate["passes"] is True
    assert root_gate["demand_mw"] == pytest.approx(62.0)
    assert root_gate["minimum_dispatch_mw"] == pytest.approx(0.0)
    assert root_gate["maximum_dispatch_mw"] == pytest.approx(100.0)

    off_masks = RegionMasks(np.asarray([True]), np.asarray([False]))
    off = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=off_masks,
        initial_pairs=(),
    )
    off_gate = _region_pmin_pmax_capacity_gate(
        case=case,
        master=off,
        masks=off_masks,
        tolerance_pu=1e-6,
    )
    assert off_gate["passes"] is False
    assert off_gate["capacity_shortfall_mw"] == pytest.approx(62.0)
    assert off_gate["exact_source_pmin_changed"] is False


def test_prepared_region_validation_maps_nonzero_source_row_to_commitment_column() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v6.json")
    case, _ = triangle_case()
    case.gen[0, GEN_STATUS] = 0.0
    case.gen[1, GEN_STATUS] = 1.0
    network = build_network(case)
    masks = RegionMasks.root(1)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
    )

    assert master.index.generator_source_rows.tolist() == [1]
    assert master.index.commitment_by_generator[1] == 0
    _validate_prepared_region_master(master, masks, ())


def test_v6_zero_phase_one_primal_warm_starts_cost_lp_without_dual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v6.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    masks = RegionMasks.root(1)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
    )
    source_values = np.zeros(master.canonical.num_columns)
    source_values[master.index.commitment_by_generator[0]] = 1.0
    source_values[master.index.dispatch_by_generator[0]] = 62.0
    remaining = 37.0
    for column, width in zip(
        master.index.segments_by_generator[0],
        master.costs[0].segment_widths_mw,
        strict=True,
    ):
        if column is not None:
            source_values[column] = min(remaining, width)
            remaining -= source_values[column]

    solve_calls: list[dict[str, object]] = []

    def fake_solve(model, **kwargs):
        solve_calls.append(kwargs)
        is_phase_one = model.variable_names[-1] == "phase1_violation_pu"
        values = (
            np.concatenate((source_values, np.asarray([0.0])))
            if is_phase_one
            else source_values.copy()
        )
        column_scale, _ = native_scaling_vectors(
            model,
            mode="power_system_per_unit_v1",
            base_mva=case.base_mva,
        )
        return ContinuousSolveResult(
            status="Optimal",
            optimal=True,
            primal_objective=float(np.asarray(model.objective) @ values),
            dual_objective=0.0,
            values=values,
            native_primal=values / column_scale,
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {"passed": True, "primal_feasible": True},
            },
        )

    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    rejected = RegionAttemptRejected(
        "registered precheck",
        reason="phase_one_first_precheck",
        master=master,
        security_pairs=(),
        rounds=[],
    )
    phase = _run_phase_one_attempt(
        region_id="r0",
        masks=masks,
        rejected=rejected,
        case=case,
        config=config,
        deadline=Deadline(10.0, 0.0, 0.0),
        attempt_kind="pre_cost_lp",
        time_limit_seconds=2.0,
    )
    assert phase.record["prune_certified"] is False
    assert phase.record["source_feasible_warm_start"]["eligible"] is True
    assert phase.record["source_feasible_warm_start"]["dual_transferred"] is False
    assert phase.source_native_primal is not None

    monkeypatch.setattr(
        experiment_module,
        "canonical_row_duals",
        lambda _master, native_row_dual, **_kwargs: np.asarray(native_row_dual),
    )
    monkeypatch.setattr(
        experiment_module,
        "optimize_lagrangian_bound_cupy",
        lambda _master, row_dual, _region, **_kwargs: (
            np.asarray(row_dual),
            {
                "backend": "fixture",
                "best_raw_lower_bound": 0.0,
                "best_minimizing_commitment": np.asarray([0], dtype=np.int8),
            },
        ),
    )
    solved = _solve_region(
        region_id="r0",
        masks=masks,
        case=case,
        network=network,
        catalog=catalog,
        config=config,
        deadline=Deadline(10.0, 0.0, 0.0),
        initial_pairs=(),
        screener=ContingencyScreener(network, catalog, backend="numpy"),
        checkpoint=lambda: None,
        prepared_master=master,
        initial_native_primal=phase.source_native_primal,
        initial_warm_start_origin="phase_one_zero_violation_primal_v1",
    )
    assert len(solve_calls) == 2
    assert np.array_equal(solve_calls[1]["initial_native_primal"], phase.source_native_primal)
    assert solve_calls[1]["initial_native_row_dual"] is None
    assert solved.rounds[0]["initial_warm_start_origin"] == ("phase_one_zero_violation_primal_v1")


def test_lagrangian_config_rejects_case_mismatched_identity() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v1.json")
    config.raw["benchmark"]["id"] = "activsg2000-gpu-lagrangian-v1"
    with pytest.raises(ScopfError, match="case changed"):
        validate_lagrangian_experiment_config(config)


def test_tiny_region_flow_reaches_exhaustive_screen_without_integer_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v1.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    template = build_reduced_master(case, network)

    def fake_solve(model, **_kwargs):
        values = np.zeros(model.num_columns)
        values[template.index.commitment_by_generator[0]] = 1.0
        values[template.index.dispatch_by_generator[0]] = 62.0
        remaining = 37.0
        for column, width in zip(
            template.index.segments_by_generator[0],
            template.costs[0].segment_widths_mw,
            strict=True,
        ):
            if column is not None:
                values[column] = min(remaining, width)
                remaining -= values[column]
        return ContinuousSolveResult(
            status="Optimal",
            optimal=True,
            primal_objective=float(np.asarray(model.objective) @ values),
            dual_objective=0.0,
            values=values,
            native_primal=values.copy(),
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {"passed": True, "primal_feasible": True},
            },
        )

    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    monkeypatch.setattr(
        experiment_module,
        "canonical_row_duals",
        lambda master, native_row_dual, **_kwargs: np.asarray(native_row_dual),
    )
    monkeypatch.setattr(
        experiment_module,
        "optimize_lagrangian_bound_cupy",
        lambda master, row_dual, region, **_kwargs: (
            np.asarray(row_dual),
            {
                "backend": "fixture",
                "best_raw_lower_bound": 0.0,
                "best_minimizing_commitment": np.asarray([0], dtype=np.int8),
            },
        ),
    )
    solved = _solve_region(
        region_id="r",
        masks=RegionMasks.root(1),
        case=case,
        network=network,
        catalog=catalog,
        config=config,
        deadline=Deadline(10.0, 0.0, 0.0),
        initial_pairs=(),
        screener=ContingencyScreener(network, catalog, backend="numpy"),
        checkpoint=lambda: None,
    )
    assert solved.solve.statistics["native_integer_columns"] == 0
    assert solved.final_screen["new_violated_pairs"] == 0
    assert solved.lagrangian.conservative_lower_bound == -0.01


def test_v5_root_refines_optimal_solution_that_misses_canonical_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v5.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    template = build_reduced_master(case, network)
    solve_calls: list[dict[str, object]] = []

    def feasible_values(model) -> np.ndarray:
        values = np.zeros(model.num_columns)
        values[template.index.commitment_by_generator[0]] = 1.0
        values[template.index.dispatch_by_generator[0]] = 62.0
        remaining = 37.0
        for column, width in zip(
            template.index.segments_by_generator[0],
            template.costs[0].segment_widths_mw,
            strict=True,
        ):
            if column is not None:
                values[column] = min(remaining, width)
                remaining -= values[column]
        return values

    def fake_solve(model, **kwargs):
        solve_calls.append(kwargs)
        values = np.zeros(model.num_columns) if len(solve_calls) == 1 else feasible_values(model)
        return ContinuousSolveResult(
            status="Optimal",
            optimal=True,
            primal_objective=float(np.asarray(model.objective) @ values),
            dual_objective=0.0,
            values=values,
            native_primal=values.copy(),
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {"passed": True, "primal_feasible": True},
            },
        )

    class CountingScreener:
        def __init__(self) -> None:
            self.calls = 0
            self.delegate = ContingencyScreener(network, catalog, backend="numpy")

        def screen(self, *args, **kwargs):
            self.calls += 1
            return self.delegate.screen(*args, **kwargs)

    screener = CountingScreener()
    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    monkeypatch.setattr(
        experiment_module,
        "canonical_row_duals",
        lambda master, native_row_dual, **_kwargs: np.asarray(native_row_dual),
    )
    monkeypatch.setattr(
        experiment_module,
        "optimize_lagrangian_bound_cupy",
        lambda master, row_dual, region, **_kwargs: (
            np.asarray(row_dual),
            {
                "backend": "fixture",
                "best_raw_lower_bound": 0.0,
                "best_minimizing_commitment": np.asarray([0], dtype=np.int8),
            },
        ),
    )

    solved = _solve_region(
        region_id="r",
        masks=RegionMasks.root(1),
        case=case,
        network=network,
        catalog=catalog,
        config=config,
        deadline=Deadline(10.0, 0.0, 0.0),
        initial_pairs=(),
        screener=screener,  # type: ignore[arg-type]
        checkpoint=lambda: None,
    )

    assert len(solve_calls) == 2
    assert solve_calls[0]["optimality_tolerance"] == 1e-8
    assert solve_calls[1]["optimality_tolerance"] == 1e-10
    assert solve_calls[1]["initial_native_primal"] is not None
    assert solve_calls[1]["initial_native_row_dual"] is not None
    assert screener.calls == 1
    assert len(solved.rounds) == 1
    refinement = solved.rounds[0]["canonical_residual_refinement"]
    assert refinement["accepted"] is True
    assert len(refinement["attempts"]) == 1
    assert (
        solved.rounds[0]["pre_refinement_primal_acceptance"]["canonical_model_residual_passed"]
        is False
    )
    assert solved.rounds[0]["primal_acceptance"]["canonical_model_residual_passed"] is True
    assert (
        solved.rounds[0]["pre_refinement_primal_acceptance"]["worst_canonical_row"]["row_name"]
        == "lag_balance"
    )


def test_region_continues_without_screening_a_primal_infeasible_pdlp_iterate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v2.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    template = build_reduced_master(case, network)
    solve_calls: list[dict[str, object]] = []

    def feasible_values(model) -> np.ndarray:
        values = np.zeros(model.num_columns)
        values[template.index.commitment_by_generator[0]] = 1.0
        values[template.index.dispatch_by_generator[0]] = 62.0
        remaining = 37.0
        for column, width in zip(
            template.index.segments_by_generator[0],
            template.costs[0].segment_widths_mw,
            strict=True,
        ):
            if column is not None:
                values[column] = min(remaining, width)
                remaining -= values[column]
        return values

    def fake_solve(model, **kwargs):
        solve_calls.append(kwargs)
        first = len(solve_calls) == 1
        values = np.zeros(model.num_columns) if first else feasible_values(model)
        return ContinuousSolveResult(
            status="TimeLimit" if first else "Optimal",
            optimal=not first,
            primal_objective=float(np.asarray(model.objective) @ values),
            dual_objective=0.0,
            values=values,
            native_primal=values.copy(),
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {
                    "passed": not first,
                    "primal_feasible": not first,
                },
            },
        )

    class CountingScreener:
        def __init__(self) -> None:
            self.calls = 0
            self.delegate = ContingencyScreener(network, catalog, backend="numpy")

        def screen(self, *args, **kwargs):
            self.calls += 1
            return self.delegate.screen(*args, **kwargs)

    screener = CountingScreener()
    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    monkeypatch.setattr(
        experiment_module,
        "canonical_row_duals",
        lambda master, native_row_dual, **_kwargs: np.asarray(native_row_dual),
    )
    monkeypatch.setattr(
        experiment_module,
        "optimize_lagrangian_bound_cupy",
        lambda master, row_dual, region, **_kwargs: (
            np.asarray(row_dual),
            {
                "backend": "fixture",
                "best_raw_lower_bound": 0.0,
                "best_minimizing_commitment": np.asarray([0], dtype=np.int8),
            },
        ),
    )
    solved = _solve_region(
        region_id="r",
        masks=RegionMasks.root(1),
        case=case,
        network=network,
        catalog=catalog,
        config=config,
        deadline=Deadline(10.0, 0.0, 0.0),
        initial_pairs=(),
        screener=screener,  # type: ignore[arg-type]
        checkpoint=lambda: None,
    )
    assert len(solve_calls) == 2
    assert screener.calls == 1
    assert solved.rounds[0]["screen"]["reason"] == "pdlp_primal_infeasible"
    assert solve_calls[1]["initial_native_primal"] is not None
    assert solve_calls[1]["initial_native_row_dual"] is not None


def test_candidate_rejects_divergent_dual_without_screening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v3.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    solve_calls = 0

    def fake_solve(model, **_kwargs):
        nonlocal solve_calls
        solve_calls += 1
        values = np.zeros(model.num_columns)
        return ContinuousSolveResult(
            status="TimeLimit",
            optimal=False,
            primal_objective=1.0,
            dual_objective=1e9,
            values=values,
            native_primal=values.copy(),
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {"passed": False, "primal_feasible": False},
            },
        )

    class RejectIfScreened:
        def screen(self, *_args, **_kwargs):
            raise AssertionError("primal-infeasible candidate must not be screened")

    progress: list[dict[str, object]] = []
    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    with pytest.raises(PrimalCandidateRejected, match="dual_objective_divergence"):
        _solve_region(
            region_id="p1",
            masks=RegionMasks.root(1),
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(30.0, 0.0, 0.0),
            initial_pairs=(),
            screener=RejectIfScreened(),  # type: ignore[arg-type]
            checkpoint=lambda: None,
            progress=progress.append,
            candidate_policy=PrimalCandidatePolicy.from_config(config),
        )
    assert solve_calls == 1
    final_round = progress[-1]["constraint_generation_rounds"][-1]  # type: ignore[index]
    assert final_round["candidate_gate"]["reason"] == "dual_objective_divergence"


def test_candidate_rejects_two_round_residual_stagnation_and_uses_warm_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v3.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    solve_calls: list[dict[str, object]] = []

    def fake_solve(model, **kwargs):
        solve_calls.append(kwargs)
        values = np.zeros(model.num_columns)
        return ContinuousSolveResult(
            status="TimeLimit",
            optimal=False,
            primal_objective=1.0,
            dual_objective=0.0,
            values=values,
            native_primal=values.copy(),
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {"passed": False, "primal_feasible": False},
            },
        )

    class RejectIfScreened:
        def screen(self, *_args, **_kwargs):
            raise AssertionError("primal-infeasible candidate must not be screened")

    progress: list[dict[str, object]] = []
    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    with pytest.raises(PrimalCandidateRejected, match="primal_residual_stagnation"):
        _solve_region(
            region_id="p1",
            masks=RegionMasks.root(1),
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(30.0, 0.0, 0.0),
            initial_pairs=(),
            screener=RejectIfScreened(),  # type: ignore[arg-type]
            checkpoint=lambda: None,
            progress=progress.append,
            candidate_policy=PrimalCandidatePolicy.from_config(config),
        )
    assert len(solve_calls) == 2
    assert solve_calls[1]["initial_native_primal"] is not None
    assert solve_calls[1]["initial_native_row_dual"] is not None
    final_round = progress[-1]["constraint_generation_rounds"][-1]  # type: ignore[index]
    assert final_round["candidate_gate"]["reason"] == "primal_residual_stagnation"


def test_candidate_does_not_relabel_adapter_error_as_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v3.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)

    def fake_solve(_model, **_kwargs):
        return ContinuousSolveResult(
            status="Error",
            optimal=False,
            primal_objective=None,
            dual_objective=None,
            values=None,
            native_primal=None,
            native_row_dual=None,
            solve_time_seconds=0.001,
            statistics={
                "error_status": "InternalError",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {"passed": False, "primal_feasible": False},
            },
        )

    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    with pytest.raises(ScopfError, match="did not return usable vectors") as caught:
        _solve_region(
            region_id="p1",
            masks=RegionMasks.root(1),
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(30.0, 0.0, 0.0),
            initial_pairs=(),
            screener=ContingencyScreener(network, catalog, backend="numpy"),
            checkpoint=lambda: None,
            candidate_policy=PrimalCandidatePolicy.from_config(config),
        )
    assert type(caught.value) is ScopfError


def test_v4_region_uses_warm_attempt_then_exactly_one_cold_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v4.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    template = build_reduced_master(case, network)
    solve_calls: list[dict[str, object]] = []

    def feasible_values(model) -> np.ndarray:
        values = np.zeros(model.num_columns)
        values[template.index.commitment_by_generator[0]] = 1.0
        values[template.index.dispatch_by_generator[0]] = 62.0
        remaining = 37.0
        for column, width in zip(
            template.index.segments_by_generator[0],
            template.costs[0].segment_widths_mw,
            strict=True,
        ):
            if column is not None:
                values[column] = min(remaining, width)
                remaining -= values[column]
        return values

    def fake_solve(model, **kwargs):
        solve_calls.append(kwargs)
        if len(solve_calls) == 1:
            values = feasible_values(model)
            primal_feasible = True
            dual_objective = 0.0
        else:
            values = np.zeros(model.num_columns)
            primal_feasible = False
            dual_objective = 1e9
        return ContinuousSolveResult(
            status="TimeLimit",
            optimal=False,
            primal_objective=1.0,
            dual_objective=dual_objective,
            values=values,
            native_primal=values.copy(),
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {
                    "passed": primal_feasible,
                    "primal_feasible": primal_feasible,
                },
            },
        )

    outage = catalog.valid[0]
    pair = SecurityPair(
        outage.contingency_label,
        int(network.active_branch_source_rows[1]) + 1,
        "upper",
        0,
        1,
        outage.active_branch_index,
        float(catalog.lodf[1, 0]),
    )

    class OneViolation:
        calls = 0

        def screen(self, *_args, **_kwargs):
            self.calls += 1
            return ScreenResult((pair,), 0.1, pair.pair_id, 1)

    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    with pytest.raises(RegionAttemptRejected, match="cold_restart_failed"):
        _solve_region(
            region_id="r0",
            masks=RegionMasks.root(1),
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(30.0, 0.0, 0.0),
            initial_pairs=(),
            screener=OneViolation(),  # type: ignore[arg-type]
            checkpoint=lambda: None,
            candidate_policy=PrimalCandidatePolicy.from_config(config, scope="disjunctive_region"),
        )
    assert len(solve_calls) == 3
    assert solve_calls[1]["initial_native_primal"] is not None
    assert solve_calls[1]["initial_native_row_dual"] is not None
    assert solve_calls[2]["initial_native_primal"] is None
    assert solve_calls[2]["initial_native_row_dual"] is None


def test_cleanup_replay_preserves_proof_but_allows_architecture_dust_counts() -> None:
    recorded = {
        "policy": "cleanup-v1",
        "zero_tolerance": 1e-14,
        "physical_injection_operator_changed": False,
        "solver_rows_are_relaxations_of_original_rows": True,
        "potential_flow_operator_dust": {
            "zero_tolerance": 1e-14,
            "dropped_coefficient_count": 100,
            "maximum_absolute_dropped_coefficient": 8e-15,
        },
        "dropped_generator_coefficient_count": 50,
        "maximum_absolute_dropped_generator_coefficient": 7e-15,
        "total_rhs_outward_relaxation": 2e-10,
        "maximum_row_rhs_outward_relaxation": 3e-12,
        "exact_duplicate_security_row_count": 1,
    }
    rebuilt = {
        **recorded,
        "potential_flow_operator_dust": {
            **recorded["potential_flow_operator_dust"],
            "dropped_coefficient_count": 120,
            "maximum_absolute_dropped_coefficient": 9e-15,
        },
        "dropped_generator_coefficient_count": 60,
        "total_rhs_outward_relaxation": 2.1e-10,
    }

    comparison = _replay_cleanup_audit_comparison(recorded, rebuilt)
    assert comparison["maximum_integer_difference"] == 20
    assert comparison["maximum_fp64_difference"] == pytest.approx(1e-11)

    invalid = {**rebuilt, "solver_rows_are_relaxations_of_original_rows": False}
    with pytest.raises(ScopfError, match="boolean mismatch"):
        _replay_cleanup_audit_comparison(recorded, invalid)


def test_relative_gap_tracks_current_frontier_bound() -> None:
    assert _relative_gap(objective=79_410.65143217964, lower_bound=77_996.45965698606) == (
        pytest.approx(0.01780859053147761)
    )

    with pytest.raises(ScopfError, match="lower bound exceeds"):
        _relative_gap(objective=100.0, lower_bound=101.0)
