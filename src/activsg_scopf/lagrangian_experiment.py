"""One-shot ACTIVSg GPU Lagrangian/disjunctive experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .cardinality import (
    CardinalitySplit,
    choose_cardinality_split,
    commitment_branch_subsets,
    exact_type_group_rounding,
)
from .commitment_cuts import (
    CommitmentCardinalityCut,
    CommitmentFeasibilityCut,
    CommitmentUpperCut,
    add_commitment_upper_cuts,
    commitment_upper_cut_from_record,
    derive_commitment_feasibility_cut,
    generate_commitment_cut_repairs,
)
from .config import RunConfig
from .costs import pwl_approximation_report
from .deadline import Deadline, PeakMemorySampler
from .environment import environment_manifest, validate_platform
from .errors import (
    DeadlineExceeded,
    PrimalCandidateRejected,
    ProvenanceError,
    ScopfError,
)
from .fixed_commitment import (
    FixedCommitmentProjection,
    FixedCommitmentProjectionInfeasible,
    build_fixed_commitment_projection,
    repair_along_feasible_segment,
)
from .lagrangian import (
    LagrangianEvaluation,
    RegionMasks,
    bus_prices_from_coupling_duals,
    canonical_row_duals,
    choose_split_generator,
    evaluate_lagrangian_bound,
    optimize_lagrangian_bound_cupy,
    replay_lagrangian_certificate,
    verify_cardinality_disjunctive_cover,
)
from .matpower import (
    GEN_STATUS,
    PMAX,
    PMIN,
    read_contingency_table,
    read_matpower_case,
)
from .model import build_master
from .network import (
    ContingencyCatalog,
    NetworkData,
    build_contingency_catalog,
    build_network,
    contingency_catalog_report,
)
from .official import frozen_identity
from .paths import guard_output_path, guard_runtime_environment
from .phase_one import (
    build_phase_one_model,
    phase_one_certificate,
    phase_one_semantic_row_key,
    replay_phase_one_certificate,
)
from .provenance import build_source_manifest, write_json_atomic
from .reduced import (
    ReducedMaster,
    add_reduced_security_pairs,
    build_reduced_master,
    commitment_vector,
    fix_commitments,
    reconstruct_full_values,
    reduced_dispatch,
    security_pair_from_record,
    security_pair_record,
)
from .screening import ContingencyScreener, SecurityPair, add_security_pairs
from .solution import serialize_solution
from .solvers.cuopt import (
    FULL_MIP_START,
    native_scaling_vectors,
    solve_cuopt,
    validate_full_mip_start_feasibility,
)
from .solvers.cuopt_lp import (
    ContinuousSolveResult,
    derive_rate_a_angle_bounds,
    solve_cuopt_continuous_pdlp,
)
from .verify import verify_serialized_solution

EXPERIMENT_ID = "activsg500-gpu-lagrangian-v1"
EXPERIMENT_TAG = "experiment-500-gpu-lagrangian-v1"
ACTIVSG2000_EXPERIMENT_ID = "activsg2000-gpu-lagrangian-v1"
ACTIVSG2000_V2_EXPERIMENT_ID = "activsg2000-gpu-lagrangian-v2"
ACTIVSG2000_V3_EXPERIMENT_ID = "activsg2000-gpu-lagrangian-v3"
ACTIVSG2000_V4_EXPERIMENT_ID = "activsg2000-gpu-lagrangian-v4"
ACTIVSG2000_V5_EXPERIMENT_ID = "activsg2000-gpu-lagrangian-v5"
ACTIVSG2000_V6_EXPERIMENT_ID = "activsg2000-gpu-lagrangian-v6"
ACTIVSG2000_V7_EXPERIMENT_ID = "activsg2000-gpu-lagrangian-v7"
ACTIVSG2000_V8_EXPERIMENT_ID = "activsg2000-gpu-lagrangian-v8"
ACTIVSG2000_V9_EXPERIMENT_ID = "activsg2000-gpu-lagrangian-v9"
ACTIVSG2000_V10_EXPERIMENT_ID = "activsg2000-gpu-lagrangian-v10"
ACTIVSG2000_V11_EXPERIMENT_ID = "activsg2000-gpu-lagrangian-v11"
ACTIVSG2000_V4_PLUS_EXPERIMENT_IDS = frozenset(
    {
        ACTIVSG2000_V4_EXPERIMENT_ID,
        ACTIVSG2000_V5_EXPERIMENT_ID,
        ACTIVSG2000_V6_EXPERIMENT_ID,
        ACTIVSG2000_V7_EXPERIMENT_ID,
        ACTIVSG2000_V8_EXPERIMENT_ID,
        ACTIVSG2000_V9_EXPERIMENT_ID,
        ACTIVSG2000_V10_EXPERIMENT_ID,
        ACTIVSG2000_V11_EXPERIMENT_ID,
    }
)
ACTIVSG2000_V8_PLUS_EXPERIMENT_IDS = frozenset(
    {
        ACTIVSG2000_V8_EXPERIMENT_ID,
        ACTIVSG2000_V9_EXPERIMENT_ID,
        ACTIVSG2000_V10_EXPERIMENT_ID,
        ACTIVSG2000_V11_EXPERIMENT_ID,
    }
)
REGISTERED_EXPERIMENTS = {
    EXPERIMENT_ID: {
        "tag": EXPERIMENT_TAG,
        "policy": "gpu_pdlp_primal_plus_replayable_lagrangian_cover_v1",
    },
    "activsg500-gpu-lagrangian-v2": {
        "tag": "experiment-500-gpu-lagrangian-v2",
        "policy": "gpu_pdlp_primal_plus_replayable_lagrangian_cover_v2",
    },
    "activsg500-gpu-lagrangian-v3": {
        "tag": "experiment-500-gpu-lagrangian-v3",
        "policy": "gpu_pdlp_bounded_candidate_queue_plus_lagrangian_cover_v3",
    },
    "activsg500-gpu-lagrangian-v4": {
        "tag": "experiment-500-gpu-lagrangian-v4",
        "policy": "gpu_pdlp_phase_one_pruning_plus_lagrangian_cover_v4",
    },
    "activsg500-gpu-lagrangian-v5": {
        "tag": "experiment-500-gpu-lagrangian-v5",
        "policy": "gpu_pdlp_phase_one_pruning_plus_lagrangian_cover_v5",
    },
    "activsg500-gpu-lagrangian-v6": {
        "tag": "experiment-500-gpu-lagrangian-v6",
        "policy": "gpu_pdlp_phase_one_first_plus_lagrangian_cover_v6",
    },
    "activsg500-gpu-lagrangian-v7": {
        "tag": "experiment-500-gpu-lagrangian-v7",
        "policy": "gpu_pdlp_phase_one_first_plus_lagrangian_cover_v7",
    },
    ACTIVSG2000_EXPERIMENT_ID: {
        "case_name": "ACTIVSg2000",
        "tag": "experiment-2000-gpu-lagrangian-v1",
        "policy": "gpu_pdlp_phase_one_first_plus_lagrangian_cover_activsg2000_v1",
    },
    ACTIVSG2000_V2_EXPERIMENT_ID: {
        "case_name": "ACTIVSg2000",
        "tag": "experiment-2000-gpu-lagrangian-v2",
        "policy": ("gpu_dispatch_projection_phase_one_primal_plus_lagrangian_cover_activsg2000_v2"),
    },
    ACTIVSG2000_V3_EXPERIMENT_ID: {
        "case_name": "ACTIVSg2000",
        "tag": "experiment-2000-gpu-lagrangian-v3",
        "policy": (
            "gpu_dispatch_projection_candidate_rejection_plus_lagrangian_cover_activsg2000_v3"
        ),
    },
    ACTIVSG2000_V4_EXPERIMENT_ID: {
        "case_name": "ACTIVSg2000",
        "tag": "experiment-2000-gpu-lagrangian-v4",
        "policy": (
            "gpu_network_aware_commitment_repair_plus_phase_one_first_"
            "lagrangian_cover_activsg2000_v4"
        ),
    },
    ACTIVSG2000_V5_EXPERIMENT_ID: {
        "case_name": "ACTIVSg2000",
        "tag": "experiment-2000-gpu-lagrangian-v5",
        "policy": (
            "gpu_network_aware_commitment_repair_plus_phase_one_first_"
            "safe_residual_refinement_lagrangian_cover_activsg2000_v5"
        ),
    },
    ACTIVSG2000_V6_EXPERIMENT_ID: {
        "case_name": "ACTIVSg2000",
        "tag": "experiment-2000-gpu-lagrangian-v6",
        "policy": (
            "gpu_heuristics_primal_plus_stable2_stateful_phase_one_"
            "lagrangian_cover_activsg2000_v6"
        ),
    },
    ACTIVSG2000_V7_EXPERIMENT_ID: {
        "case_name": "ACTIVSg2000",
        "tag": "experiment-2000-gpu-lagrangian-v7",
        "policy": (
            "gpu_heuristics_primal_plus_fail_closed_stable2_state_"
            "lagrangian_cover_activsg2000_v7"
        ),
    },
    ACTIVSG2000_V8_EXPERIMENT_ID: {
        "case_name": "ACTIVSg2000",
        "tag": "experiment-2000-gpu-lagrangian-v8",
        "policy": (
            "gpu_phase_one_benders_commitment_repair_plus_final_only_"
            "replayed_lagrangian_cover_activsg2000_v8"
        ),
    },
    ACTIVSG2000_V9_EXPERIMENT_ID: {
        "case_name": "ACTIVSg2000",
        "tag": "experiment-2000-gpu-lagrangian-v9",
        "policy": (
            "gpu_phase_one_benders_commitment_repair_plus_order_independent_"
            "replayed_lagrangian_cover_activsg2000_v9"
        ),
    },
    ACTIVSG2000_V10_EXPERIMENT_ID: {
        "case_name": "ACTIVSg2000",
        "tag": "experiment-2000-gpu-lagrangian-v10",
        "policy": (
            "gpu_type_group_primal_plus_cardinality_disjunctive_"
            "replayed_lagrangian_cover_activsg2000_v10"
        ),
    },
    ACTIVSG2000_V11_EXPERIMENT_ID: {
        "case_name": "ACTIVSg2000",
        "tag": "experiment-2000-gpu-lagrangian-v11",
        "policy": (
            "gpu_secure_incumbent_mip_start_plus_preconditioned_global_cuts_"
            "and_monotone_sequential_cardinality_cover_activsg2000_v11"
        ),
    },
}
V2_BUGFIX_CHANGE = {
    "comparison_baseline": "activsg500-gpu-lagrangian-v1",
    "coefficient_cleanup": (
        "drop_affine_dispatch_coefficients_abs_le_1e-14_with_box_rhs_relaxation"
    ),
    "pdlp_primal_gate": "never_screen_or_add_rows_from_primal_infeasible_vector",
    "continuation": ("warm_start_same_master_after_time_limit_until_feasible_or_global_deadline"),
}
V3_CONTROLLER_CHANGE = {
    "comparison_baseline": "activsg500-gpu-lagrangian-v2",
    "root_certificate_persistence": "checkpoint_before_any_primal_repair",
    "candidate_budget": "15_seconds_total_with_5_second_pdlp_slices",
    "candidate_gate": "reject_dual_divergence_or_two_round_residual_stagnation",
    "candidate_queue": (
        "root_pdlp_rounding_then_gpu_lagrangian_then_low_threshold_then_all_online"
    ),
    "global_deadline": "never_convert_deadline_exceeded_into_candidate_rejection",
}
V4_CONTROLLER_CHANGE = {
    "comparison_baseline": "activsg500-gpu-lagrangian-v3",
    "security_row_deduplication": (
        "exact_post_cleanup_fp64_solver_rows_with_all_source_pair_ids_retained"
    ),
    "region_attempts": "bounded_warm_attempt_then_one_cold_restart",
    "failed_split_policy": "rollback_transaction_and_try_next_deterministic_generator",
    "infeasible_leaf_gate": "replayable_gpu_phase_one_box_dual_certificate",
    "secure_incumbent": "serialize_and_independently_verify_before_bound_refinement",
    "portable_lodf_replay": "absolute_tolerance_1e-12",
}
V5_BUGFIX_CHANGE = {
    "comparison_baseline": "activsg500-gpu-lagrangian-v4",
    "phase_one_row_identity": "semantic_source_row_and_side_order_independent_v1",
    "legacy_phase_one_replay": "ordered_hash_then_semantic_alignment",
    "cleanup_audit_replay": "exact_proof_invariants_with_portable_fp64_dust_telemetry",
    "gap_bookkeeping": "refresh_at_every_frontier_checkpoint",
}
V6_CONTROLLER_CHANGE = {
    "comparison_baseline": "activsg500-gpu-lagrangian-v5",
    "child_order": "exact_pmin_pmax_capacity_gate_then_short_gpu_phase_one_then_cost_lp",
    "capacity_gate": "aggregate_exact_source_pmin_pmax_interval_telemetry_only",
    "phase_one_precheck_budget": "2_seconds_per_child_within_600_second_global_deadline",
    "prune_authority": "positive_independently_replayable_phase_one_dual_only",
    "feasible_child_handoff": "phase_one_source_native_primal_to_cost_lp_without_row_dual",
    "uncertain_child_fallback": "ordinary_cost_lp_then_full_phase_one_if_cost_attempt_rejects",
}
V7_BUGFIX_CHANGE = {
    "comparison_baseline": "activsg500-gpu-lagrangian-v6",
    "prepared_region_commitment_column_mapping": (
        "generator_source_row_key_to_canonical_commitment_column_v1"
    ),
    "failed_v6_run_preserved": True,
}
ACTIVSG2000_V1_SCALE_CHANGE = {
    "algorithm_baseline": "activsg500-gpu-lagrangian-v7",
    "case_name": "ACTIVSg2000",
    "deadline_seconds": 1800.0,
    "root_pdlp_round_cap_seconds": 480.0,
    "bounded_candidate_and_region_seconds": 90.0,
    "short_phase_one_seconds_per_child": 10.0,
    "full_phase_one_seconds": 60.0,
    "coefficient_cleanup_zero_tolerance": 1e-9,
    "coefficient_cleanup_proof": "box_rhs_outward_relaxation_v1",
    "mathematical_model_changed": False,
    "exact_source_pmin_changed": False,
}
ACTIVSG2000_V2_FEASIBILITY_FIX = {
    "comparison_baseline": "activsg2000-gpu-lagrangian-v1",
    "fixed_commitment_phase_one_projection": (
        "free_committed_dispatch_columns_plus_unchanged_coupling_rows_v1"
    ),
    "local_generator_variable_elimination": (
        "fixed_commitment_and_pwl_segment_columns_removed_for_feasibility_only"
    ),
    "exact_source_pmin_pmax": "retained_without_clipping_relaxation_or_replacement",
    "secure_primal_checkpoint": (
        "lift_to_exact_ten_segment_model_and_independently_verify_before_cost_polish"
    ),
    "cost_polish_warm_start": "verified_phase_one_dispatch_in_exact_source_column_space",
    "cost_polish_failure_policy": "retain_verified_feasibility_incumbent",
    "mathematical_feasible_set_changed": False,
    "cpu_commitment_or_dispatch_seeded": False,
}
ACTIVSG2000_V3_CANDIDATE_REJECTION_FIX = {
    "comparison_baseline": "activsg2000-gpu-lagrangian-v2",
    "failed_v2_run_preserved": True,
    "constant_coupling_row_result": ("reject_only_the_fixed_commitment_candidate_before_pdlp"),
    "full_model_infeasibility_claimed": False,
    "continue_candidate_queue": True,
    "mathematical_model_changed": False,
    "exact_source_pmin_changed": False,
    "runtime_policy_changed": False,
    "cpu_commitment_or_dispatch_seeded": False,
}
ACTIVSG2000_V4_UTILIZATION_CHANGE = {
    "comparison_baseline": "activsg2000-gpu-lagrangian-v3",
    "initial_candidate_pipeline": ("exact_capacity_then_exact_projection_phase_one_then_cost_lp"),
    "network_aware_candidate_generation": (
        "violated_constant_coupling_row_ranked_one_and_two_flip_repairs_v1"
    ),
    "phase_one_cost_warm_start": "mapped_source_primal_and_native_row_dual_v1",
    "native_scaling": "power_system_equilibrated_v2_exact_diagonal",
    "security_row_policy": ("retain_nearly_dependent_rows_and_remove_only_proven_exact_duplicates"),
    "parallel_child_policy": (
        "two_independent_solver_contexts_after_capability_smoke_and_frontier_size_two"
    ),
    "exact_source_pmin_changed": False,
    "mathematical_feasible_set_changed": False,
    "cpu_commitment_or_dispatch_seeded": False,
}
ACTIVSG2000_V5_NUMERICAL_FIX = {
    "comparison_baseline": "activsg2000-gpu-lagrangian-v4",
    "failed_v4_run_preserved": True,
    "native_scaling": (
        "power_system_equilibrated_safe_v3_never_weakens_per_unit_row_scale"
    ),
    "root_canonical_residual_refinement": (
        "one_warm_pdlp_resolve_at_1e-10_before_fail_closed_rejection"
    ),
    "canonical_residual_attribution": (
        "worst_row_name_side_activity_bound_violation_and_native_scale_v1"
    ),
    "inherited_v4_utilization_changes": True,
    "exact_source_pmin_changed": False,
    "mathematical_feasible_set_changed": False,
    "cpu_commitment_or_dispatch_seeded": False,
}
ACTIVSG2000_V6_NUMERICAL_AND_RUNTIME_FIX = {
    "comparison_baseline": "activsg2000-gpu-lagrangian-v5",
    "failed_v5_run_preserved": True,
    "same_shape_pdlp_continuation": "full_native_stable2_solver_state_v1",
    "changed_shape_pdlp_continuation": "raw_primal_plus_identity_safe_dual_v1",
    "best_primal_retention": True,
    "primal_generator": (
        "reduced_gpu_heuristics_then_sparse_full_gpu_heuristics_"
        "with_complete_gpu_feasible_start_v3"
    ),
    "primal_native_conditioning": "rate_a_redundant_finite_angle_bounds_v1",
    "primal_budget_policy": "75_second_seed_with_105_second_pipeline_cap_v1",
    "primal_dual_bound_imported": False,
    "child_bound_engine": (
        "phase_one_feasible_point_plus_parent_inherited_gpu_lagrangian_dual_v1"
    ),
    "ordinary_child_cost_lp_removed": True,
    "partial_mip_start_policy": (
        "never_submit_unextended_commitment_as_native_full_assignment"
    ),
    "exact_source_pmin_changed": False,
    "mathematical_feasible_set_changed": False,
    "cpu_commitment_dispatch_objective_or_bound_seeded": False,
}
ACTIVSG2000_V7_NUMERICAL_ROBUSTNESS_FIX = {
    "comparison_baseline": "activsg2000-gpu-lagrangian-v6",
    "failed_v6_run_preserved": True,
    "opaque_v6_exception": "cuopt_26_6_time_limit_stable2_state_len_none",
    "same_shape_pdlp_continuation": (
        "optimal_complete_state_else_raw_primal_dual_v3"
    ),
    "cost_polish_initial_dual_policy": (
        "verified_phase_one_primal_only_because_phase_one_dual_has_different_objective"
    ),
    "cost_polish_formulation": (
        "exact_fixed_commitment_convex_pwl_epigraph_without_fixed_u_or_segment_columns_v1"
    ),
    "cost_polish_solver": (
        "single_cuopt_gpu_pdlp_slice_then_cupy_balance_and_feasible_segment_v2"
    ),
    "barrier_policy": "disabled_after_first_newton_step_nan_factorization_diagnostic",
    "time_limit_primal_recovery": (
        "gpu_boxed_balance_then_exhaustive_target_screen_then_convex_segment_v1"
    ),
    "repair_residual_budget_fraction": 0.5,
    "pricing_policy": "fail_closed_if_repair_breaks_primal_dual_complementarity",
    "exception_attribution": "bounded_worker_traceback_persisted_v1",
    "best_primal_retention": True,
    "exact_source_pmin_changed": False,
    "mathematical_feasible_set_changed": False,
    "cpu_commitment_dispatch_objective_or_bound_seeded": False,
}
ACTIVSG2000_V8_CERTIFICATE_CONTROLLER_FIX = {
    "comparison_baseline": "activsg2000-gpu-lagrangian-v7",
    "failed_v7_run_preserved": True,
    "fixed_commitment_positive_phase_one": (
        "single_optimal_solve_then_independently_replayed_dual_certificate_v1"
    ),
    "commitment_repair": (
        "phase_one_dual_binary_benders_cut_with_exact_conditional_pmin_pmax_v1"
    ),
    "repair_acceptance": "exact_projected_phase_one_then_exhaustive_gpu_screen",
    "secure_seed_numerical_margin": (
        "quarter_tolerance_phase_one_then_expanded_row_reprojection_v2"
    ),
    "candidate_scheduler": "64_unique_attempt_cap_with_certified_cut_repairs_first",
    "frontier_replay": "root_checkpoint_then_one_mandatory_final_full_replay_v1",
    "frontier_replay_master_cache": "exact_security_pair_signature_v1",
    "lagrangian_certificate_serialization": "sparse_nonzero_dual_identity_hashed_v2",
    "gpu_cut_lower_bound_used": False,
    "cut_pruning_authority_used": False,
    "exact_source_pmin_changed": False,
    "mathematical_feasible_set_changed": False,
    "cpu_commitment_dispatch_objective_or_bound_seeded": False,
}
ACTIVSG2000_V9_CERTIFICATE_CONTROLLER_FIX = {
    **ACTIVSG2000_V8_CERTIFICATE_CONTROLLER_FIX,
    "lagrangian_certificate_serialization": (
        "sparse_nonzero_dual_order_independent_identity_v3"
    ),
}
ACTIVSG2000_V11_CERTIFICATE_CONTROLLER_FIX = {
    **ACTIVSG2000_V9_CERTIFICATE_CONTROLLER_FIX,
    "comparison_baseline": "activsg2000-gpu-lagrangian-v10",
    "gpu_cut_lower_bound_used": True,
}
ACTIVSG2000_V9_COMPACT_REPLAY_FIX = {
    "comparison_baseline": "activsg2000-gpu-lagrangian-v8",
    "failed_v8_run_preserved": True,
    "failure": "compact_coupling_row_insertion_order_hash_mismatch",
    "coupling_row_identity": "sorted_complete_row_name_set_sha256_v1",
    "lagrangian_accumulation_order": "row_name_sorted_cpu_and_gpu_v1",
    "derived_fp64_vector_hash_gate_removed": True,
    "replayed_bound_tolerance_dollars_unchanged": 1e-6,
    "exact_source_pmin_changed": False,
    "mathematical_feasible_set_changed": False,
    "cpu_commitment_dispatch_objective_or_bound_seeded": False,
}
ACTIVSG2000_V10_CARDINALITY_REFINEMENT = {
    "comparison_baseline": "activsg2000-gpu-lagrangian-v9",
    "v9_result_preserved": True,
    "v9_numerical_failures": 0,
    "v9_phase_one_prechecks": 83,
    "v9_phase_one_prunes": 0,
    "v9_child_lagrangian_improvement_dollars": 0.0,
    "primal_candidate": (
        "fresh_root_gpu_lp_exact_pmin_pmax_pwl_type_count_rounding_v1"
    ),
    "disjunction": (
        "multi_unit_integer_cardinality_sums_exact_types_then_laminar_"
        "with_binary_completeness_fallback_v2"
    ),
    "minimum_cardinality_subset_size": 2,
    "binary_fallback": "only_after_no_fractional_multi_unit_sum_remains",
    "child_phase_one": "disabled_after_zero_of_83_v9_prunes",
    "child_warm_start": (
        "row_name_mapped_parent_dual_only_because_parent_primal_violates_"
        "the_new_cardinality_branch_v2"
    ),
    "certificate": "commitment_upper_cut_lagrangian_dual_replay_v1",
    "exact_source_pmin_changed": False,
    "mathematical_original_integer_optimum_changed": False,
    "cpu_commitment_dispatch_objective_or_bound_seeded": False,
}
ACTIVSG2000_V11_NUMERICAL_RUNTIME_FIX = {
    "comparison_baseline": "activsg2000-gpu-lagrangian-v10",
    "failed_v10_run_preserved": True,
    "concurrent_child_contexts": "disabled_after_missing_vector_and_stagnation_failures_v1",
    "split_transaction": (
        "two_sequential_bounded_children_preflighted_before_launch_and_rollback_on_deadline_v1"
    ),
    "child_certificate": "maximum_of_child_gpu_certificate_and_replayed_parent_certificate_v1",
    "lagrangian_preconditioning": (
        "exact_native_positive_row_scaling_for_coupling_and_commitment_cut_subgradients_v1"
    ),
    "global_feasibility_cuts": (
        "replayed_phase_one_binary_benders_cuts_applied_to_every_frontier_region_v1"
    ),
    "primal_heuristics": (
        "always_run_with_independently_verified_gpu_incumbent_as_complete_dimension_matched_start_v1"
    ),
    "mip_start": "all_canonical_columns_presolve_off_native_readback_fail_closed_v2",
    "finalization": "verification_and_serialization_reserve_cannot_be_consumed_by_child_solve_v1",
    "exact_source_pmin_changed": False,
    "mathematical_original_integer_optimum_changed": False,
    "cpu_commitment_dispatch_objective_or_bound_seeded": False,
}
ACTIVSG2000_V1_RUNTIME = {
    "deadline_seconds": 1800.0,
    "verification_reserve_seconds": 120.0,
    "serialization_reserve_seconds": 30.0,
    "maximum_constraint_generation_rounds": 100,
    "maximum_pdlp_round_seconds": 480.0,
    "maximum_frontier_regions": 64,
    "maximum_primal_repairs": 8,
    "maximum_primal_candidate_seconds": 90.0,
    "maximum_primal_candidate_round_seconds": 30.0,
    "minimum_primal_candidate_round_seconds": 1.0,
    "primal_candidate_stagnation_window_rounds": 2,
    "primal_candidate_minimum_relative_residual_improvement": 0.01,
    "primal_candidate_dual_divergence_multiple": 1e6,
    "primal_candidate_cold_restart_attempts": 1,
    "maximum_region_attempt_seconds": 90.0,
    "maximum_region_attempt_round_seconds": 30.0,
    "minimum_region_attempt_round_seconds": 1.0,
    "region_attempt_stagnation_window_rounds": 2,
    "region_attempt_minimum_relative_residual_improvement": 0.01,
    "region_attempt_dual_divergence_multiple": 1e6,
    "region_attempt_cold_restart_attempts": 1,
    "maximum_failed_split_attempts": 16,
    "precheck_phase_one_time_limit_seconds": 10.0,
    "phase_one_time_limit_seconds": 60.0,
    "phase_one_maximum_violation_pu": 1e6,
    "phase_one_safety_margin_pu": 1e-8,
    "phase_one_infeasibility_threshold_pu": 1e-6,
}
ACTIVSG2000_V4_RUNTIME = {
    **ACTIVSG2000_V1_RUNTIME,
    "maximum_primal_repairs": 12,
    "maximum_network_repair_candidates_per_rejection": 4,
    "network_repair_pair_search_limit": 32,
    "parallel_child_solver_contexts": 2,
    "parallel_child_minimum_frontier_regions": 2,
}
ACTIVSG2000_V5_RUNTIME = {
    **ACTIVSG2000_V4_RUNTIME,
    "root_canonical_residual_refinement_attempts": 1,
    "root_canonical_residual_refinement_optimality_tolerance": 1e-10,
}
ACTIVSG2000_V6_RUNTIME = {
    **ACTIVSG2000_V5_RUNTIME,
    "deadline_seconds": 990.0,
    "verification_reserve_seconds": 60.0,
    "serialization_reserve_seconds": 15.0,
    "maximum_pdlp_round_seconds": 180.0,
    "maximum_frontier_regions": 128,
    "maximum_failed_split_attempts": 64,
    "gpu_primal_heuristics_seconds": 105.0,
    "gpu_primal_seed_seconds": 75.0,
    "phase_lagrangian_gpu_iterations": 2048,
}
ACTIVSG2000_V7_RUNTIME = dict(ACTIVSG2000_V6_RUNTIME)
ACTIVSG2000_V8_RUNTIME = {
    **ACTIVSG2000_V7_RUNTIME,
    "maximum_frontier_regions": 192,
    "maximum_primal_repairs": 64,
    "maximum_commitment_cut_repairs_per_rejection": 3,
    "fixed_commitment_secure_seed_tolerance_fraction": 0.25,
    "frontier_replay_after_every_split": False,
    "compact_lagrangian_certificates": True,
    "minimum_refinement_launch_seconds": 20.0,
}
ACTIVSG2000_V9_RUNTIME = dict(ACTIVSG2000_V8_RUNTIME)
ACTIVSG2000_V10_RUNTIME = {
    **ACTIVSG2000_V9_RUNTIME,
    "maximum_frontier_regions": 256,
    "phase_lagrangian_gpu_iterations": 512,
}
ACTIVSG2000_V11_RUNTIME = {
    **ACTIVSG2000_V10_RUNTIME,
    "parallel_child_solver_contexts": 1,
    "gpu_primal_seed_seconds": 1.0,
    "minimum_refinement_launch_seconds": 195.0,
    "split_transaction_margin_seconds": 15.0,
    "lagrangian_diagonal_preconditioning": True,
    "global_feasibility_cuts_in_lagrangian": True,
    "always_run_gpu_primal_heuristics": True,
}


@dataclass
class SolvedRegion:
    region_id: str
    masks: RegionMasks
    master: ReducedMaster
    solve: ContinuousSolveResult
    canonical_row_dual: np.ndarray
    lagrangian: LagrangianEvaluation
    commitment: np.ndarray
    security_pairs: tuple[SecurityPair, ...]
    rounds: list[dict[str, Any]]
    final_screen: dict[str, Any]
    gpu_lagrangian: dict[str, Any]
    commitment_cuts: tuple[CommitmentUpperCut, ...] = ()
    commitment_cut_row_by_id: dict[str, int] | None = None


@dataclass(frozen=True)
class PrimalCandidatePolicy:
    total_seconds: float
    maximum_round_seconds: float
    minimum_round_seconds: float
    stagnation_window_rounds: int
    minimum_relative_residual_improvement: float
    dual_divergence_multiple: float
    cold_restart_attempts: int

    @classmethod
    def from_config(
        cls, config: RunConfig, *, scope: str = "primal_candidate"
    ) -> PrimalCandidatePolicy:
        runtime = config.runtime
        if scope not in {"primal_candidate", "disjunctive_region"}:
            raise ScopfError(f"Unknown bounded PDLP policy scope: {scope}")
        prefix = "primal_candidate" if scope == "primal_candidate" else "region_attempt"
        return cls(
            total_seconds=float(runtime[f"maximum_{prefix}_seconds"]),
            maximum_round_seconds=float(runtime[f"maximum_{prefix}_round_seconds"]),
            minimum_round_seconds=float(runtime[f"minimum_{prefix}_round_seconds"]),
            stagnation_window_rounds=int(runtime[f"{prefix}_stagnation_window_rounds"]),
            minimum_relative_residual_improvement=float(
                runtime[f"{prefix}_minimum_relative_residual_improvement"]
            ),
            dual_divergence_multiple=float(runtime[f"{prefix}_dual_divergence_multiple"]),
            cold_restart_attempts=int(runtime.get(f"{prefix}_cold_restart_attempts", 0)),
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_seconds": self.total_seconds,
            "maximum_round_seconds": self.maximum_round_seconds,
            "minimum_round_seconds": self.minimum_round_seconds,
            "stagnation_window_rounds": self.stagnation_window_rounds,
            "minimum_relative_residual_improvement": (self.minimum_relative_residual_improvement),
            "dual_divergence_multiple": self.dual_divergence_multiple,
            "cold_restart_attempts": self.cold_restart_attempts,
        }


class RegionAttemptRejected(PrimalCandidateRejected):
    """Bounded PDLP attempt rejected with its exact append-only model retained."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        master: ReducedMaster,
        security_pairs: tuple[SecurityPair, ...],
        rounds: list[dict[str, Any]],
        commitment_feasibility_cut: CommitmentFeasibilityCut | None = None,
        commitment_feasibility_cut_record: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.master = master
        self.security_pairs = security_pairs
        self.rounds = rounds
        self.commitment_feasibility_cut = commitment_feasibility_cut
        self.commitment_feasibility_cut_record = commitment_feasibility_cut_record


@dataclass
class PhaseOneAttemptResult:
    record: dict[str, Any]
    source_native_primal: np.ndarray | None
    source_native_row_dual: np.ndarray | None
    source_values: np.ndarray | None = None


@dataclass
class FixedCommitmentFeasibilityResult:
    master: ReducedMaster
    source_values: np.ndarray
    source_native_row_dual: np.ndarray | None
    security_pairs: tuple[SecurityPair, ...]
    rounds: list[dict[str, Any]]
    final_screen: dict[str, Any]


@dataclass
class FixedCommitmentCostResult:
    master: ReducedMaster
    source_values: np.ndarray
    canonical_row_dual: np.ndarray
    projected_solve: ContinuousSolveResult
    security_pairs: tuple[SecurityPair, ...]
    rounds: list[dict[str, Any]]
    final_screen: dict[str, Any]
    pricing_certified: bool
    pricing_audit: dict[str, Any]


@dataclass(frozen=True)
class NetworkCommitmentRepair:
    """A deterministic binary repair for one violated coupling row."""

    commitment: np.ndarray
    audit: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read experiment evidence {path}: {exc}") from exc


def _write_console(path: Path, stdout: str | bytes | None, stderr: str | bytes | None) -> None:
    def decode(value: str | bytes | None) -> str:
        return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(decode(stdout) + decode(stderr), encoding="utf-8")


def validate_lagrangian_experiment_config(config: RunConfig) -> dict[str, Any]:
    benchmark = config.raw["benchmark"]
    experiment = REGISTERED_EXPERIMENTS.get(config.benchmark_id)
    if experiment is None:
        raise ScopfError(f"Unregistered GPU Lagrangian experiment: {config.benchmark_id!r}")
    expected_case_name = str(experiment.get("case_name", "ACTIVSg500"))
    if config.case_name != expected_case_name:
        raise ScopfError(
            "GPU Lagrangian experiment case changed: "
            f"expected={expected_case_name!r}, observed={config.case_name!r}"
        )
    is_activsg2000_v1 = config.benchmark_id == ACTIVSG2000_EXPERIMENT_ID
    is_activsg2000_v2 = config.benchmark_id == ACTIVSG2000_V2_EXPERIMENT_ID
    is_activsg2000_v3 = config.benchmark_id == ACTIVSG2000_V3_EXPERIMENT_ID
    is_activsg2000_v4 = config.benchmark_id == ACTIVSG2000_V4_EXPERIMENT_ID
    is_activsg2000_v5 = config.benchmark_id == ACTIVSG2000_V5_EXPERIMENT_ID
    is_activsg2000_v6 = config.benchmark_id == ACTIVSG2000_V6_EXPERIMENT_ID
    is_activsg2000_v7 = config.benchmark_id == ACTIVSG2000_V7_EXPERIMENT_ID
    is_activsg2000_v8 = config.benchmark_id == ACTIVSG2000_V8_EXPERIMENT_ID
    is_activsg2000_v9 = config.benchmark_id == ACTIVSG2000_V9_EXPERIMENT_ID
    is_activsg2000_v10 = config.benchmark_id == ACTIVSG2000_V10_EXPERIMENT_ID
    is_activsg2000_v11 = config.benchmark_id == ACTIVSG2000_V11_EXPERIMENT_ID
    is_activsg2000_v8_plus = (
        is_activsg2000_v8
        or is_activsg2000_v9
        or is_activsg2000_v10
        or is_activsg2000_v11
    )
    is_activsg2000 = (
        is_activsg2000_v1
        or is_activsg2000_v2
        or is_activsg2000_v3
        or is_activsg2000_v4
        or is_activsg2000_v5
        or is_activsg2000_v6
        or is_activsg2000_v7
        or is_activsg2000_v8
        or is_activsg2000_v9
        or is_activsg2000_v10
        or is_activsg2000_v11
    )
    if benchmark.get("kind") != "gpu_lagrangian_disjunctive_experiment":
        raise ScopfError("GPU Lagrangian experiment kind changed")
    if benchmark.get("required_git_tag") != experiment["tag"]:
        raise ScopfError("GPU Lagrangian frozen tag changed")
    coefficient_cleanup_experiment = (
        config.benchmark_id.endswith(
            (
                "-v2",
                "-v3",
                "-v4",
                "-v5",
                "-v6",
                "-v7",
                "-v8",
                "-v9",
                "-v10",
                "-v11",
            )
        )
        or is_activsg2000
    )
    expected_coefficient_tolerance = 1e-9 if is_activsg2000 else 1e-14
    if (
        coefficient_cleanup_experiment
        and float(config.model.get("reduced_coefficient_zero_tolerance", -1.0))
        != expected_coefficient_tolerance
    ):
        raise ScopfError("GPU Lagrangian coefficient threshold changed")
    if config.benchmark_id.endswith("-v2") and not is_activsg2000_v2:
        observed_change = benchmark.get("bugfix_change")
        if observed_change != V2_BUGFIX_CHANGE:
            raise ScopfError(
                "GPU Lagrangian v2 bugfix identity changed: "
                f"expected={V2_BUGFIX_CHANGE}, observed={observed_change}"
            )
    if config.benchmark_id.endswith("-v3") and not is_activsg2000_v3:
        observed_change = benchmark.get("controller_change")
        if observed_change != V3_CONTROLLER_CHANGE:
            raise ScopfError(
                "GPU Lagrangian v3 controller identity changed: "
                f"expected={V3_CONTROLLER_CHANGE}, observed={observed_change}"
            )
    if config.benchmark_id.endswith("-v4") and not is_activsg2000_v4:
        observed_change = benchmark.get("controller_change")
        if observed_change != V4_CONTROLLER_CHANGE:
            raise ScopfError(
                "GPU Lagrangian v4 controller identity changed: "
                f"expected={V4_CONTROLLER_CHANGE}, observed={observed_change}"
            )
    if config.benchmark_id.endswith("-v5") and not is_activsg2000_v5:
        observed_change = benchmark.get("bugfix_change")
        if observed_change != V5_BUGFIX_CHANGE:
            raise ScopfError(
                "GPU Lagrangian v5 bugfix identity changed: "
                f"expected={V5_BUGFIX_CHANGE}, observed={observed_change}"
            )
    if config.benchmark_id.endswith("-v6") and not is_activsg2000_v6:
        observed_change = benchmark.get("controller_change")
        if observed_change != V6_CONTROLLER_CHANGE:
            raise ScopfError(
                "GPU Lagrangian v6 controller identity changed: "
                f"expected={V6_CONTROLLER_CHANGE}, observed={observed_change}"
            )
    if config.benchmark_id.endswith("-v7") and not is_activsg2000_v7:
        observed_change = benchmark.get("bugfix_change")
        if observed_change != V7_BUGFIX_CHANGE:
            raise ScopfError(
                "GPU Lagrangian v7 bugfix identity changed: "
                f"expected={V7_BUGFIX_CHANGE}, observed={observed_change}"
            )
    if is_activsg2000_v1:
        observed_change = benchmark.get("scale_change")
        if observed_change != ACTIVSG2000_V1_SCALE_CHANGE:
            raise ScopfError(
                "ACTIVSg2000 GPU Lagrangian v1 scale identity changed: "
                f"expected={ACTIVSG2000_V1_SCALE_CHANGE}, observed={observed_change}"
            )
    if is_activsg2000_v2:
        observed_change = benchmark.get("feasibility_fix")
        if observed_change != ACTIVSG2000_V2_FEASIBILITY_FIX:
            raise ScopfError(
                "ACTIVSg2000 GPU Lagrangian v2 feasibility-fix identity changed: "
                f"expected={ACTIVSG2000_V2_FEASIBILITY_FIX}, "
                f"observed={observed_change}"
            )
    if is_activsg2000_v3:
        observed_change = benchmark.get("candidate_rejection_fix")
        if observed_change != ACTIVSG2000_V3_CANDIDATE_REJECTION_FIX:
            raise ScopfError(
                "ACTIVSg2000 GPU Lagrangian v3 candidate-rejection identity changed: "
                f"expected={ACTIVSG2000_V3_CANDIDATE_REJECTION_FIX}, "
                f"observed={observed_change}"
            )
    if is_activsg2000_v4:
        observed_change = benchmark.get("utilization_change")
        if observed_change != ACTIVSG2000_V4_UTILIZATION_CHANGE:
            raise ScopfError(
                "ACTIVSg2000 GPU Lagrangian v4 utilization identity changed: "
                f"expected={ACTIVSG2000_V4_UTILIZATION_CHANGE}, "
                f"observed={observed_change}"
            )
    if is_activsg2000_v5:
        observed_change = benchmark.get("numerical_fix")
        if observed_change != ACTIVSG2000_V5_NUMERICAL_FIX:
            raise ScopfError(
                "ACTIVSg2000 GPU Lagrangian v5 numerical-fix identity changed: "
                f"expected={ACTIVSG2000_V5_NUMERICAL_FIX}, "
                f"observed={observed_change}"
            )
    if is_activsg2000_v6:
        observed_change = benchmark.get("numerical_and_runtime_fix")
        if observed_change != ACTIVSG2000_V6_NUMERICAL_AND_RUNTIME_FIX:
            raise ScopfError(
                "ACTIVSg2000 GPU Lagrangian v6 fix identity changed: "
                f"expected={ACTIVSG2000_V6_NUMERICAL_AND_RUNTIME_FIX}, "
                f"observed={observed_change}"
            )
    if is_activsg2000_v7:
        observed_change = benchmark.get("numerical_robustness_fix")
        if observed_change != ACTIVSG2000_V7_NUMERICAL_ROBUSTNESS_FIX:
            raise ScopfError(
                "ACTIVSg2000 GPU Lagrangian v7 numerical-robustness identity changed: "
                f"expected={ACTIVSG2000_V7_NUMERICAL_ROBUSTNESS_FIX}, "
                f"observed={observed_change}"
            )
    if is_activsg2000_v8_plus:
        observed_change = benchmark.get("certificate_controller_fix")
        expected_change = (
            ACTIVSG2000_V11_CERTIFICATE_CONTROLLER_FIX
            if is_activsg2000_v11
            else ACTIVSG2000_V9_CERTIFICATE_CONTROLLER_FIX
            if (is_activsg2000_v9 or is_activsg2000_v10)
            else ACTIVSG2000_V8_CERTIFICATE_CONTROLLER_FIX
        )
        if observed_change != expected_change:
            raise ScopfError(
                "ACTIVSg2000 GPU Lagrangian v8 certificate/controller identity changed: "
                f"expected={expected_change}, "
                f"observed={observed_change}"
            )
    if is_activsg2000_v9 or is_activsg2000_v10 or is_activsg2000_v11:
        observed_change = benchmark.get("compact_replay_fix")
        if observed_change != ACTIVSG2000_V9_COMPACT_REPLAY_FIX:
            raise ScopfError(
                "ACTIVSg2000 GPU Lagrangian v9 compact-replay identity changed: "
                f"expected={ACTIVSG2000_V9_COMPACT_REPLAY_FIX}, "
                f"observed={observed_change}"
            )
    if is_activsg2000_v10 or is_activsg2000_v11:
        observed_change = benchmark.get("cardinality_refinement")
        if observed_change != ACTIVSG2000_V10_CARDINALITY_REFINEMENT:
            raise ScopfError(
                "ACTIVSg2000 GPU Lagrangian v10 cardinality identity changed: "
                f"expected={ACTIVSG2000_V10_CARDINALITY_REFINEMENT}, "
                f"observed={observed_change}"
            )
    if is_activsg2000_v11:
        observed_change = benchmark.get("numerical_runtime_fix")
        if observed_change != ACTIVSG2000_V11_NUMERICAL_RUNTIME_FIX:
            raise ScopfError(
                "ACTIVSg2000 GPU Lagrangian v11 numerical/runtime identity changed: "
                f"expected={ACTIVSG2000_V11_NUMERICAL_RUNTIME_FIX}, "
                f"observed={observed_change}"
            )
    profile = config.raw["platforms"].get("dgx_spark", {})
    required_profile = {
        "solver": "cuopt",
        "screening": "cupy",
        "lp_method": "pdlp",
        "pdlp_precision": "fp64",
        "native_scaling_mode": (
            "power_system_equilibrated_safe_v3"
            if (
                is_activsg2000_v5
                or is_activsg2000_v6
                or is_activsg2000_v7
                or is_activsg2000_v8
                or is_activsg2000_v9
                or is_activsg2000_v10
                or is_activsg2000_v11
            )
            else (
                "power_system_equilibrated_v2"
                if is_activsg2000_v4
                else "power_system_per_unit_v1"
            )
        ),
        "presolve": 0,
        "lagrangian_gpu_iterations": 512,
        "lagrangian_polyak_fraction": 0.5,
        "console_logging": True,
        "integer_solver": (
            "cuopt_gpu_heuristics_only_for_primal"
            if (
                is_activsg2000_v6
                or is_activsg2000_v7
                or is_activsg2000_v8
                or is_activsg2000_v9
                or is_activsg2000_v10
                or is_activsg2000_v11
            )
            else "none"
        ),
        "branch_and_bound": False,
        "custom_cuda_kernels": "allowed_but_not_required",
    }
    observed = {key: profile.get(key) for key in required_profile}
    if observed != required_profile:
        raise ScopfError(
            f"GPU Lagrangian profile changed: expected={required_profile}, observed={observed}"
        )
    runtime = config.runtime
    expected_deadline = (
        990.0
        if (
            is_activsg2000_v6
            or is_activsg2000_v7
            or is_activsg2000_v8
            or is_activsg2000_v9
            or is_activsg2000_v10
            or is_activsg2000_v11
        )
        else (1800.0 if is_activsg2000 else 600.0)
    )
    if float(runtime.get("deadline_seconds", 0.0)) != expected_deadline:
        raise ScopfError(
            "The registered GPU Lagrangian experiment deadline changed: "
            f"expected={expected_deadline}, observed={runtime.get('deadline_seconds')}"
        )
    expected_activsg2000_runtime = (
        ACTIVSG2000_V11_RUNTIME
        if is_activsg2000_v11
        else (
            ACTIVSG2000_V10_RUNTIME
            if is_activsg2000_v10
            else (
                ACTIVSG2000_V9_RUNTIME
                if is_activsg2000_v9
                else (
                    ACTIVSG2000_V8_RUNTIME
                    if is_activsg2000_v8
                    else (
                        ACTIVSG2000_V7_RUNTIME
                        if is_activsg2000_v7
                        else (
                            ACTIVSG2000_V6_RUNTIME
                            if is_activsg2000_v6
                            else (
                                ACTIVSG2000_V5_RUNTIME
                                if is_activsg2000_v5
                                else (
                                    ACTIVSG2000_V4_RUNTIME
                                    if is_activsg2000_v4
                                    else ACTIVSG2000_V1_RUNTIME
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    if is_activsg2000 and runtime != expected_activsg2000_runtime:
        raise ScopfError(
            "ACTIVSg2000 GPU Lagrangian runtime policy changed: "
            f"expected={expected_activsg2000_runtime}, observed={runtime}"
        )
    if int(runtime.get("maximum_frontier_regions", 0)) < 1:
        raise ScopfError("maximum_frontier_regions must be positive")
    if float(config.model["mip_relative_gap_tolerance"]) != 1e-3:
        raise ScopfError("The registered Lagrangian target is exactly 1e-3")
    if config.benchmark_id.endswith("-v3") and not is_activsg2000_v3:
        required_runtime = {
            "maximum_primal_candidate_seconds": 15.0,
            "maximum_primal_candidate_round_seconds": 5.0,
            "minimum_primal_candidate_round_seconds": 0.25,
            "primal_candidate_stagnation_window_rounds": 2,
            "primal_candidate_minimum_relative_residual_improvement": 0.01,
            "primal_candidate_dual_divergence_multiple": 1e6,
        }
        observed_runtime = {key: runtime.get(key) for key in required_runtime}
        if observed_runtime != required_runtime:
            raise ScopfError(
                "GPU Lagrangian v3 candidate policy changed: "
                f"expected={required_runtime}, observed={observed_runtime}"
            )
    if config.benchmark_id.endswith(
        ("-v4", "-v5", "-v6", "-v7", "-v8", "-v9", "-v10", "-v11")
    ) or is_activsg2000:
        required_runtime = {
            "maximum_primal_candidate_seconds": (90.0 if is_activsg2000 else 15.0),
            "maximum_primal_candidate_round_seconds": (30.0 if is_activsg2000 else 5.0),
            "minimum_primal_candidate_round_seconds": (1.0 if is_activsg2000 else 0.25),
            "primal_candidate_stagnation_window_rounds": 2,
            "primal_candidate_minimum_relative_residual_improvement": 0.01,
            "primal_candidate_dual_divergence_multiple": 1e6,
            "primal_candidate_cold_restart_attempts": 1,
            "maximum_region_attempt_seconds": 90.0 if is_activsg2000 else 15.0,
            "maximum_region_attempt_round_seconds": (30.0 if is_activsg2000 else 5.0),
            "minimum_region_attempt_round_seconds": (1.0 if is_activsg2000 else 0.25),
            "region_attempt_stagnation_window_rounds": 2,
            "region_attempt_minimum_relative_residual_improvement": 0.01,
            "region_attempt_dual_divergence_multiple": 1e6,
            "region_attempt_cold_restart_attempts": 1,
            "maximum_failed_split_attempts": (
                64
                if (
                    is_activsg2000_v6
                    or is_activsg2000_v7
                    or is_activsg2000_v8
                    or is_activsg2000_v9
                    or is_activsg2000_v10
                    or is_activsg2000_v11
                )
                else (16 if is_activsg2000 else 8)
            ),
            "phase_one_time_limit_seconds": 60.0 if is_activsg2000 else 15.0,
            "phase_one_maximum_violation_pu": 1e6,
            "phase_one_safety_margin_pu": 1e-8,
            "phase_one_infeasibility_threshold_pu": 1e-6,
        }
        observed_runtime = {key: runtime.get(key) for key in required_runtime}
        if observed_runtime != required_runtime:
            raise ScopfError(
                "GPU Lagrangian bounded-region policy changed: "
                f"expected={required_runtime}, observed={observed_runtime}"
            )
        if config.benchmark_id.endswith(
            ("-v6", "-v7", "-v8", "-v9", "-v10", "-v11")
        ) or is_activsg2000:
            expected_precheck = 10.0 if is_activsg2000 else 2.0
            if float(runtime.get("precheck_phase_one_time_limit_seconds", -1.0)) != (
                expected_precheck
            ):
                raise ScopfError("GPU Lagrangian short Phase-I budget changed")
        if float(config.model.get("serialized_lodf_replay_tolerance", -1.0)) != 1e-12:
            raise ScopfError("GPU Lagrangian v4/v5/v6/v7 LODF replay tolerance changed")
        if float(config.model.get("security_equivalence_replay_tolerance", -1.0)) != 1e-12:
            raise ScopfError("GPU Lagrangian v4/v5/v6/v7 security-row replay tolerance changed")
        if float(config.model.get("phase_one_replay_tolerance_pu", -1.0)) != 1e-10:
            raise ScopfError("GPU Lagrangian v4/v5/v6/v7 Phase-I replay tolerance changed")
    if (
        is_activsg2000_v6
        or is_activsg2000_v7
        or is_activsg2000_v8
        or is_activsg2000_v9
        or is_activsg2000_v10
        or is_activsg2000_v11
    ):
        required_v6_profile = {
            "pdlp_solver_mode_native": 1,
            "pdlp_solver_mode": "stable2",
            "save_best_primal_so_far": True,
            "integer_solver": "cuopt_gpu_heuristics_only_for_primal",
            "branch_and_bound": False,
        }
        observed_v6_profile = {
            key: profile.get(key) for key in required_v6_profile
        }
        if observed_v6_profile != required_v6_profile:
            raise ScopfError(
                "ACTIVSg2000 v6/v7 numerical profile changed: "
                f"expected={required_v6_profile}, observed={observed_v6_profile}"
            )
    if is_activsg2000_v7 and any(
        key in profile
        for key in ("cost_polish_lp_method", "cost_polish_method_native")
    ):
        raise ScopfError("ACTIVSg2000 v7 must use the registered PDLP-only profile")
    return {
        "benchmark": benchmark,
        "profile": profile,
        "cpu_comparison": benchmark["cpu_comparison"],
        "experiment_policy": experiment["policy"],
    }


def _solve_summary(result: ContinuousSolveResult) -> dict[str, Any]:
    certificate = result.statistics.get("dual_certificate", {})
    return {
        "status": result.status,
        "optimal": result.optimal,
        "primal_objective": result.primal_objective,
        "dual_objective": result.dual_objective,
        "native_solve_time_seconds": result.solve_time_seconds,
        "solved_by": result.statistics.get("solved_by"),
        "solved_by_pdlp": result.statistics.get("solved_by_pdlp"),
        "native_integer_columns": result.statistics.get("native_integer_columns"),
        "dual_certificate_passed": certificate.get("passed"),
        "primal_feasible": certificate.get("primal_feasible"),
        "native_log_final_metrics": result.statistics.get("native_log_final_metrics"),
        "native_log_audit": result.statistics.get("native_log_audit"),
        "native_log_sha256": result.statistics.get("native_log_sha256"),
        "concurrent_solver_context": result.statistics.get("concurrent_solver_context"),
        "concurrent_context_proof": result.statistics.get("concurrent_context_proof"),
        "native_lp_stats": result.statistics.get("lp_stats"),
        "termination_reason": result.statistics.get("termination_reason"),
        "warm_start": result.statistics.get("warm_start"),
        "returned_pdlp_warm_start_state": result.statistics.get(
            "returned_pdlp_warm_start_state"
        ),
        "pdlp_warm_start_state_returned": result.pdlp_warm_start_data is not None,
    }


def _worst_canonical_row_violation(
    model: Any,
    values: np.ndarray,
    *,
    base_mva: float,
    native_scaling_mode: str,
) -> dict[str, Any]:
    """Attribute the exact canonical row residual and its native scaling."""

    candidate = np.asarray(values, dtype=np.float64)
    if candidate.shape != (model.num_columns,):
        raise ScopfError("Canonical residual attribution received the wrong vector shape")
    activity = np.asarray(model.matrix_csr() @ candidate, dtype=np.float64)
    lower, upper = model.row_bound_arrays()
    lower_violation = np.where(np.isfinite(lower), lower - activity, -np.inf)
    upper_violation = np.where(np.isfinite(upper), activity - upper, -np.inf)
    lower_row = int(np.argmax(lower_violation)) if lower_violation.size else -1
    upper_row = int(np.argmax(upper_violation)) if upper_violation.size else -1
    lower_maximum = float(lower_violation[lower_row]) if lower_row >= 0 else -np.inf
    upper_maximum = float(upper_violation[upper_row]) if upper_row >= 0 else -np.inf
    if lower_maximum >= upper_maximum:
        row = lower_row
        side = "lower"
        bound = float(lower[row])
        violation = max(0.0, lower_maximum)
    else:
        row = upper_row
        side = "upper"
        bound = float(upper[row])
        violation = max(0.0, upper_maximum)
    if row < 0:
        return {
            "row_index": None,
            "row_name": None,
            "side": None,
            "activity": None,
            "bound": None,
            "violation": 0.0,
            "violation_pu": 0.0,
            "native_row_scale": None,
            "native_scaled_violation": 0.0,
        }
    _column_scale, row_scale = native_scaling_vectors(
        model,
        mode=native_scaling_mode,
        base_mva=base_mva,
    )
    return {
        "row_index": row,
        "row_name": model.row_names[row],
        "side": side,
        "activity": float(activity[row]),
        "bound": bound,
        "violation": violation,
        "violation_pu": violation / float(base_mva),
        "native_row_scale": float(row_scale[row]),
        "native_scaled_violation": violation * float(row_scale[row]),
    }


def _prepare_region_master(
    *,
    case: Any,
    network: NetworkData,
    config: RunConfig,
    masks: RegionMasks,
    initial_pairs: tuple[SecurityPair, ...],
    commitment_cuts: tuple[CommitmentUpperCut, ...] = (),
) -> ReducedMaster:
    master = build_reduced_master(
        case,
        network,
        segments=int(config.model["pwl_segments"]),
        coefficient_zero_tolerance=float(
            config.model.get("reduced_coefficient_zero_tolerance", 1e-14)
        ),
    )
    add_reduced_security_pairs(master, network, initial_pairs)
    fix_commitments(master, masks.fixed_off, masks.fixed_on)
    add_commitment_upper_cuts(master, commitment_cuts)
    return master


def _validate_prepared_region_master(
    master: ReducedMaster,
    masks: RegionMasks,
    initial_pairs: tuple[SecurityPair, ...],
    commitment_cuts: tuple[CommitmentUpperCut, ...] = (),
) -> None:
    expected_pair_ids = {pair.pair_id for pair in initial_pairs}
    if set(master.security_pair_ids) != expected_pair_ids:
        raise ScopfError("Prepared region master security-pair identity changed")
    masks.validate(master.index.generator_source_rows.size)
    for position, generator_source_row in enumerate(master.index.generator_source_rows):
        column = master.index.commitment_by_generator[int(generator_source_row)]
        expected_lower = 1.0 if masks.fixed_on[position] else 0.0
        expected_upper = 0.0 if masks.fixed_off[position] else 1.0
        if (
            float(master.canonical.column_lower[column]) != expected_lower
            or float(master.canonical.column_upper[column]) != expected_upper
        ):
            raise ScopfError("Prepared region master commitment bounds changed")
    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64)
    expected_cut_ids = {cut.cut_id for cut in commitment_cuts}
    if len(expected_cut_ids) != len(commitment_cuts):
        raise ScopfError("Prepared region contains duplicate commitment cuts")
    row_by_name = {name: row for row, name in enumerate(master.canonical.row_names)}
    for cut in commitment_cuts:
        cut.validate(source_rows.size)
        row = row_by_name.get(cut.cut_id)
        if row is None:
            raise ScopfError("Prepared region master omits a commitment cut")
        indices, values = master.canonical.row_entries(row)
        expected = {
            master.index.commitment_by_generator[int(source_rows[position])]: float(value)
            for position, value in enumerate(cut.coefficients)
            if value != 0.0
        }
        observed = dict(zip(indices, values, strict=True))
        if observed != expected or float(master.canonical.row_upper[row]) != float(cut.rhs):
            raise ScopfError("Prepared region commitment cut changed")


def _region_pmin_pmax_capacity_gate(
    *,
    case: Any,
    master: ReducedMaster,
    masks: RegionMasks,
    tolerance_pu: float,
) -> dict[str, Any]:
    """Check the exact aggregate dispatch interval implied by region masks."""

    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64)
    masks.validate(source_rows.size)
    pmin = np.asarray(case.gen[source_rows, PMIN], dtype=np.float64)
    pmax = np.asarray(case.gen[source_rows, PMAX], dtype=np.float64)
    if not np.all(np.isfinite(pmin)) or not np.all(np.isfinite(pmax)):
        raise ScopfError("Region PMIN/PMAX capacity gate found nonfinite source data")
    if np.any(pmin > pmax):
        raise ScopfError("Region PMIN/PMAX capacity gate found PMIN above PMAX")
    free = ~(masks.fixed_off | masks.fixed_on)
    minimum_by_generator = np.zeros(source_rows.size, dtype=np.float64)
    maximum_by_generator = np.zeros(source_rows.size, dtype=np.float64)
    minimum_by_generator[masks.fixed_on] = pmin[masks.fixed_on]
    maximum_by_generator[masks.fixed_on] = pmax[masks.fixed_on]
    minimum_by_generator[free] = np.minimum(0.0, pmin[free])
    maximum_by_generator[free] = np.maximum(0.0, pmax[free])
    minimum_dispatch = float(np.sum(minimum_by_generator))
    maximum_dispatch = float(np.sum(maximum_by_generator))
    demand = float(master.operator.total_demand_mw)
    tolerance_mw = float(tolerance_pu) * float(case.base_mva)
    shortfall = max(0.0, minimum_dispatch - demand, demand - maximum_dispatch)
    return {
        "kind": "exact_aggregate_region_pmin_pmax_interval_v1",
        "passes": bool(shortfall <= tolerance_mw),
        "demand_mw": demand,
        "minimum_dispatch_mw": minimum_dispatch,
        "maximum_dispatch_mw": maximum_dispatch,
        "capacity_shortfall_mw": shortfall,
        "registered_tolerance_pu": float(tolerance_pu),
        "registered_tolerance_mw": tolerance_mw,
        "fixed_off_count": int(np.count_nonzero(masks.fixed_off)),
        "fixed_on_count": int(np.count_nonzero(masks.fixed_on)),
        "free_count": int(np.count_nonzero(free)),
        "exact_source_pmin_changed": False,
    }


def _solve_region(
    *,
    region_id: str,
    masks: RegionMasks,
    case: Any,
    network: NetworkData,
    catalog: ContingencyCatalog,
    config: RunConfig,
    deadline: Deadline,
    initial_pairs: tuple[SecurityPair, ...],
    screener: ContingencyScreener,
    checkpoint: Callable[[], None],
    progress: Callable[[dict[str, Any]], None] | None = None,
    candidate_policy: PrimalCandidatePolicy | None = None,
    prepared_master: ReducedMaster | None = None,
    initial_native_primal: np.ndarray | None = None,
    initial_native_row_dual: np.ndarray | None = None,
    initial_warm_start_origin: str | None = None,
    concurrent_solver_context: bool = False,
    commitment_cuts: tuple[CommitmentUpperCut, ...] = (),
) -> SolvedRegion:
    region_started = time.perf_counter()
    profile = config.raw["platforms"]["dgx_spark"]
    master = prepared_master or _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=initial_pairs,
        commitment_cuts=commitment_cuts,
    )
    _validate_prepared_region_master(master, masks, initial_pairs, commitment_cuts)
    commitment_cut_row_by_id = {
        cut.cut_id: master.canonical.row_names.index(cut.cut_id)
        for cut in commitment_cuts
    }
    pairs_by_id = {pair.pair_id: pair for pair in initial_pairs}
    native_primal = (
        None
        if initial_native_primal is None
        else np.asarray(initial_native_primal, dtype=np.float64).copy()
    )
    native_dual = (
        None
        if initial_native_row_dual is None
        else np.asarray(initial_native_row_dual, dtype=np.float64).copy()
    )
    pdlp_warm_start_data: Any | None = None
    if native_primal is not None and native_primal.shape != (master.canonical.num_columns,):
        raise ScopfError("Prepared region native-primal warm start has the wrong shape")
    if native_dual is not None and (
        native_dual.ndim != 1 or native_dual.size > len(_native_constraint_layout(master.canonical))
    ):
        raise ScopfError("Prepared region native-dual warm start has the wrong shape")
    if (native_primal is not None or native_dual is not None) and not initial_warm_start_origin:
        raise ScopfError("Prepared region warm start lacks an origin")
    rounds: list[dict[str, Any]] = []
    final_screen: dict[str, Any] | None = None
    last_solve: ContinuousSolveResult | None = None
    usable_primal_seen = False
    infeasible_residuals: list[float] = []
    cold_restarts_remaining = (
        candidate_policy.cold_restart_attempts if candidate_policy is not None else 0
    )
    cold_restart_active = False

    def reject_attempt(reason: str, detail: str) -> None:
        raise RegionAttemptRejected(
            detail,
            reason=reason,
            master=master,
            security_pairs=tuple(sorted(pairs_by_id.values())),
            rounds=rounds,
        )

    def emit_progress() -> None:
        region_elapsed = time.perf_counter() - region_started
        if progress is not None:
            progress(
                {
                    "region_id": region_id,
                    "rows": master.canonical.num_rows,
                    "security_pair_count": len(pairs_by_id),
                    "constraint_generation_rounds": rounds,
                    "usable_primal_seen": usable_primal_seen,
                    "region_elapsed_seconds": region_elapsed,
                    "candidate_policy": (
                        candidate_policy.as_dict() if candidate_policy is not None else None
                    ),
                    "initial_warm_start_origin": initial_warm_start_origin,
                }
            )
        else:
            checkpoint()

    maximum_rounds = int(config.runtime["maximum_constraint_generation_rounds"])
    for round_number in range(1, maximum_rounds + 1):
        deadline.require(f"PDLP region {region_id} round {round_number}")
        solver_budget = min(
            deadline.solver_budget(), float(config.runtime["maximum_pdlp_round_seconds"])
        )
        if candidate_policy is not None:
            candidate_remaining = candidate_policy.total_seconds - (
                time.perf_counter() - region_started
            )
            if candidate_remaining < candidate_policy.minimum_round_seconds:
                reject_attempt(
                    "attempt_budget_exhausted",
                    f"Region {region_id} exhausted its {candidate_policy.total_seconds:g}s "
                    "bounded PDLP-attempt budget",
                )
            solver_budget = min(
                solver_budget,
                candidate_policy.maximum_round_seconds,
                candidate_remaining,
            )
        emit_progress()
        started = time.perf_counter()
        last_solve = solve_cuopt_continuous_pdlp(
            master.canonical,
            time_limit_seconds=solver_budget,
            optimality_tolerance=float(profile["pdlp_optimality_tolerance"]),
            primal_feasibility_tolerance=float(config.model["model_residual_tolerance_pu"]),
            certificate_residual_tolerance=float(profile["dual_certificate_residual_tolerance"]),
            native_scaling_mode=str(profile["native_scaling_mode"]),
            native_base_mva=float(case.base_mva),
            log_to_console=True,
            per_constraint_residual=bool(profile["per_constraint_residual"]),
            presolve=int(profile["presolve"]),
            initial_native_primal=(
                None if pdlp_warm_start_data is not None else native_primal
            ),
            initial_native_row_dual=(
                None if pdlp_warm_start_data is not None else native_dual
            ),
            initial_pdlp_warm_start_data=pdlp_warm_start_data,
            pdlp_solver_mode=int(profile.get("pdlp_solver_mode_native", 4)),
            concurrent_solver_context=concurrent_solver_context,
        )
        adapter_wall = time.perf_counter() - started
        round_record: dict[str, Any] = {
            "round": round_number,
            "rows_before_solve": master.canonical.num_rows,
            "security_pairs_before_solve": len(pairs_by_id),
            "solver_budget_seconds": solver_budget,
            "optimality_tolerance": float(profile["pdlp_optimality_tolerance"]),
            "adapter_wall_time_seconds": adapter_wall,
            "solve": _solve_summary(last_solve),
            "cold_restart": cold_restart_active,
            "concurrent_solver_context": concurrent_solver_context,
        }
        if round_number == 1 and initial_warm_start_origin is not None:
            round_record["initial_warm_start_origin"] = initial_warm_start_origin
        rounds.append(round_record)
        emit_progress()
        error_status = str(last_solve.statistics.get("error_status", ""))
        if (
            error_status != "Success"
            or last_solve.values is None
            or last_solve.native_row_dual is None
        ):
            if (
                candidate_policy is not None
                and error_status == "Success"
                and last_solve.status in {"Infeasible", "TimeLimit"}
            ):
                if cold_restarts_remaining > 0 and not cold_restart_active:
                    cold_restarts_remaining -= 1
                    native_primal = None
                    native_dual = None
                    pdlp_warm_start_data = None
                    infeasible_residuals.clear()
                    cold_restart_active = True
                    round_record["candidate_gate"] = {
                        "rejected": False,
                        "reason": "cold_restart_scheduled_after_missing_vectors",
                        "remaining_cold_restarts": cold_restarts_remaining,
                    }
                    emit_progress()
                    continue
                round_record["candidate_gate"] = {
                    "rejected": True,
                    "reason": (
                        "cold_restart_returned_no_usable_primal_vectors"
                        if cold_restart_active
                        else "solver_returned_no_usable_primal_vectors"
                    ),
                    "solve_status": last_solve.status,
                }
                emit_progress()
                reject_attempt(
                    str(round_record["candidate_gate"]["reason"]),
                    f"Region {region_id} rejected candidate after round "
                    f"{round_number}: solver returned no usable primal vectors "
                    f"with status={last_solve.status}",
                )
            raise ScopfError(
                f"Region {region_id} PDLP did not return usable vectors: "
                f"status={last_solve.status}, error={error_status}"
            )
        native_primal = last_solve.native_primal
        native_dual = last_solve.native_row_dual
        pdlp_warm_start_data = last_solve.pdlp_warm_start_data
        dual_certificate = last_solve.statistics.get("dual_certificate", {})
        certificate_primal_feasible = bool(dual_certificate.get("primal_feasible", False))
        canonical_residual_pu = (
            master.canonical.max_row_violation(last_solve.values) / case.base_mva
        )
        canonical_primal_feasible = canonical_residual_pu <= float(
            config.model["model_residual_tolerance_pu"]
        )
        worst_row = _worst_canonical_row_violation(
            master.canonical,
            last_solve.values,
            base_mva=float(case.base_mva),
            native_scaling_mode=str(profile["native_scaling_mode"]),
        )
        round_record["primal_acceptance"] = {
            "numeric_certificate_primal_feasible": certificate_primal_feasible,
            "canonical_model_residual_pu": canonical_residual_pu,
            "canonical_model_residual_passed": canonical_primal_feasible,
            "worst_canonical_row": worst_row,
        }
        refinement_attempts = int(
            config.runtime.get("root_canonical_residual_refinement_attempts", 0)
        )
        if (
            candidate_policy is None
            and last_solve.status == "Optimal"
            and not canonical_primal_feasible
            and refinement_attempts > 0
        ):
            refinement_tolerance = float(
                config.runtime[
                    "root_canonical_residual_refinement_optimality_tolerance"
                ]
            )
            refinement_records: list[dict[str, Any]] = []
            refinement_payload: dict[str, Any] = {
                "configured_attempts": refinement_attempts,
                "target_optimality_tolerance": refinement_tolerance,
                "attempts": refinement_records,
                "accepted": False,
            }
            round_record["canonical_residual_refinement"] = refinement_payload
            round_record["pre_refinement_solve"] = round_record["solve"]
            round_record["pre_refinement_primal_acceptance"] = round_record[
                "primal_acceptance"
            ]
            for refinement_attempt in range(1, refinement_attempts + 1):
                deadline.require(
                    f"PDLP region {region_id} round {round_number} canonical refinement"
                )
                refinement_budget = min(
                    deadline.solver_budget(),
                    float(config.runtime["maximum_pdlp_round_seconds"]),
                )
                refinement_started = time.perf_counter()
                refined = solve_cuopt_continuous_pdlp(
                    master.canonical,
                    time_limit_seconds=refinement_budget,
                    optimality_tolerance=refinement_tolerance,
                    primal_feasibility_tolerance=float(
                        config.model["model_residual_tolerance_pu"]
                    ),
                    certificate_residual_tolerance=float(
                        profile["dual_certificate_residual_tolerance"]
                    ),
                    native_scaling_mode=str(profile["native_scaling_mode"]),
                    native_base_mva=float(case.base_mva),
                    log_to_console=True,
                    per_constraint_residual=bool(profile["per_constraint_residual"]),
                    presolve=int(profile["presolve"]),
                    initial_native_primal=(
                        None if pdlp_warm_start_data is not None else native_primal
                    ),
                    initial_native_row_dual=(
                        None if pdlp_warm_start_data is not None else native_dual
                    ),
                    initial_pdlp_warm_start_data=pdlp_warm_start_data,
                    pdlp_solver_mode=int(profile.get("pdlp_solver_mode_native", 4)),
                    concurrent_solver_context=concurrent_solver_context,
                )
                refinement_wall = time.perf_counter() - refinement_started
                refinement_record: dict[str, Any] = {
                    "attempt": refinement_attempt,
                    "solver_budget_seconds": refinement_budget,
                    "optimality_tolerance": refinement_tolerance,
                    "adapter_wall_time_seconds": refinement_wall,
                    "warm_start_origin": "same_master_prior_pdlp_primal_and_dual",
                    "solve": _solve_summary(refined),
                }
                refinement_records.append(refinement_record)
                round_record["adapter_wall_time_seconds"] += refinement_wall
                refined_error_status = str(refined.statistics.get("error_status", ""))
                if (
                    refined_error_status != "Success"
                    or refined.values is None
                    or refined.native_row_dual is None
                ):
                    refinement_record["accepted"] = False
                    refinement_record["rejection_reason"] = "solver_returned_no_usable_vectors"
                    break
                last_solve = refined
                native_primal = refined.native_primal
                native_dual = refined.native_row_dual
                pdlp_warm_start_data = refined.pdlp_warm_start_data
                dual_certificate = refined.statistics.get("dual_certificate", {})
                certificate_primal_feasible = bool(
                    dual_certificate.get("primal_feasible", False)
                )
                canonical_residual_pu = (
                    master.canonical.max_row_violation(refined.values) / case.base_mva
                )
                canonical_primal_feasible = canonical_residual_pu <= float(
                    config.model["model_residual_tolerance_pu"]
                )
                worst_row = _worst_canonical_row_violation(
                    master.canonical,
                    refined.values,
                    base_mva=float(case.base_mva),
                    native_scaling_mode=str(profile["native_scaling_mode"]),
                )
                refinement_record["primal_acceptance"] = {
                    "numeric_certificate_primal_feasible": certificate_primal_feasible,
                    "canonical_model_residual_pu": canonical_residual_pu,
                    "canonical_model_residual_passed": canonical_primal_feasible,
                    "worst_canonical_row": worst_row,
                }
                refinement_record["accepted"] = bool(
                    certificate_primal_feasible and canonical_primal_feasible
                )
                refinement_payload["accepted"] = refinement_record["accepted"]
                round_record["solve"] = _solve_summary(last_solve)
                round_record["primal_acceptance"] = refinement_record[
                    "primal_acceptance"
                ]
                emit_progress()
                if refinement_record["accepted"]:
                    break
        if not certificate_primal_feasible or not canonical_primal_feasible:
            infeasible_residuals.append(canonical_residual_pu)
            round_record["screen"] = {
                "skipped": True,
                "reason": "pdlp_primal_infeasible",
            }
            if candidate_policy is not None:
                primal_scale = max(
                    1.0,
                    abs(float(last_solve.primal_objective or 0.0)),
                )
                dual_magnitude = abs(float(last_solve.dual_objective or 0.0))
                dual_multiple = dual_magnitude / primal_scale
                gate: dict[str, Any] = {
                    "dual_objective_multiple_of_primal_scale": dual_multiple,
                    "dual_divergence_threshold": (candidate_policy.dual_divergence_multiple),
                    "residual_history_pu": list(infeasible_residuals),
                    "rejected": False,
                }
                rejection_reason: str | None = None
                if dual_multiple >= candidate_policy.dual_divergence_multiple:
                    rejection_reason = "dual_objective_divergence"
                window = candidate_policy.stagnation_window_rounds
                if rejection_reason is None and len(infeasible_residuals) >= window:
                    selected = infeasible_residuals[-window:]
                    baseline = max(
                        selected[0],
                        float(config.model["model_residual_tolerance_pu"]),
                    )
                    best_followup = min(selected[1:])
                    improvement = (selected[0] - best_followup) / baseline
                    gate["window_relative_residual_improvement"] = improvement
                    gate["minimum_required_relative_residual_improvement"] = (
                        candidate_policy.minimum_relative_residual_improvement
                    )
                    if improvement < candidate_policy.minimum_relative_residual_improvement:
                        rejection_reason = "primal_residual_stagnation"
                if rejection_reason is not None:
                    if cold_restarts_remaining > 0 and not cold_restart_active:
                        cold_restarts_remaining -= 1
                        native_primal = None
                        native_dual = None
                        pdlp_warm_start_data = None
                        infeasible_residuals.clear()
                        cold_restart_active = True
                        gate["cold_restart_scheduled"] = True
                        gate["trigger_reason"] = rejection_reason
                        gate["reason"] = "cold_restart_scheduled"
                        round_record["candidate_gate"] = gate
                        emit_progress()
                        continue
                    if cold_restart_active:
                        rejection_reason = f"cold_restart_failed_{rejection_reason}"
                    gate["rejected"] = True
                    gate["reason"] = rejection_reason
                    round_record["candidate_gate"] = gate
                    emit_progress()
                    reject_attempt(
                        rejection_reason,
                        f"Region {region_id} rejected candidate after round "
                        f"{round_number}: {rejection_reason}",
                    )
                round_record["candidate_gate"] = gate
            if last_solve.status == "TimeLimit" and native_primal is not None:
                if cold_restart_active:
                    round_record["candidate_gate"] = {
                        "rejected": True,
                        "reason": "cold_restart_primal_infeasible",
                        "canonical_model_residual_pu": canonical_residual_pu,
                    }
                    emit_progress()
                    reject_attempt(
                        "cold_restart_primal_infeasible",
                        f"Region {region_id} cold restart remained primal infeasible "
                        f"after round {round_number}",
                    )
                emit_progress()
                continue
            if candidate_policy is not None:
                round_record["candidate_gate"] = {
                    "rejected": True,
                    "reason": "solver_terminated_with_primal_infeasibility",
                    "solve_status": last_solve.status,
                    "canonical_model_residual_pu": canonical_residual_pu,
                }
                emit_progress()
                reject_attempt(
                    "solver_terminated_with_primal_infeasibility",
                    f"Region {region_id} rejected candidate after round "
                    f"{round_number}: status={last_solve.status}, "
                    f"residual_pu={canonical_residual_pu:.6e}",
                )
            raise ScopfError(
                f"Region {region_id} PDLP primal is unusable: "
                f"status={last_solve.status}, residual_pu={canonical_residual_pu:.6e}"
            )
        usable_primal_seen = True
        cold_restart_active = False
        dispatch = reduced_dispatch(master, last_solve.values)
        flow = master.operator.flows(dispatch)
        screen_started = time.perf_counter()
        screened = screener.screen(
            flow,
            tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
            already_added=set(pairs_by_id),
        )
        final_screen = {
            "wall_time_seconds": time.perf_counter() - screen_started,
            "evaluated_sides": screened.evaluated_pairs,
            "new_violated_pairs": len(screened.violations),
            "maximum_violation_pu": screened.maximum_violation_pu,
            "maximum_pair_id": screened.maximum_pair_id,
        }
        round_record["screen"] = final_screen
        emit_progress()
        if not screened.violations:
            if screened.maximum_violation_pu > float(
                config.model["security_violation_tolerance_pu"]
            ):
                raise ScopfError(f"Region {region_id} final screen residual exceeds tolerance")
            break
        add_reduced_security_pairs(master, network, screened.violations)
        # Appended rows change the native PDLP state dimension.  Preserve the
        # raw primal/dual vectors (the dual is safely zero-extended by row),
        # but never submit an incompatible full solver context.
        pdlp_warm_start_data = None
        pairs_by_id.update((pair.pair_id, pair) for pair in screened.violations)
        round_record["added_pair_ids"] = [pair.pair_id for pair in screened.violations]
        emit_progress()
    else:
        reason = (
            "without a primal-feasible PDLP vector"
            if not usable_primal_seen
            else "before a zero-violation exhaustive screen"
        )
        if candidate_policy is not None:
            reject_attempt(
                "constraint_generation_round_limit",
                f"Region {region_id} reached its constraint-generation limit {reason}",
            )
        raise ScopfError(f"Region {region_id} reached its constraint-generation limit {reason}")

    assert last_solve is not None and final_screen is not None
    if last_solve.native_row_dual is None or last_solve.values is None:
        raise ScopfError(f"Region {region_id} lost its final PDLP vectors")
    row_dual = canonical_row_duals(
        master,
        last_solve.native_row_dual,
        native_scaling_mode=str(profile["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    if last_solve.primal_objective is None:
        raise ScopfError(f"Region {region_id} lacks a finite LP primal target")
    gpu_started = time.perf_counter()
    initial_commitment_cut_dual = np.asarray(
        [row_dual[commitment_cut_row_by_id[cut.cut_id]] for cut in commitment_cuts],
        dtype=np.float64,
    )
    coupling_row_scales: np.ndarray | None = None
    commitment_cut_scales: np.ndarray | None = None
    if bool(config.runtime.get("lagrangian_diagonal_preconditioning", False)):
        _column_scale, lagrangian_row_scale = native_scaling_vectors(
            master.canonical,
            mode=str(profile["native_scaling_mode"]),
            base_mva=float(case.base_mva),
        )
        coupling_row_scales = np.asarray(
            [
                lagrangian_row_scale[row.row_index]
                for row in sorted(master.coupling_rows, key=lambda row: row.row_name)
            ],
            dtype=np.float64,
        )
        commitment_cut_scales = np.asarray(
            [
                lagrangian_row_scale[commitment_cut_row_by_id[cut.cut_id]]
                for cut in commitment_cuts
            ],
            dtype=np.float64,
        )
    polished_row_dual, gpu_evaluation = optimize_lagrangian_bound_cupy(
        master,
        row_dual,
        masks,
        relaxation_primal_objective=float(last_solve.primal_objective),
        iterations=int(profile["lagrangian_gpu_iterations"]),
        polyak_fraction=float(profile["lagrangian_polyak_fraction"]),
        commitment_cuts=commitment_cuts,
        initial_commitment_cut_dual=initial_commitment_cut_dual,
        coupling_row_scales=coupling_row_scales,
        commitment_cut_scales=commitment_cut_scales,
    )
    gpu_evaluation["wall_time_seconds"] = time.perf_counter() - gpu_started
    best_commitment_cut_dual = np.asarray(
        gpu_evaluation.get(
            "best_commitment_cut_dual",
            np.zeros(len(commitment_cuts), dtype=np.float64),
        ),
        dtype=np.float64,
    )
    if best_commitment_cut_dual.shape != (len(commitment_cuts),) or not np.all(
        np.isfinite(best_commitment_cut_dual)
    ):
        raise ScopfError(
            f"Region {region_id} GPU Lagrangian cut-dual result is invalid"
        )
    gpu_evaluation["best_commitment_cut_dual"] = best_commitment_cut_dual
    replay_started = time.perf_counter()
    evaluation = evaluate_lagrangian_bound(
        master,
        polished_row_dual,
        masks,
        safety_margin_dollars=float(config.raw["benchmark"]["certificate_safety_margin_dollars"]),
        commitment_cuts=commitment_cuts,
        commitment_cut_dual=best_commitment_cut_dual,
    )
    gpu_evaluation["independent_cpu_replay_wall_time_seconds"] = (
        time.perf_counter() - replay_started
    )
    replay_difference = abs(
        float(gpu_evaluation["best_raw_lower_bound"]) - evaluation.raw_lower_bound
    )
    if replay_difference > float(config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]):
        raise ScopfError(
            f"Region {region_id} GPU/CPU Lagrangian replay differs by {replay_difference}"
        )
    gpu_evaluation["cpu_replay_difference_dollars"] = replay_difference
    return SolvedRegion(
        region_id=region_id,
        masks=masks,
        master=master,
        solve=last_solve,
        canonical_row_dual=row_dual,
        lagrangian=evaluation,
        commitment=commitment_vector(master, last_solve.values),
        security_pairs=tuple(sorted(pairs_by_id.values())),
        rounds=rounds,
        final_screen=final_screen,
        gpu_lagrangian=gpu_evaluation,
        commitment_cuts=commitment_cuts,
        commitment_cut_row_by_id=commitment_cut_row_by_id,
    )


def _solve_fixed_commitment_feasibility(
    *,
    region_id: str,
    commitment: np.ndarray,
    case: Any,
    network: NetworkData,
    catalog: ContingencyCatalog,
    config: RunConfig,
    deadline: Deadline,
    initial_pairs: tuple[SecurityPair, ...],
    screener: ContingencyScreener,
    checkpoint: Callable[[], None],
    progress: Callable[[dict[str, Any]], None] | None,
    policy: PrimalCandidatePolicy,
) -> FixedCommitmentFeasibilityResult:
    """Find and exhaustively screen a fixed-u dispatch before cost polishing.

    Phase I is solved on the exact dispatch-only projection.  This avoids
    asking PDLP to discover feasibility through thousands of already fixed
    commitment and local PWL variables.  A zero-violation dispatch is lifted
    back into the original ten-segment source model before it is accepted.
    """

    attempt_started = time.perf_counter()
    profile = config.raw["platforms"]["dgx_spark"]
    binary = np.asarray(commitment, dtype=np.int8)
    masks = RegionMasks(binary == 0, binary == 1)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=initial_pairs,
    )
    _validate_prepared_region_master(master, masks, initial_pairs)
    pairs_by_id = {pair.pair_id: pair for pair in initial_pairs}
    rounds: list[dict[str, Any]] = []
    native_phase_primal: np.ndarray | None = None
    native_phase_dual: np.ndarray | None = None
    pdlp_warm_start_data: Any | None = None

    def emit_progress() -> None:
        if progress is not None:
            progress(
                {
                    "region_id": region_id,
                    "policy": "fixed_commitment_dispatch_projection_phase_one_v1",
                    "security_pair_count": len(pairs_by_id),
                    "constraint_generation_rounds": rounds,
                    "elapsed_seconds": time.perf_counter() - attempt_started,
                    "candidate_policy": policy.as_dict(),
                }
            )
        else:
            checkpoint()

    def reject(
        reason: str,
        detail: str,
        *,
        commitment_feasibility_cut: CommitmentFeasibilityCut | None = None,
        commitment_feasibility_cut_record: dict[str, Any] | None = None,
    ) -> None:
        raise RegionAttemptRejected(
            detail,
            reason=reason,
            master=master,
            security_pairs=tuple(sorted(pairs_by_id.values())),
            rounds=rounds,
            commitment_feasibility_cut=commitment_feasibility_cut,
            commitment_feasibility_cut_record=commitment_feasibility_cut_record,
        )

    maximum_rounds = int(config.runtime["maximum_constraint_generation_rounds"])
    tolerance_pu = float(config.model["model_residual_tolerance_pu"])
    tolerance_mw = tolerance_pu * float(case.base_mva)
    secure_seed_tolerance_fraction = (
        float(config.runtime["fixed_commitment_secure_seed_tolerance_fraction"])
        if config.benchmark_id in ACTIVSG2000_V8_PLUS_EXPERIMENT_IDS
        else 1.0
    )
    if not 0.0 < secure_seed_tolerance_fraction <= 1.0:
        raise ScopfError(
            "Fixed-commitment secure-seed tolerance fraction must be in (0, 1]"
        )
    secure_seed_tolerance_pu = tolerance_pu * secure_seed_tolerance_fraction
    secure_seed_tolerance_mw = tolerance_mw * secure_seed_tolerance_fraction
    for constraint_round in range(1, maximum_rounds + 1):
        deadline.require(f"fixed-commitment projected Phase I {region_id} round {constraint_round}")
        try:
            projection: FixedCommitmentProjection = build_fixed_commitment_projection(
                master, binary
            )
        except FixedCommitmentProjectionInfeasible as exc:
            rounds.append(
                {
                    "round": constraint_round,
                    "security_pairs_before_solve": len(pairs_by_id),
                    "projection_precheck": {
                        "status": "candidate_infeasible_constant_coupling_row",
                        "row_name": exc.row_name,
                        "shifted_row_lower": exc.shifted_lower,
                        "shifted_row_upper": exc.shifted_upper,
                        "pruning_scope": "fixed_commitment_candidate_only",
                        "full_model_infeasibility_claimed": False,
                    },
                }
            )
            emit_progress()
            reject(
                "projected_constant_coupling_row_violation",
                f"Candidate {region_id} is infeasible because fixed dispatch "
                f"violates constant coupling row {exc.row_name}",
            )
        phase_model = build_phase_one_model(
            projection.canonical,
            base_mva=float(case.base_mva),
            maximum_violation_pu=float(config.runtime["phase_one_maximum_violation_pu"]),
        )
        if native_phase_primal is not None and native_phase_primal.shape != (
            phase_model.num_columns,
        ):
            native_phase_primal = None
            pdlp_warm_start_data = None
        round_record: dict[str, Any] = {
            "round": constraint_round,
            "security_pairs_before_solve": len(pairs_by_id),
            "projection": dict(projection.audit),
            "adapter_wall_time_seconds": 0.0,
            "phase_one_model": {
                "columns": phase_model.num_columns,
                "rows": phase_model.num_rows,
            },
            "solve_attempts": [],
        }
        rounds.append(round_record)
        emit_progress()
        residual_history: list[float] = []
        cold_restarts_remaining = int(policy.cold_restart_attempts)
        cold_restart_active = False
        projected_values: np.ndarray | None = None
        source_values: np.ndarray | None = None

        while projected_values is None:
            candidate_remaining = policy.total_seconds - (time.perf_counter() - attempt_started)
            if candidate_remaining < policy.minimum_round_seconds:
                reject(
                    "projected_phase_one_budget_exhausted",
                    f"Candidate {region_id} exhausted its {policy.total_seconds:g}s "
                    "dispatch-projection Phase-I budget",
                )
            deadline.require(f"projected Phase-I solve for {region_id}")
            solver_budget = min(
                deadline.solver_budget(),
                policy.maximum_round_seconds,
                candidate_remaining,
            )
            solve_started = time.perf_counter()
            solve = solve_cuopt_continuous_pdlp(
                phase_model,
                time_limit_seconds=solver_budget,
                optimality_tolerance=float(profile["pdlp_optimality_tolerance"]),
                primal_feasibility_tolerance=secure_seed_tolerance_pu,
                certificate_residual_tolerance=float(
                    profile["dual_certificate_residual_tolerance"]
                ),
                native_scaling_mode=str(profile["native_scaling_mode"]),
                native_base_mva=float(case.base_mva),
                log_to_console=True,
                per_constraint_residual=bool(profile["per_constraint_residual"]),
                presolve=int(profile["presolve"]),
                initial_native_primal=(
                    None if pdlp_warm_start_data is not None else native_phase_primal
                ),
                initial_native_row_dual=(
                    None if pdlp_warm_start_data is not None else native_phase_dual
                ),
                initial_pdlp_warm_start_data=pdlp_warm_start_data,
                pdlp_solver_mode=int(profile.get("pdlp_solver_mode_native", 4)),
            )
            solve_record: dict[str, Any] = {
                "attempt": len(round_record["solve_attempts"]) + 1,
                "solver_budget_seconds": solver_budget,
                "adapter_wall_time_seconds": time.perf_counter() - solve_started,
                "cold_restart": cold_restart_active,
                "solve": _solve_summary(solve),
                "error_status": solve.statistics.get("error_status"),
                "numerical_tolerances": {
                    "official_model_residual_tolerance_pu": tolerance_pu,
                    "secure_seed_fraction": secure_seed_tolerance_fraction,
                    "requested_solver_and_lift_tolerance_pu": (
                        secure_seed_tolerance_pu
                    ),
                    "mathematical_feasible_set_changed": False,
                },
            }
            round_record["solve_attempts"].append(solve_record)
            round_record["adapter_wall_time_seconds"] = float(
                round_record["adapter_wall_time_seconds"]
            ) + float(solve_record["adapter_wall_time_seconds"])
            emit_progress()
            error_status = str(solve.statistics.get("error_status", ""))
            vectors_available = bool(
                error_status == "Success"
                and solve.values is not None
                and solve.native_primal is not None
                and np.asarray(solve.values).shape == (phase_model.num_columns,)
                and np.all(np.isfinite(solve.values))
            )
            if not vectors_available:
                if cold_restarts_remaining > 0 and not cold_restart_active:
                    cold_restarts_remaining -= 1
                    native_phase_primal = None
                    native_phase_dual = None
                    pdlp_warm_start_data = None
                    residual_history.clear()
                    cold_restart_active = True
                    solve_record["continuation"] = "cold_restart_after_missing_vectors"
                    emit_progress()
                    continue
                reject(
                    "projected_phase_one_missing_vectors",
                    f"Candidate {region_id} projected Phase I returned no usable "
                    f"vectors: status={solve.status}, error={error_status}",
                )

            phase_values = np.asarray(solve.values, dtype=np.float64)
            native_phase_primal = np.asarray(solve.native_primal, dtype=np.float64)
            pdlp_warm_start_data = solve.pdlp_warm_start_data
            if solve.native_row_dual is not None:
                native_phase_dual = np.asarray(solve.native_row_dual, dtype=np.float64).copy()
            candidate_projected = phase_values[:-1]
            phase_violation_pu = max(0.0, float(phase_values[-1]))
            lift_validation = projection.validate_lift(
                candidate_projected, tolerance_mw=secure_seed_tolerance_mw
            )
            numeric_primal_feasible = bool(
                solve.statistics.get("dual_certificate", {}).get("primal_feasible", False)
            )
            accepted = bool(
                numeric_primal_feasible
                and phase_violation_pu <= secure_seed_tolerance_pu
                and lift_validation["passed"]
            )
            metric_pu = max(
                phase_violation_pu,
                float(lift_validation["projected_maximum_violation_mw"]) / float(case.base_mva),
                float(lift_validation["lifted_source_maximum_violation_mw"]) / float(case.base_mva),
            )
            solve_record["primal_acceptance"] = {
                "accepted": accepted,
                "numeric_certificate_primal_feasible": numeric_primal_feasible,
                "phase_one_violation_pu": phase_violation_pu,
                "maximum_combined_residual_pu": metric_pu,
                **lift_validation,
            }
            if accepted:
                projected_values = candidate_projected.copy()
                source_values = projection.lift(projected_values)
                break

            if (
                solve.status == "Optimal"
                and phase_violation_pu > tolerance_pu
                and solve.native_row_dual is not None
                and np.asarray(solve.native_row_dual).shape == (phase_model.num_rows,)
            ):
                _phase_column_scale, phase_row_scale = native_scaling_vectors(
                    phase_model,
                    mode=str(profile["native_scaling_mode"]),
                    base_mva=float(case.base_mva),
                )
                canonical_phase_dual = (
                    np.asarray(solve.native_row_dual, dtype=np.float64)
                    * phase_row_scale
                )
                certificate = phase_one_certificate(
                    phase_model,
                    canonical_phase_dual,
                    safety_margin_pu=float(
                        config.runtime["phase_one_safety_margin_pu"]
                    ),
                    infeasibility_threshold_pu=float(
                        config.runtime["phase_one_infeasibility_threshold_pu"]
                    ),
                )
                replayed_certificate = replay_phase_one_certificate(
                    phase_model, certificate
                )
                solve_record["phase_one_infeasibility_certificate"] = certificate
                solve_record["phase_one_infeasibility_replay"] = replayed_certificate
                if bool(replayed_certificate["prune_certified"]):
                    cut, cut_audit = derive_commitment_feasibility_cut(
                        master=master,
                        phase_model=phase_model,
                        phase_certificate=certificate,
                        source_commitment=binary,
                        replay_tolerance_pu=float(
                            config.model["phase_one_replay_tolerance_pu"]
                        ),
                    )
                    cut_record = {
                        **cut_audit,
                        "source_commitment_generator_rows": (
                            master.index.generator_source_rows[binary == 1] + 1
                        ).tolist(),
                        "security_pairs": [
                            security_pair_record(pair)
                            for pair in sorted(pairs_by_id.values())
                        ],
                        "projection_audit": dict(projection.audit),
                    }
                    solve_record["commitment_feasibility_cut"] = cut_record
                    solve_record["continuation"] = (
                        "reject_without_cold_restart_after_replayed_positive_phase_one_dual"
                    )
                    emit_progress()
                    reject(
                        "projected_phase_one_positive_optimum_certified",
                        f"Candidate {region_id} has a replay-certified positive "
                        f"Phase-I optimum {phase_violation_pu:.6e} p.u.",
                        commitment_feasibility_cut=cut,
                        commitment_feasibility_cut_record=cut_record,
                    )

            residual_history.append(metric_pu)
            rejection_reason: str | None = None
            dual_magnitude = abs(float(solve.dual_objective or 0.0))
            primal_scale = max(1.0, abs(float(solve.primal_objective or 0.0)))
            solve_record["dual_divergence_ratio"] = dual_magnitude / primal_scale
            if dual_magnitude > policy.dual_divergence_multiple * primal_scale:
                rejection_reason = "projected_phase_one_dual_divergence"
            window = int(policy.stagnation_window_rounds)
            if len(residual_history) >= window + 1:
                selected = residual_history[-(window + 1) :]
                baseline = max(selected[0], secure_seed_tolerance_pu, 1e-30)
                improvement = (selected[0] - min(selected[1:])) / baseline
                solve_record["window_relative_residual_improvement"] = improvement
                if improvement < policy.minimum_relative_residual_improvement:
                    rejection_reason = "projected_phase_one_residual_stagnation"
            if solve.status == "Optimal" and phase_violation_pu > tolerance_pu:
                rejection_reason = "projected_phase_one_positive_optimum"
            if rejection_reason is not None:
                if cold_restarts_remaining > 0 and not cold_restart_active:
                    cold_restarts_remaining -= 1
                    native_phase_primal = None
                    native_phase_dual = None
                    pdlp_warm_start_data = None
                    residual_history.clear()
                    cold_restart_active = True
                    solve_record["continuation"] = f"cold_restart_after_{rejection_reason}"
                    emit_progress()
                    continue
                reject(
                    rejection_reason,
                    f"Candidate {region_id} projected Phase I rejected after "
                    f"residual {metric_pu:.6e} p.u.: {rejection_reason}",
                )
            solve_record["continuation"] = "warm_start_same_projected_phase_one"
            cold_restart_active = False
            emit_progress()

        assert projected_values is not None and source_values is not None
        source_native_row_dual: np.ndarray | None = None
        if (
            config.benchmark_id in ACTIVSG2000_V4_PLUS_EXPERIMENT_IDS
            and native_phase_dual is not None
        ):
            projected_native_dual, phase_mapping = _map_phase_one_dual_to_source_native(
                phase_model,
                projection.canonical,
                native_phase_dual,
                scaling_mode=str(profile["native_scaling_mode"]),
                base_mva=float(case.base_mva),
            )
            source_native_row_dual, source_mapping = _map_native_row_dual_by_identity(
                projection.canonical,
                master.canonical,
                projected_native_dual,
                scaling_mode=str(profile["native_scaling_mode"]),
                base_mva=float(case.base_mva),
            )
            round_record["cost_lp_dual_warm_start"] = {
                "eligible": True,
                "phase_to_projection": phase_mapping,
                "projection_to_source": source_mapping,
                "source_native_row_dual_count": int(source_native_row_dual.size),
            }
        dispatch = reduced_dispatch(master, source_values)
        flow = master.operator.flows(dispatch)
        screen_started = time.perf_counter()
        screened = screener.screen(
            flow,
            tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
            already_added=set(pairs_by_id),
        )
        final_screen = {
            "wall_time_seconds": time.perf_counter() - screen_started,
            "evaluated_sides": screened.evaluated_pairs,
            "new_violated_pairs": len(screened.violations),
            "maximum_violation_pu": screened.maximum_violation_pu,
            "maximum_pair_id": screened.maximum_pair_id,
        }
        round_record["screen"] = final_screen
        emit_progress()
        if not screened.violations:
            if screened.maximum_violation_pu > float(
                config.model["security_violation_tolerance_pu"]
            ):
                raise ScopfError(f"Candidate {region_id} exhaustive screen exceeds tolerance")
            return FixedCommitmentFeasibilityResult(
                master=master,
                source_values=source_values,
                source_native_row_dual=source_native_row_dual,
                security_pairs=tuple(sorted(pairs_by_id.values())),
                rounds=rounds,
                final_screen=final_screen,
            )
        add_reduced_security_pairs(master, network, screened.violations)
        # The rebuilt Phase-I model inserts new split rows, so neither its
        # full state nor the prior native row-dual order is compatible.
        pdlp_warm_start_data = None
        native_phase_dual = None
        pairs_by_id.update((pair.pair_id, pair) for pair in screened.violations)
        round_record["added_pair_ids"] = [pair.pair_id for pair in screened.violations]
        emit_progress()

    reject(
        "projected_phase_one_constraint_generation_round_limit",
        f"Candidate {region_id} reached the constraint-generation round limit",
    )
    raise AssertionError("unreachable fixed-commitment feasibility rejection")


def _solve_fixed_commitment_cost_projection(
    *,
    region_id: str,
    commitment: np.ndarray,
    case: Any,
    network: NetworkData,
    catalog: ContingencyCatalog,
    config: RunConfig,
    deadline: Deadline,
    prepared_master: ReducedMaster,
    initial_source_values: np.ndarray,
    initial_pairs: tuple[SecurityPair, ...],
    screener: ContingencyScreener,
    checkpoint: Callable[[], None],
    progress: Callable[[dict[str, Any]], None] | None,
    policy: PrimalCandidatePolicy,
) -> FixedCommitmentCostResult:
    """Polish a secure fixed commitment and repair PDLP numerical noise.

    A single bounded PDLP solve targets production cost in the exact convex
    PWL epigraph.  Its dispatch is then projected onto PMIN/PMAX and exact
    balance on the selected array backend.  Any newly exposed N-1 rows are
    inserted, and the largest feasible point on the segment from the already
    verified secure dispatch to that target is computed on the GPU.  This
    avoids treating a time-limit PDLP residual as physical infeasibility and
    avoids a second factorization or opaque Stable2-state submission.
    """

    if config.benchmark_id not in {
        ACTIVSG2000_V7_EXPERIMENT_ID,
        ACTIVSG2000_V8_EXPERIMENT_ID,
        ACTIVSG2000_V9_EXPERIMENT_ID,
        ACTIVSG2000_V10_EXPERIMENT_ID,
        ACTIVSG2000_V11_EXPERIMENT_ID,
    }:
        raise ScopfError("The fixed-commitment cost projection is registered only for v7-v11")
    started = time.perf_counter()
    master = prepared_master
    binary = np.asarray(commitment, dtype=np.int8)
    pairs_by_id = {pair.pair_id: pair for pair in initial_pairs}
    source_seed = np.asarray(initial_source_values, dtype=np.float64).copy()
    if source_seed.shape != (master.canonical.num_columns,):
        raise ScopfError("Fixed-commitment cost projection source seed has the wrong shape")
    rounds: list[dict[str, Any]] = []

    def emit_progress() -> None:
        if progress is not None:
            progress(
                {
                    "region_id": region_id,
                    "policy": (
                        "single_pdlp_then_gpu_balance_and_convex_feasible_segment_v2"
                    ),
                    "security_pair_count": len(pairs_by_id),
                    "constraint_generation_rounds": rounds,
                    "elapsed_seconds": time.perf_counter() - started,
                    "candidate_policy": policy.as_dict(),
                }
            )
        else:
            checkpoint()

    def reject(reason: str, detail: str) -> None:
        raise RegionAttemptRejected(
            detail,
            reason=reason,
            master=master,
            security_pairs=tuple(sorted(pairs_by_id.values())),
            rounds=rounds,
        )

    profile = config.raw["platforms"]["dgx_spark"]
    tolerance_pu = float(config.model["model_residual_tolerance_pu"])
    tolerance_mw = tolerance_pu * float(case.base_mva)
    repair_tolerance_mw = 0.5 * tolerance_mw
    maximum_rounds = int(config.runtime["maximum_constraint_generation_rounds"])
    repair_backend = str(screener.backend)
    deadline.require(f"fixed-commitment projected cost {region_id}")
    solve_projection = build_fixed_commitment_projection(
        master,
        binary,
        include_cost_epigraph=True,
    )
    projected_seed = solve_projection.project_source_values(source_seed)
    seed_validation = solve_projection.validate_lift(
        projected_seed,
        tolerance_mw=tolerance_mw,
    )
    if not seed_validation["passed"]:
        raise ScopfError("Secure source seed did not survive exact cost projection")
    column_scale, _ = native_scaling_vectors(
        solve_projection.canonical,
        mode=str(profile["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    candidate_remaining = policy.total_seconds - (time.perf_counter() - started)
    if candidate_remaining < policy.minimum_round_seconds:
        reject(
            "projected_cost_budget_exhausted",
            f"Candidate {region_id} has no exact cost-projection solve budget",
        )
    solver_budget = min(
        deadline.solver_budget(),
        policy.maximum_round_seconds,
        candidate_remaining,
    )
    round_record: dict[str, Any] = {
        "round": 1,
        "security_pairs_before_solve": len(pairs_by_id),
        "projection": dict(solve_projection.audit),
        "seed": {
            "canonical_residual_pu": (
                solve_projection.canonical.max_row_violation(projected_seed)
                / float(case.base_mva)
            ),
            "lift_validation": seed_validation,
            "source_objective": float(
                np.asarray(master.canonical.objective) @ source_seed
            ),
        },
        "solve_attempts": [],
        "repair_rounds": [],
        "secure_seed_margin_refinements": [],
    }
    rounds.append(round_record)
    emit_progress()
    solve_started = time.perf_counter()
    solve = solve_cuopt_continuous_pdlp(
        solve_projection.canonical,
        time_limit_seconds=solver_budget,
        optimality_tolerance=float(profile["pdlp_optimality_tolerance"]),
        primal_feasibility_tolerance=tolerance_pu,
        certificate_residual_tolerance=float(
            profile["dual_certificate_residual_tolerance"]
        ),
        native_scaling_mode=str(profile["native_scaling_mode"]),
        native_base_mva=float(case.base_mva),
        log_to_console=True,
        per_constraint_residual=bool(profile["per_constraint_residual"]),
        presolve=int(profile["presolve"]),
        initial_native_primal=projected_seed / column_scale,
        initial_native_row_dual=None,
        initial_pdlp_warm_start_data=None,
        pdlp_solver_mode=int(profile["pdlp_solver_mode_native"]),
    )
    solve_record: dict[str, Any] = {
        "attempt": 1,
        "solver_budget_seconds": solver_budget,
        "adapter_wall_time_seconds": time.perf_counter() - solve_started,
        "solve": _solve_summary(solve),
        "continuation": "none_single_bounded_cost_solve",
    }
    round_record["solve_attempts"].append(solve_record)
    round_record["adapter_wall_time_seconds"] = float(
        solve_record["adapter_wall_time_seconds"]
    )
    emit_progress()
    if (
        str(solve.statistics.get("error_status", "")) != "Success"
        or solve.values is None
        or solve.native_primal is None
        or solve.native_row_dual is None
    ):
        reject(
            "projected_cost_missing_vectors",
            f"Candidate {region_id} projected cost LP returned no usable vectors: "
            f"status={solve.status}",
        )

    raw_projected = np.asarray(solve.values, dtype=np.float64)
    raw_source = solve_projection.lift(raw_projected)
    raw_metrics = {
        "numeric_certificate_primal_feasible": bool(
            solve.statistics.get("dual_certificate", {}).get(
                "primal_feasible", False
            )
        ),
        "projected_canonical_residual_pu": (
            solve_projection.canonical.max_row_violation(raw_projected)
            / float(case.base_mva)
        ),
        "lifted_source_residual_pu": (
            master.canonical.max_row_violation(raw_source) / float(case.base_mva)
        ),
        "projected_objective": float(
            np.asarray(solve_projection.canonical.objective) @ raw_projected
        ),
        "lifted_exact_pwl_objective": float(
            np.asarray(master.canonical.objective) @ raw_source
        ),
    }
    balanced_projected, balance_audit = solve_projection.rebalance_dispatch(
        raw_projected,
        total_demand_mw=float(master.operator.total_demand_mw),
        backend=repair_backend,
    )
    target_source = solve_projection.lift(balanced_projected)
    target_objective = float(np.asarray(master.canonical.objective) @ target_source)
    solve_record["raw_primal"] = raw_metrics
    solve_record["gpu_balance_projection"] = balance_audit
    solve_record["balanced_target"] = {
        "source_objective": target_objective,
        "source_canonical_residual_pu": (
            master.canonical.max_row_violation(target_source) / float(case.base_mva)
        ),
    }

    target_screen_started = time.perf_counter()
    target_screened = screener.screen(
        master.operator.flows(reduced_dispatch(master, target_source)),
        tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
        already_added=set(pairs_by_id),
    )
    target_screen = {
        "wall_time_seconds": time.perf_counter() - target_screen_started,
        "evaluated_sides": target_screened.evaluated_pairs,
        "new_violated_pairs": len(target_screened.violations),
        "maximum_violation_pu": target_screened.maximum_violation_pu,
        "maximum_pair_id": target_screened.maximum_pair_id,
        "purpose": "discover_all_target_endpoint_security_rows_before_segment_repair",
    }
    round_record["target_screen"] = target_screen
    if target_screened.violations:
        add_reduced_security_pairs(master, network, target_screened.violations)
        pairs_by_id.update((pair.pair_id, pair) for pair in target_screened.violations)
        round_record["target_added_pair_ids"] = [
            pair.pair_id for pair in target_screened.violations
        ]
    emit_progress()

    accepted_source_values: np.ndarray | None = None
    final_screen: dict[str, Any] | None = None
    final_projection: FixedCommitmentProjection | None = None
    final_repair_audit: dict[str, Any] | None = None
    for repair_round in range(1, maximum_rounds + 1):
        deadline.require(f"fixed-commitment numerical repair {region_id}")
        projection = build_fixed_commitment_projection(
            master,
            binary,
            include_cost_epigraph=True,
        )
        repair_seed = projection.project_source_values(source_seed)
        repair_target = projection.project_source_values(target_source)
        seed_recheck = projection.validate_lift(
            repair_seed,
            tolerance_mw=repair_tolerance_mw,
        )
        if not seed_recheck["passed"]:
            margin_record: dict[str, Any] = {
                "repair_round": repair_round,
                "security_pair_count": len(pairs_by_id),
                "pre_refinement_lift_validation": seed_recheck,
                "registered_repair_tolerance_pu": (
                    repair_tolerance_mw / float(case.base_mva)
                ),
                "expanded_security_rows": (
                    len(pairs_by_id)
                    - int(round_record["security_pairs_before_solve"])
                ),
            }
            round_record["secure_seed_margin_refinements"].append(margin_record)
            emit_progress()
            if config.benchmark_id not in ACTIVSG2000_V8_PLUS_EXPERIMENT_IDS:
                reject(
                    "projected_cost_secure_seed_lost_margin",
                    "Secure seed lacks the registered numerical-repair safety margin",
                )
            refinement_started = time.perf_counter()
            try:
                refined = _solve_fixed_commitment_feasibility(
                    region_id=f"{region_id}_margin_refinement_{repair_round}",
                    commitment=binary,
                    case=case,
                    network=network,
                    catalog=catalog,
                    config=config,
                    deadline=deadline,
                    initial_pairs=tuple(sorted(pairs_by_id.values())),
                    screener=screener,
                    checkpoint=checkpoint,
                    progress=None,
                    policy=policy,
                )
            except RegionAttemptRejected as exc:
                margin_record.update(
                    {
                        "status": "rejected",
                        "reason": exc.reason,
                        "error": str(exc),
                        "wall_time_seconds": time.perf_counter()
                        - refinement_started,
                        "constraint_generation_rounds": exc.rounds,
                    }
                )
                emit_progress()
                reject(
                    "projected_cost_secure_seed_margin_refinement_rejected",
                    "Expanded-row GPU Phase I could not restore the secure-seed "
                    f"numerical margin: {exc.reason}",
                )
            master = refined.master
            source_seed = refined.source_values.copy()
            pairs_by_id = {
                pair.pair_id: pair for pair in refined.security_pairs
            }
            margin_record.update(
                {
                    "status": "secure_reprojected",
                    "wall_time_seconds": time.perf_counter()
                    - refinement_started,
                    "constraint_generation_rounds": refined.rounds,
                    "final_screen": refined.final_screen,
                    "post_refinement_source_residual_pu": (
                        master.canonical.max_row_violation(source_seed)
                        / float(case.base_mva)
                    ),
                    "mathematical_feasible_set_changed": False,
                }
            )
            emit_progress()
            continue
        repaired_projected, segment_audit = repair_along_feasible_segment(
            projection.canonical,
            repair_seed,
            repair_target,
            tolerance=repair_tolerance_mw,
            backend=repair_backend,
        )
        repaired_source = projection.lift(repaired_projected)
        lift_validation = projection.validate_lift(
            repaired_projected,
            tolerance_mw=tolerance_mw,
        )
        source_residual_pu = (
            master.canonical.max_row_violation(repaired_source) / float(case.base_mva)
        )
        repair_record: dict[str, Any] = {
            "repair_round": repair_round,
            "security_pair_count": len(pairs_by_id),
            "projection": dict(projection.audit),
            "segment": segment_audit,
            "lift_validation": lift_validation,
            "source_canonical_residual_pu": source_residual_pu,
            "seed_source_objective": float(
                np.asarray(master.canonical.objective) @ source_seed
            ),
            "target_source_objective": target_objective,
            "repaired_source_objective": float(
                np.asarray(master.canonical.objective) @ repaired_source
            ),
        }
        round_record["repair_rounds"].append(repair_record)
        if not lift_validation["passed"] or source_residual_pu > tolerance_pu:
            reject(
                "projected_cost_segment_repair_residual",
                "GPU feasible-segment repair exceeded the canonical tolerance",
            )
        screen_started = time.perf_counter()
        screened = screener.screen(
            master.operator.flows(reduced_dispatch(master, repaired_source)),
            tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
            already_added=set(pairs_by_id),
        )
        final_screen = {
            "wall_time_seconds": time.perf_counter() - screen_started,
            "evaluated_sides": screened.evaluated_pairs,
            "new_violated_pairs": len(screened.violations),
            "maximum_violation_pu": screened.maximum_violation_pu,
            "maximum_pair_id": screened.maximum_pair_id,
        }
        repair_record["screen"] = final_screen
        emit_progress()
        if not screened.violations:
            accepted_source_values = repaired_source
            final_projection = projection
            final_repair_audit = segment_audit
            break
        add_reduced_security_pairs(master, network, screened.violations)
        pairs_by_id.update((pair.pair_id, pair) for pair in screened.violations)
        repair_record["added_pair_ids"] = [pair.pair_id for pair in screened.violations]

    if (
        accepted_source_values is None
        or final_screen is None
        or final_projection is None
        or final_repair_audit is None
    ):
        reject(
            "projected_cost_segment_repair_round_limit",
            f"Candidate {region_id} reached the numerical-repair round limit",
        )

    mapped_native_dual, mapping = _map_native_row_dual_by_identity(
        solve_projection.canonical,
        master.canonical,
        np.asarray(solve.native_row_dual, dtype=np.float64),
        scaling_mode=str(profile["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    round_record["source_dual_mapping"] = mapping
    dispatch_changed = bool(
        np.max(np.abs(accepted_source_values - raw_source)) > 1e-9
    )
    added_after_solve = len(pairs_by_id) > int(round_record["security_pairs_before_solve"])
    pricing_certified = bool(
        solve.optimal
        and solve.statistics.get("dual_certificate", {}).get("passed", False)
        and not dispatch_changed
        and not added_after_solve
        and float(final_repair_audit["step_fraction"]) == 1.0
    )
    pricing_audit = {
        "certified_for_returned_dispatch": pricing_certified,
        "source": "fixed_commitment_cost_pdlp_row_dual",
        "pdlp_termination_status": solve.status,
        "dispatch_changed_by_numerical_repair": dispatch_changed,
        "security_rows_added_after_solve": (
            len(pairs_by_id) - int(round_record["security_pairs_before_solve"])
        ),
        "source_rows_zero_filled_by_projection_mapping": int(
            mapping["zero_filled_native_constraint_count"]
        ),
        "interpretation": (
            "certified_fixed_commitment_lp_prices"
            if pricing_certified
            else "provisional_dual_candidate_not_complementary_to_repaired_dispatch"
        ),
    }
    round_record["pricing_audit"] = pricing_audit
    return FixedCommitmentCostResult(
        master=master,
        source_values=accepted_source_values,
        canonical_row_dual=canonical_row_duals(
            master,
            mapped_native_dual,
            native_scaling_mode=str(profile["native_scaling_mode"]),
            base_mva=float(case.base_mva),
        ),
        projected_solve=solve,
        security_pairs=tuple(sorted(pairs_by_id.values())),
        rounds=rounds,
        final_screen=final_screen,
        pricing_certified=pricing_certified,
        pricing_audit=pricing_audit,
    )


def _network_feasible_commitment_repairs(
    *,
    case: Any,
    master: ReducedMaster,
    commitment: np.ndarray,
    masks: RegionMasks,
    violated_row_name: str,
    demand_mw: float,
    on_values: np.ndarray,
    maximum_repairs: int,
    pair_search_limit: int,
    tolerance_mw: float,
) -> list[NetworkCommitmentRepair]:
    """Generate one/two-flip candidates whose exact row box contains its RHS.

    This is candidate generation only.  Every returned commitment must still
    pass the exact dispatch-projection Phase I and exhaustive contingency
    screen before it can become an incumbent.
    """

    if maximum_repairs < 1 or pair_search_limit < 2:
        raise ScopfError("Network commitment repair limits must be positive")
    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64)
    binary = np.asarray(commitment, dtype=np.int8)
    values = np.asarray(on_values, dtype=np.float64)
    masks.validate(source_rows.size)
    if binary.shape != (source_rows.size,) or np.any((binary != 0) & (binary != 1)):
        raise ScopfError("Network commitment repair requires a binary commitment")
    if values.shape != binary.shape or not np.all(np.isfinite(values)):
        raise ScopfError("Network commitment repair received invalid on-values")
    if np.any(binary[masks.fixed_off] != 0) or np.any(binary[masks.fixed_on] != 1):
        raise ScopfError("Network commitment repair candidate violates its region masks")

    matching = [row for row in master.coupling_rows if row.row_name == violated_row_name]
    if len(matching) != 1:
        raise ScopfError(
            f"Network commitment repair expected one coupling row {violated_row_name!r}"
        )
    coupling = matching[0]
    row_lower, row_upper = master.canonical.row_bound_arrays()
    lower = float(row_lower[coupling.row_index])
    upper = float(row_upper[coupling.row_index])
    coefficients = np.asarray(coupling.generator_coefficients, dtype=np.float64)
    pmin = np.asarray(case.gen[source_rows, PMIN], dtype=np.float64)
    pmax = np.asarray(case.gen[source_rows, PMAX], dtype=np.float64)
    if not (
        coefficients.shape == pmin.shape == pmax.shape == binary.shape
        and np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(pmin))
        and np.all(np.isfinite(pmax))
        and np.all(pmin <= pmax)
    ):
        raise ScopfError("Network commitment repair found invalid source row data")

    def row_interval(candidate: np.ndarray) -> tuple[float, float]:
        active = candidate == 1
        low_dispatch = np.where(coefficients >= 0.0, pmin, pmax)
        high_dispatch = np.where(coefficients >= 0.0, pmax, pmin)
        minimum = float(np.sum(coefficients[active] * low_dispatch[active]))
        maximum = float(np.sum(coefficients[active] * high_dispatch[active]))
        return (
            float(np.nextafter(minimum, -np.inf)),
            float(np.nextafter(maximum, np.inf)),
        )

    def row_shortfall(interval: tuple[float, float]) -> float:
        minimum, maximum = interval
        return float(max(0.0, lower - maximum, minimum - upper))

    def capacity_interval(candidate: np.ndarray) -> tuple[float, float, float]:
        minimum = float(pmin @ candidate)
        maximum = float(pmax @ candidate)
        shortfall = float(max(0.0, minimum - demand_mw, demand_mw - maximum))
        return minimum, maximum, shortfall

    base_interval = row_interval(binary)
    base_shortfall = row_shortfall(base_interval)
    if base_shortfall <= tolerance_mw:
        return []

    flippable = np.flatnonzero(
        (coefficients != 0.0)
        & (((binary == 0) & ~masks.fixed_off) | ((binary == 1) & ~masks.fixed_on))
    )
    evaluated_singles: list[tuple[tuple[Any, ...], int, np.ndarray, dict[str, Any]]] = []

    def evaluate(
        candidate: np.ndarray, positions: tuple[int, ...]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        interval = row_interval(candidate)
        shortfall = row_shortfall(interval)
        minimum_capacity, maximum_capacity, capacity_shortfall = capacity_interval(candidate)
        economic_delta = float(
            sum(
                values[position] if candidate[position] else -values[position]
                for position in positions
            )
        )
        source_keys = tuple(int(source_rows[position]) + 1 for position in positions)
        rank = (
            int(shortfall > tolerance_mw),
            shortfall,
            int(capacity_shortfall > tolerance_mw),
            capacity_shortfall,
            len(positions),
            economic_delta,
            source_keys,
        )
        audit = {
            "policy": "violated_constant_coupling_row_one_two_flip_v1",
            "trigger_row_name": violated_row_name,
            "trigger_row_lower": lower,
            "trigger_row_upper": upper,
            "base_row_minimum_activity": base_interval[0],
            "base_row_maximum_activity": base_interval[1],
            "base_row_shortfall_mw": base_shortfall,
            "repaired_row_minimum_activity": interval[0],
            "repaired_row_maximum_activity": interval[1],
            "repaired_row_shortfall_mw": shortfall,
            "minimum_dispatch_mw": minimum_capacity,
            "maximum_dispatch_mw": maximum_capacity,
            "capacity_shortfall_mw": capacity_shortfall,
            "hamming_distance": len(positions),
            "flipped_generator_source_rows": list(source_keys),
            "turned_on_generator_source_rows": [
                int(source_rows[position]) + 1 for position in positions if candidate[position] == 1
            ],
            "turned_off_generator_source_rows": [
                int(source_rows[position]) + 1 for position in positions if candidate[position] == 0
            ],
            "economic_rank_delta": economic_delta,
            "exact_source_pmin_pmax_retained": True,
            "candidate_only_not_feasibility_proof": True,
        }
        return rank, audit

    for position in flippable:
        candidate = binary.copy()
        candidate[position] = np.int8(1 - candidate[position])
        rank, audit = evaluate(candidate, (int(position),))
        evaluated_singles.append((rank, int(position), candidate, audit))

    eligible: list[tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]] = [
        (rank, candidate, audit)
        for rank, _position, candidate, audit in evaluated_singles
        if audit["repaired_row_shortfall_mw"] <= tolerance_mw
        and audit["capacity_shortfall_mw"] <= tolerance_mw
    ]
    if len(eligible) < maximum_repairs:
        pair_pool = sorted(evaluated_singles, key=lambda item: item[0])[:pair_search_limit]
        for left_index, left in enumerate(pair_pool):
            for right in pair_pool[left_index + 1 :]:
                first_position = left[1]
                second_position = right[1]
                candidate = binary.copy()
                candidate[first_position] = np.int8(1 - candidate[first_position])
                candidate[second_position] = np.int8(1 - candidate[second_position])
                rank, audit = evaluate(candidate, (int(first_position), int(second_position)))
                if (
                    audit["repaired_row_shortfall_mw"] <= tolerance_mw
                    and audit["capacity_shortfall_mw"] <= tolerance_mw
                ):
                    eligible.append((rank, candidate, audit))

    repairs: list[NetworkCommitmentRepair] = []
    seen: set[str] = set()
    for rank, candidate, audit in sorted(eligible, key=lambda item: item[0]):
        digest = hashlib.sha256(candidate.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        audit["candidate_commitment_sha256"] = digest
        audit["deterministic_rank"] = list(rank[:-1]) + [list(rank[-1])]
        repairs.append(NetworkCommitmentRepair(candidate.copy(), audit))
        if len(repairs) >= maximum_repairs:
            break
    return repairs


def _capacity_repaired_commitment(
    case: Any,
    source_rows: np.ndarray,
    proposed: np.ndarray,
    on_values: np.ndarray,
    masks: RegionMasks,
    demand_mw: float,
) -> np.ndarray:
    commitment = np.asarray(proposed >= 0.5, dtype=np.int8)
    commitment[masks.fixed_off] = 0
    commitment[masks.fixed_on] = 1
    pmax = case.gen[source_rows, PMAX]
    pmin = case.gen[source_rows, PMIN]
    free_off = np.flatnonzero((commitment == 0) & ~masks.fixed_off)
    for position in free_off[np.argsort(on_values[free_off])]:
        if float(pmax @ commitment) >= demand_mw:
            break
        commitment[position] = 1
    if float(pmax @ commitment) < demand_mw:
        raise ScopfError("Candidate commitment cannot cover demand at PMAX")
    removable = np.flatnonzero((commitment == 1) & ~masks.fixed_on)
    for position in removable[np.argsort(on_values[removable])[::-1]]:
        if float(pmin @ commitment) <= demand_mw:
            break
        trial = commitment.copy()
        trial[position] = 0
        if float(pmax @ trial) >= demand_mw:
            commitment = trial
    if float(pmin @ commitment) > demand_mw:
        raise ScopfError("Candidate commitment exceeds demand at PMIN")
    return commitment


def _commitment_upper_cut_record(
    cut: CommitmentUpperCut, generator_source_rows: np.ndarray
) -> dict[str, Any]:
    """Serialize either cut type with one-based public MATPOWER row identity."""

    rows = np.asarray(generator_source_rows, dtype=np.int64)
    if isinstance(cut, CommitmentFeasibilityCut):
        return cut.as_dict(rows)
    return cut.as_dict(rows + 1)


def _certificate_dual_arrays(
    master: ReducedMaster,
    evaluation: LagrangianEvaluation,
    commitment_cuts: tuple[CommitmentUpperCut, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Place a replayable certificate into one master's canonical row order."""

    row_by_name = {name: row for row, name in enumerate(master.canonical.row_names)}
    if len(row_by_name) != master.canonical.num_rows:
        raise ScopfError("Certificate dual mapping found duplicate canonical row names")
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    for row_name, value in evaluation.coupling_duals:
        row = row_by_name.get(row_name)
        if row is None:
            raise ScopfError("Certificate dual mapping lost a coupling row")
        row_dual[row] = float(value)
    cut_dual_by_id = {
        str(cut_id): float(value)
        for cut_id, value in evaluation.commitment_cut_duals
    }
    if len(cut_dual_by_id) != len(evaluation.commitment_cut_duals):
        raise ScopfError("Certificate dual mapping found duplicate commitment cuts")
    available_ids = {cut.cut_id for cut in commitment_cuts}
    if not set(cut_dual_by_id).issubset(available_ids):
        raise ScopfError("Certificate dual mapping references a missing commitment cut")
    cut_dual = np.asarray(
        [cut_dual_by_id.get(cut.cut_id, 0.0) for cut in commitment_cuts],
        dtype=np.float64,
    )
    for cut, value in zip(commitment_cuts, cut_dual, strict=True):
        row = row_by_name.get(cut.cut_id)
        if row is None:
            raise ScopfError("Certificate dual mapping lost a commitment-cut row")
        row_dual[row] = float(value)
    return row_dual, cut_dual


def _lagrangian_precondition_scales(
    *,
    master: ReducedMaster,
    commitment_cuts: tuple[CommitmentUpperCut, ...],
    config: RunConfig,
    base_mva: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not bool(config.runtime.get("lagrangian_diagonal_preconditioning", False)):
        return None, None
    profile = config.raw["platforms"]["dgx_spark"]
    _column_scale, row_scale = native_scaling_vectors(
        master.canonical,
        mode=str(profile["native_scaling_mode"]),
        base_mva=float(base_mva),
    )
    row_by_name = {name: row for row, name in enumerate(master.canonical.row_names)}
    coupling = np.asarray(
        [
            row_scale[row.row_index]
            for row in sorted(master.coupling_rows, key=lambda row: row.row_name)
        ],
        dtype=np.float64,
    )
    cuts = np.asarray(
        [row_scale[row_by_name[cut.cut_id]] for cut in commitment_cuts],
        dtype=np.float64,
    )
    return coupling, cuts


def _refresh_region_with_global_commitment_cuts(
    *,
    region: SolvedRegion,
    global_cuts: tuple[CommitmentFeasibilityCut, ...],
    case: Any,
    network: NetworkData,
    config: RunConfig,
    dual_target_objective: float,
) -> SolvedRegion:
    """Add globally valid cuts and retain the stronger replayable certificate."""

    existing_by_id = {cut.cut_id: cut for cut in region.commitment_cuts}
    added_globals = tuple(
        sorted(
            (
                cut
                for cut in global_cuts
                if cut.cut_id not in existing_by_id
            ),
            key=lambda cut: cut.cut_id,
        )
    )
    combined = added_globals + region.commitment_cuts
    if tuple(cut.cut_id for cut in combined) == tuple(
        cut.cut_id for cut in region.commitment_cuts
    ):
        return region
    rebuilt = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=region.masks,
        initial_pairs=region.security_pairs,
        commitment_cuts=combined,
    )
    _validate_prepared_region_master(
        rebuilt, region.masks, region.security_pairs, combined
    )
    inherited_row_dual, inherited_cut_dual = _certificate_dual_arrays(
        rebuilt, region.lagrangian, combined
    )
    safety = float(region.lagrangian.safety_margin_dollars)
    inherited = evaluate_lagrangian_bound(
        rebuilt,
        inherited_row_dual,
        region.masks,
        safety_margin_dollars=safety,
        commitment_cuts=combined,
        commitment_cut_dual=inherited_cut_dual,
    )
    replay_drift = abs(
        inherited.conservative_lower_bound
        - region.lagrangian.conservative_lower_bound
    )
    tolerance = float(config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"])
    if replay_drift > tolerance:
        raise ScopfError(
            "Adding zero-dual global cuts changed the inherited Lagrangian certificate"
        )
    coupling_scales, cut_scales = _lagrangian_precondition_scales(
        master=rebuilt,
        commitment_cuts=combined,
        config=config,
        base_mva=float(case.base_mva),
    )
    started = time.perf_counter()
    polished_row_dual, gpu = optimize_lagrangian_bound_cupy(
        rebuilt,
        inherited_row_dual,
        region.masks,
        relaxation_primal_objective=float(dual_target_objective),
        iterations=int(config.runtime["phase_lagrangian_gpu_iterations"]),
        polyak_fraction=float(
            config.raw["platforms"]["dgx_spark"]["lagrangian_polyak_fraction"]
        ),
        commitment_cuts=combined,
        initial_commitment_cut_dual=inherited_cut_dual,
        coupling_row_scales=coupling_scales,
        commitment_cut_scales=cut_scales,
    )
    polished_cut_dual = np.asarray(
        gpu.get("best_commitment_cut_dual", np.zeros(len(combined))),
        dtype=np.float64,
    )
    polished = evaluate_lagrangian_bound(
        rebuilt,
        polished_row_dual,
        region.masks,
        safety_margin_dollars=safety,
        commitment_cuts=combined,
        commitment_cut_dual=polished_cut_dual,
    )
    selected_polished = bool(
        polished.conservative_lower_bound >= inherited.conservative_lower_bound
    )
    selected = polished if selected_polished else inherited
    selected_row_dual = polished_row_dual if selected_polished else inherited_row_dual
    selected_cut_dual = polished_cut_dual if selected_polished else inherited_cut_dual
    gpu.update(
        {
            "wall_time_seconds": time.perf_counter() - started,
            "polished_cpu_replay_difference_dollars": abs(
                float(gpu["best_raw_lower_bound"]) - polished.raw_lower_bound
            ),
            "global_commitment_cut_refresh": True,
            "global_feasibility_cut_count": len(global_cuts),
            "inherited_conservative_lower_bound": inherited.conservative_lower_bound,
            "polished_conservative_lower_bound": polished.conservative_lower_bound,
            "selected_certificate_source": (
                "preconditioned_gpu_polish"
                if selected_polished
                else "inherited_zero_new_cut_duals"
            ),
            "selected_certificate_monotone": True,
            "best_raw_lower_bound": selected.raw_lower_bound,
            "best_minimizing_commitment": selected.minimizing_commitment,
            "best_commitment_cut_dual": selected_cut_dual,
            "cpu_replay_difference_dollars": 0.0,
        }
    )
    return replace(
        region,
        master=rebuilt,
        canonical_row_dual=np.asarray(selected_row_dual, dtype=np.float64),
        lagrangian=selected,
        gpu_lagrangian=gpu,
        commitment_cuts=combined,
        commitment_cut_row_by_id={
            cut.cut_id: rebuilt.canonical.row_names.index(cut.cut_id)
            for cut in combined
        },
    )


def _enforce_monotone_child_certificate(
    *,
    parent: SolvedRegion,
    child: SolvedRegion,
    replay_tolerance_dollars: float,
) -> SolvedRegion:
    """A child is a subset of its parent, so its inherited bound cannot fall."""

    inherited_row_dual, inherited_cut_dual = _certificate_dual_arrays(
        child.master, parent.lagrangian, child.commitment_cuts
    )
    inherited = evaluate_lagrangian_bound(
        child.master,
        inherited_row_dual,
        child.masks,
        safety_margin_dollars=float(parent.lagrangian.safety_margin_dollars),
        commitment_cuts=child.commitment_cuts,
        commitment_cut_dual=inherited_cut_dual,
    )
    if (
        inherited.conservative_lower_bound
        + float(replay_tolerance_dollars)
        < parent.lagrangian.conservative_lower_bound
    ):
        raise ScopfError("Inherited child certificate regressed below its parent")
    use_inherited = bool(
        inherited.conservative_lower_bound
        > child.lagrangian.conservative_lower_bound
    )
    audit = {
        **child.gpu_lagrangian,
        "parent_inherited_certificate": {
            "parent_region_id": parent.region_id,
            "parent_conservative_lower_bound": (
                parent.lagrangian.conservative_lower_bound
            ),
            "child_native_conservative_lower_bound": (
                child.lagrangian.conservative_lower_bound
            ),
            "inherited_child_conservative_lower_bound": (
                inherited.conservative_lower_bound
            ),
            "selected": use_inherited,
            "monotonicity_passed": True,
        },
    }
    if not use_inherited:
        return replace(child, gpu_lagrangian=audit)
    inherited_cut_by_id = dict(inherited.commitment_cut_duals)
    audit.update(
        {
            "best_raw_lower_bound": inherited.raw_lower_bound,
            "best_minimizing_commitment": inherited.minimizing_commitment,
            "best_commitment_cut_dual": np.asarray(
                [
                    inherited_cut_by_id.get(cut.cut_id, 0.0)
                    for cut in child.commitment_cuts
                ],
                dtype=np.float64,
            ),
            "selected_certificate_source": "replayed_parent_on_child_region",
            "selected_certificate_monotone": True,
        }
    )
    return replace(
        child,
        canonical_row_dual=inherited_row_dual,
        lagrangian=inherited,
        gpu_lagrangian=audit,
    )


def _region_record(
    region: SolvedRegion, *, compact_lagrangian_certificate: bool = False
) -> dict[str, Any]:
    record = {
        "region_id": region.region_id,
        **region.masks.as_dict(region.master.index.generator_source_rows + 1),
        "lp_objective": region.solve.primal_objective,
        "relaxation_solution_role": region.solve.statistics.get(
            "relaxation_solution_role", "cost_lp_solution"
        ),
        "ordinary_cost_lp_solved": region.solve.statistics.get(
            "ordinary_cost_lp_solved", True
        ),
        "lp_commitment_fractional_count": int(
            np.count_nonzero(np.abs(region.commitment - np.rint(region.commitment)) > 1e-6)
        ),
        "constraint_generation_rounds": region.rounds,
        "final_screen": region.final_screen,
        "coefficient_cleanup_audit": dict(region.master.coefficient_cleanup_audit),
        "security_row_equivalence": _security_row_equivalence_record(region.master),
        "security_pairs": [security_pair_record(pair) for pair in region.security_pairs],
        "commitment_upper_cuts": [
            _commitment_upper_cut_record(
                cut, region.master.index.generator_source_rows
            )
            for cut in region.commitment_cuts
        ],
        "gpu_lagrangian_evaluation": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in region.gpu_lagrangian.items()
        },
        "lagrangian_certificate": region.lagrangian.as_dict(
            region.master.index.generator_source_rows + 1,
            compact=compact_lagrangian_certificate,
        ),
    }
    return record


def _security_row_equivalence_record(master: ReducedMaster) -> dict[str, Any]:
    classes = {
        representative: list(members)
        for representative, members in sorted(master.security_pair_equivalence_classes.items())
    }
    return {
        "policy": "exact_post_cleanup_fp64_coefficient_and_rhs_identity",
        "logical_security_pair_count": len(master.security_pair_ids),
        "distinct_solver_security_row_count": len(classes),
        "duplicate_solver_row_count": len(master.security_pair_ids) - len(classes),
        "representative_by_pair_id": dict(
            sorted(master.security_pair_representative_by_id.items())
        ),
        "equivalence_classes": classes,
    }


def _relative_gap(*, objective: float, lower_bound: float) -> float:
    """Return the registered incumbent-relative minimization gap."""

    gap = (float(objective) - float(lower_bound)) / max(1.0, abs(float(objective)))
    if gap < -1e-9:
        raise ScopfError("Lagrangian lower bound exceeds the secure incumbent")
    return gap


def _replay_cleanup_audit_comparison(
    recorded: dict[str, Any],
    rebuilt: dict[str, Any],
) -> dict[str, Any]:
    """Validate cleanup invariants while allowing architecture-dependent dust counts."""

    maximum_difference = 0.0
    maximum_integer_difference = 0
    differing_integer_fields: list[dict[str, Any]] = []
    exact_paths = {
        "coefficient_cleanup_audit.policy",
        "coefficient_cleanup_audit.zero_tolerance",
        "coefficient_cleanup_audit.potential_flow_operator_dust.zero_tolerance",
        "coefficient_cleanup_audit.physical_injection_operator_changed",
        "coefficient_cleanup_audit.solver_rows_are_relaxations_of_original_rows",
        "coefficient_cleanup_audit.exact_duplicate_security_row_count",
    }

    def compare(left: Any, right: Any, path: str) -> None:
        nonlocal maximum_difference, maximum_integer_difference
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                raise ScopfError(f"Cleanup-audit key mismatch at {path}")
            for key in sorted(left):
                compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                raise ScopfError(f"Cleanup-audit boolean mismatch at {path}")
            return
        if isinstance(left, int) or isinstance(right, int):
            if type(left) is not type(right):
                raise ScopfError(f"Cleanup-audit numeric type mismatch at {path}")
            difference = abs(int(left) - int(right))
            if path in exact_paths and difference:
                raise ScopfError(f"Cleanup-audit exact integer mismatch at {path}")
            maximum_integer_difference = max(maximum_integer_difference, difference)
            if difference:
                differing_integer_fields.append(
                    {
                        "path": path,
                        "recorded": int(left),
                        "rebuilt": int(right),
                        "absolute_difference": difference,
                    }
                )
            return
        if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
            difference = abs(float(left) - float(right))
            if not np.isfinite(difference):
                raise ScopfError(f"Cleanup-audit nonfinite FP64 value at {path}")
            if path in exact_paths and difference:
                raise ScopfError(f"Cleanup-audit exact FP64 mismatch at {path}")
            maximum_difference = max(maximum_difference, difference)
            return
        if left != right:
            raise ScopfError(f"Cleanup-audit identity mismatch at {path}")

    compare(recorded, rebuilt, "coefficient_cleanup_audit")
    for label, audit in (("recorded", recorded), ("rebuilt", rebuilt)):
        zero_tolerance = float(audit["zero_tolerance"])
        for key in (
            "maximum_absolute_dropped_generator_coefficient",
            "maximum_row_rhs_outward_relaxation",
            "total_rhs_outward_relaxation",
        ):
            value = float(audit.get(key, 0.0))
            if not np.isfinite(value) or value < 0.0:
                raise ScopfError(f"Cleanup-audit {label} {key} is invalid")
        potential = audit["potential_flow_operator_dust"]
        maximum_dropped = float(potential["maximum_absolute_dropped_coefficient"])
        if maximum_dropped > zero_tolerance * (1.0 + 1e-12):
            raise ScopfError(f"Cleanup-audit {label} flow dust exceeds its zero tolerance")
        generator_maximum = float(audit.get("maximum_absolute_dropped_generator_coefficient", 0.0))
        if generator_maximum > zero_tolerance * (1.0 + 1e-12):
            raise ScopfError(f"Cleanup-audit {label} generator dust exceeds its zero tolerance")
        if audit.get("physical_injection_operator_changed") is not False:
            raise ScopfError(f"Cleanup-audit {label} changed the physical operator")
        if audit.get("solver_rows_are_relaxations_of_original_rows") is not True:
            raise ScopfError(f"Cleanup-audit {label} lost outward-relaxation proof")
    return {
        "maximum_fp64_difference": maximum_difference,
        "maximum_integer_difference": maximum_integer_difference,
        "differing_integer_fields": differing_integer_fields,
    }


def _native_constraint_layout(model: Any) -> list[tuple[str, str, int]]:
    """Mirror the cuOpt adapter's deterministic ranged-row expansion."""

    lower, upper = model.row_bound_arrays()
    layout: list[tuple[str, str, int]] = []
    for row, name in enumerate(model.row_names):
        lo = float(lower[row])
        hi = float(upper[row])
        if np.isfinite(lo) and np.isfinite(hi) and lo == hi:
            layout.append((name, "equality", row))
        else:
            if np.isfinite(lo):
                layout.append((name, "lower", row))
            if np.isfinite(hi):
                layout.append((name, "upper", row))
    return layout


def _map_native_row_dual_by_identity(
    source_model: Any,
    target_model: Any,
    source_native_dual: np.ndarray,
    *,
    scaling_mode: str,
    base_mva: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map same-side row multipliers between related canonical LPs."""

    source_layout = _native_constraint_layout(source_model)
    target_layout = _native_constraint_layout(target_model)
    supplied = np.asarray(source_native_dual, dtype=np.float64)
    if supplied.shape != (len(source_layout),) or not np.all(np.isfinite(supplied)):
        raise ScopfError("Source native row-dual warm start has invalid values")
    _source_columns, source_row_scale = native_scaling_vectors(
        source_model, mode=scaling_mode, base_mva=base_mva
    )
    _target_columns, target_row_scale = native_scaling_vectors(
        target_model, mode=scaling_mode, base_mva=base_mva
    )
    target_by_key = {
        (name, side): (position, row) for position, (name, side, row) in enumerate(target_layout)
    }
    mapped = np.zeros(len(target_layout), dtype=np.float64)
    mapped_count = 0
    for value, (name, side, source_row) in zip(supplied, source_layout, strict=True):
        target = target_by_key.get((name, side))
        if target is None:
            continue
        target_position, target_row = target
        mapped[target_position] = (
            float(value) * float(source_row_scale[source_row]) / float(target_row_scale[target_row])
        )
        mapped_count += 1
    return mapped, {
        "policy": "same_row_name_side_native_scaling_ratio_v1",
        "source_native_constraint_count": len(source_layout),
        "target_native_constraint_count": len(target_layout),
        "mapped_native_constraint_count": mapped_count,
        "zero_filled_native_constraint_count": len(target_layout) - mapped_count,
        "source_native_dual_sha256": hashlib.sha256(supplied.tobytes()).hexdigest(),
        "target_native_dual_sha256": hashlib.sha256(mapped.tobytes()).hexdigest(),
    }


def _map_phase_one_dual_to_source_native(
    phase_model: Any,
    source_model: Any,
    phase_native_dual: np.ndarray,
    *,
    scaling_mode: str,
    base_mva: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map Phase-I upper-row duals into the source LP's native row order."""

    supplied = np.asarray(phase_native_dual, dtype=np.float64)
    phase_layout = _native_constraint_layout(phase_model)
    if (
        supplied.shape != (len(phase_layout),)
        or len(phase_layout) != phase_model.num_rows
        or not np.all(np.isfinite(supplied))
    ):
        raise ScopfError("Phase-I native row-dual warm start has invalid values")
    if any(side != "upper" for _name, side, _row in phase_layout):
        raise ScopfError("Phase-I native row-dual mapping expects upper rows only")
    _phase_columns, phase_row_scale = native_scaling_vectors(
        phase_model, mode=scaling_mode, base_mva=base_mva
    )
    _source_columns, source_row_scale = native_scaling_vectors(
        source_model, mode=scaling_mode, base_mva=base_mva
    )
    source_layout = _native_constraint_layout(source_model)
    source_position = {
        (name, side): (position, row) for position, (name, side, row) in enumerate(source_layout)
    }
    mapped = np.zeros(len(source_layout), dtype=np.float64)
    mapped_sides: set[tuple[str, str]] = set()
    for value, (phase_name, _phase_side, phase_row) in zip(supplied, phase_layout, strict=True):
        parts = phase_name.split("__", 3)
        if (
            len(parts) != 4
            or parts[0] != "phase1"
            or not parts[1].isdigit()
            or parts[2] not in {"lower", "upper"}
        ):
            raise ScopfError(f"Malformed Phase-I warm-start row {phase_name!r}")
        source_name = parts[3]
        phase_side = parts[2]
        direct_key = (source_name, phase_side)
        target = source_position.get(direct_key)
        sign = -1.0 if phase_side == "lower" else 1.0
        if target is None:
            target = source_position.get((source_name, "equality"))
        if target is None:
            raise ScopfError(f"Phase-I warm-start row lacks source identity: {phase_name!r}")
        target_position, source_row = target
        mapped[target_position] += (
            sign
            * float(value)
            * float(phase_row_scale[phase_row])
            / float(source_row_scale[source_row])
        )
        mapped_sides.add((source_name, phase_side))
    if not np.all(np.isfinite(mapped)):
        raise ScopfError("Mapped Phase-I source row-dual is nonfinite")
    return mapped, {
        "policy": "phase_one_side_sign_and_native_scaling_ratio_v1",
        "phase_native_constraint_count": len(phase_layout),
        "source_native_constraint_count": len(source_layout),
        "mapped_phase_side_count": len(mapped_sides),
        "phase_native_dual_sha256": hashlib.sha256(supplied.tobytes()).hexdigest(),
        "source_native_dual_sha256": hashlib.sha256(mapped.tobytes()).hexdigest(),
        "certificate_claimed": False,
        "warm_start_only": True,
    }


def _run_phase_one_attempt(
    *,
    region_id: str,
    masks: RegionMasks,
    rejected: RegionAttemptRejected,
    case: Any,
    config: RunConfig,
    deadline: Deadline,
    attempt_kind: str = "post_cost_lp_rejection",
    time_limit_seconds: float | None = None,
    capacity_gate: dict[str, Any] | None = None,
    commitment_cuts: tuple[CommitmentUpperCut, ...] = (),
) -> PhaseOneAttemptResult:
    """Run one cold GPU Phase-I solve and serialize its projected box dual."""

    runtime = config.runtime
    profile = config.raw["platforms"]["dgx_spark"]
    attempt_started = time.perf_counter()
    deadline.require(f"Phase-I certificate for {region_id}")
    phase_model = build_phase_one_model(
        rejected.master.canonical,
        base_mva=float(case.base_mva),
        maximum_violation_pu=float(runtime["phase_one_maximum_violation_pu"]),
    )
    source_model = rejected.master.canonical
    if phase_model.variable_names[:-1] != source_model.variable_names:
        raise ScopfError("Phase-I and cost-LP source-column identities changed")
    if not np.array_equal(
        np.asarray(phase_model.column_lower[:-1], dtype=np.float64),
        np.asarray(source_model.column_lower, dtype=np.float64),
    ) or not np.array_equal(
        np.asarray(phase_model.column_upper[:-1], dtype=np.float64),
        np.asarray(source_model.column_upper, dtype=np.float64),
    ):
        raise ScopfError("Phase-I and cost-LP source-column bounds changed")
    source_column_scale, _ = native_scaling_vectors(
        source_model,
        mode=str(profile["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    phase_column_scale, phase_row_scale = native_scaling_vectors(
        phase_model,
        mode=str(profile["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    if not np.array_equal(source_column_scale, phase_column_scale[:-1]):
        raise ScopfError("Phase-I and cost-LP native source-column scaling changed")
    preparation_wall = time.perf_counter() - attempt_started
    budget = min(
        deadline.solver_budget(),
        (
            float(runtime["phase_one_time_limit_seconds"])
            if time_limit_seconds is None
            else float(time_limit_seconds)
        ),
    )
    solve_started = time.perf_counter()
    solve = solve_cuopt_continuous_pdlp(
        phase_model,
        time_limit_seconds=budget,
        optimality_tolerance=float(profile["pdlp_optimality_tolerance"]),
        primal_feasibility_tolerance=float(config.model["model_residual_tolerance_pu"]),
        certificate_residual_tolerance=float(profile["dual_certificate_residual_tolerance"]),
        native_scaling_mode=str(profile["native_scaling_mode"]),
        native_base_mva=float(case.base_mva),
        log_to_console=True,
        per_constraint_residual=bool(profile["per_constraint_residual"]),
        presolve=int(profile["presolve"]),
        pdlp_solver_mode=int(profile.get("pdlp_solver_mode_native", 4)),
    )
    record: dict[str, Any] = {
        "region_id": region_id,
        "phase_one_attempt_kind": attempt_kind,
        **masks.as_dict(rejected.master.index.generator_source_rows + 1),
        "source_attempt_rejection_reason": rejected.reason,
        "source_constraint_generation_rounds": rejected.rounds,
        "security_pairs": [security_pair_record(pair) for pair in rejected.security_pairs],
        "commitment_upper_cuts": [
            cut.as_dict(rejected.master.index.generator_source_rows + 1)
            for cut in commitment_cuts
        ],
        "coefficient_cleanup_audit": dict(rejected.master.coefficient_cleanup_audit),
        "security_row_equivalence": _security_row_equivalence_record(rejected.master),
        "phase_one_model": {
            "columns": phase_model.num_columns,
            "rows": phase_model.num_rows,
            "registered_maximum_violation_cap_pu": float(runtime["phase_one_maximum_violation_pu"]),
            "box_derived_violation_upper_bound_pu": float(phase_model.column_upper[-1]),
        },
        "solver_budget_seconds": budget,
        "preparation_wall_time_seconds": preparation_wall,
        "adapter_wall_time_seconds": time.perf_counter() - solve_started,
        "total_attempt_wall_time_seconds": time.perf_counter() - attempt_started,
        "solve": _solve_summary(solve),
        "prune_certified": False,
    }
    if capacity_gate is not None:
        record["pmin_pmax_capacity_gate"] = capacity_gate
    source_native_primal: np.ndarray | None = None
    source_canonical_values: np.ndarray | None = None
    source_residual_pu: float | None = None
    violation_value_pu: float | None = None
    numeric_primal_feasible = bool(
        solve.statistics.get("dual_certificate", {}).get("primal_feasible", False)
    )
    if solve.values is not None and solve.native_primal is not None:
        canonical_values = np.asarray(solve.values, dtype=np.float64)
        native_values = np.asarray(solve.native_primal, dtype=np.float64)
        if (
            canonical_values.shape == (phase_model.num_columns,)
            and native_values.shape == (phase_model.num_columns,)
            and np.all(np.isfinite(canonical_values))
            and np.all(np.isfinite(native_values))
        ):
            violation_value_pu = float(canonical_values[-1])
            source_values = canonical_values[:-1]
            source_residual_pu = rejected.master.canonical.max_row_violation(source_values) / float(
                case.base_mva
            )
            source_tolerance = float(config.model["model_residual_tolerance_pu"])
            if (
                numeric_primal_feasible
                and violation_value_pu <= source_tolerance
                and source_residual_pu <= source_tolerance
            ):
                source_native_primal = native_values[:-1].copy()
                source_canonical_values = source_values.copy()
    record["source_feasible_warm_start"] = {
        "eligible": source_native_primal is not None,
        "phase_one_violation_pu": violation_value_pu,
        "source_model_residual_pu": source_residual_pu,
        "numeric_certificate_primal_feasible": numeric_primal_feasible,
        "native_primal_count": (
            None if source_native_primal is None else int(source_native_primal.size)
        ),
        "native_primal_sha256": (
            None
            if source_native_primal is None
            else hashlib.sha256(source_native_primal.tobytes()).hexdigest()
        ),
        "dual_transferred": False,
        "dual_not_transferred_reason": (
            "phase_one_uses_split_upper_rows_while_cost_lp_uses_native_ranged_rows"
        ),
        "source_column_scaling_exactly_equal": True,
    }
    source_native_row_dual: np.ndarray | None = None
    if (
        config.benchmark_id in ACTIVSG2000_V4_PLUS_EXPERIMENT_IDS
        and source_native_primal is not None
        and solve.native_row_dual is not None
        and solve.native_row_dual.shape == (phase_model.num_rows,)
    ):
        source_native_row_dual, dual_mapping = _map_phase_one_dual_to_source_native(
            phase_model,
            source_model,
            solve.native_row_dual,
            scaling_mode=str(profile["native_scaling_mode"]),
            base_mva=float(case.base_mva),
        )
        record["source_feasible_warm_start"].update(
            {
                "dual_transferred": True,
                "dual_not_transferred_reason": None,
                "native_row_dual_count": int(source_native_row_dual.size),
                "native_row_dual_sha256": hashlib.sha256(
                    source_native_row_dual.tobytes()
                ).hexdigest(),
                "dual_mapping": dual_mapping,
            }
        )
    if str(solve.statistics.get("error_status")) != "Success" or solve.native_row_dual is None:
        record["certificate_status"] = "unavailable_solver_dual"
        record["total_attempt_wall_time_seconds"] = time.perf_counter() - attempt_started
        return PhaseOneAttemptResult(
            record,
            source_native_primal,
            source_native_row_dual,
            source_canonical_values,
        )
    if solve.native_row_dual.shape != (phase_model.num_rows,):
        record["certificate_status"] = "invalid_native_dual_shape"
        record["total_attempt_wall_time_seconds"] = time.perf_counter() - attempt_started
        return PhaseOneAttemptResult(
            record,
            source_native_primal,
            source_native_row_dual,
            source_canonical_values,
        )
    canonical_dual = np.asarray(solve.native_row_dual, dtype=np.float64) * phase_row_scale
    certificate = phase_one_certificate(
        phase_model,
        canonical_dual,
        safety_margin_pu=float(runtime["phase_one_safety_margin_pu"]),
        infeasibility_threshold_pu=float(runtime["phase_one_infeasibility_threshold_pu"]),
    )
    record["phase_one_certificate"] = certificate
    record["certificate_status"] = (
        "independently_replayable_prune"
        if certificate["prune_certified"]
        else "dual_bound_not_strong_enough"
    )
    record["prune_certified"] = bool(certificate["prune_certified"])
    if record["prune_certified"] and source_native_primal is not None:
        raise ScopfError(
            "Phase-I produced contradictory feasible-primal and positive-dual certificates"
        )
    record["total_attempt_wall_time_seconds"] = time.perf_counter() - attempt_started
    return PhaseOneAttemptResult(
        record,
        source_native_primal,
        source_native_row_dual,
        source_canonical_values,
    )


def _solve_phase_one_lagrangian_region(
    *,
    region_id: str,
    masks: RegionMasks,
    parent: SolvedRegion,
    master: ReducedMaster,
    initial_pairs: tuple[SecurityPair, ...],
    initial_precheck: PhaseOneAttemptResult,
    case: Any,
    network: NetworkData,
    config: RunConfig,
    deadline: Deadline,
    screener: ContingencyScreener,
    checkpoint: Callable[[], None],
) -> tuple[SolvedRegion | None, dict[str, Any] | None]:
    """Secure a child with Phase I and certify its bound without a cost LP.

    Phase I supplies only a feasible continuous point and can certify a
    positive infeasibility lower bound.  A child lower bound instead starts
    from the parent's already replayable coupling dual, extended with zeros
    for newly generated security rows, and is polished entirely by CuPy.  The
    feasible Phase-I point is merely a safe Polyak target; it is never called
    an optimal LP solution.
    """

    if config.benchmark_id not in {
        ACTIVSG2000_V6_EXPERIMENT_ID,
        ACTIVSG2000_V7_EXPERIMENT_ID,
        ACTIVSG2000_V8_EXPERIMENT_ID,
        ACTIVSG2000_V9_EXPERIMENT_ID,
    }:
        raise ScopfError(
            "Phase-I Lagrangian child engine is registered only for ACTIVSg2000 v6-v9"
        )
    pairs_by_id = {pair.pair_id: pair for pair in initial_pairs}
    if set(pairs_by_id) != set(master.security_pair_ids):
        raise ScopfError("Phase-I child initial security-pair identity changed")

    current = initial_precheck
    rounds: list[dict[str, Any]] = []
    maximum_rounds = int(config.runtime["maximum_constraint_generation_rounds"])
    for round_number in range(1, maximum_rounds + 1):
        deadline.require(f"Phase-I Lagrangian region {region_id} round {round_number}")
        record = current.record
        round_record: dict[str, Any] = {
            "round": round_number,
            "rows_before_phase_one": master.canonical.num_rows,
            "security_pairs_before_phase_one": len(pairs_by_id),
            "phase_one": _phase_one_attempt_summary(record),
            "adapter_wall_time_seconds": float(record["adapter_wall_time_seconds"]),
            "solve": record["solve"],
        }
        rounds.append(round_record)
        if record["prune_certified"]:
            return None, record
        if current.source_values is None or current.source_native_primal is None:
            raise RegionAttemptRejected(
                f"Region {region_id} Phase I returned neither a feasible point nor a prune",
                reason="phase_one_lagrangian_inconclusive",
                master=master,
                security_pairs=tuple(sorted(pairs_by_id.values())),
                rounds=rounds,
            )
        source_values = np.asarray(current.source_values, dtype=np.float64)
        dispatch = reduced_dispatch(master, source_values)
        screen_started = time.perf_counter()
        screened = screener.screen(
            master.operator.flows(dispatch),
            tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
            already_added=set(pairs_by_id),
        )
        final_screen = {
            "wall_time_seconds": time.perf_counter() - screen_started,
            "evaluated_sides": screened.evaluated_pairs,
            "new_violated_pairs": len(screened.violations),
            "maximum_violation_pu": screened.maximum_violation_pu,
            "maximum_pair_id": screened.maximum_pair_id,
        }
        round_record["screen"] = final_screen
        checkpoint()
        if not screened.violations:
            if screened.maximum_violation_pu > float(
                config.model["security_violation_tolerance_pu"]
            ):
                raise ScopfError(
                    f"Region {region_id} Phase-I final screen exceeds tolerance"
                )
            parent_dual_by_name = dict(parent.lagrangian.coupling_duals)
            inherited_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
            inherited_count = 0
            for coupling in master.coupling_rows:
                if coupling.row_name in parent_dual_by_name:
                    inherited_dual[coupling.row_index] = parent_dual_by_name[
                        coupling.row_name
                    ]
                    inherited_count += 1
            objective, _lower, _upper, _integrality = master.canonical.column_arrays()
            feasible_cost = float(objective @ source_values)
            gpu_started = time.perf_counter()
            polished_dual, gpu_evaluation = optimize_lagrangian_bound_cupy(
                master,
                inherited_dual,
                masks,
                relaxation_primal_objective=feasible_cost,
                iterations=int(config.runtime["phase_lagrangian_gpu_iterations"]),
                polyak_fraction=float(
                    config.raw["platforms"]["dgx_spark"][
                        "lagrangian_polyak_fraction"
                    ]
                ),
            )
            gpu_evaluation.update(
                {
                    "wall_time_seconds": time.perf_counter() - gpu_started,
                    "certificate_seed": "parent_coupling_dual_by_row_identity_v1",
                    "inherited_coupling_row_count": inherited_count,
                    "zero_initialized_new_coupling_row_count": (
                        len(master.coupling_rows) - inherited_count
                    ),
                    "phase_one_feasible_cost_target": feasible_cost,
                    "ordinary_cost_lp_solved": False,
                }
            )
            evaluation = evaluate_lagrangian_bound(
                master,
                polished_dual,
                masks,
                safety_margin_dollars=float(
                    config.raw["benchmark"]["certificate_safety_margin_dollars"]
                ),
            )
            inherited_evaluation = evaluate_lagrangian_bound(
                master,
                inherited_dual,
                masks,
                safety_margin_dollars=float(
                    config.raw["benchmark"]["certificate_safety_margin_dollars"]
                ),
            )
            if evaluation.conservative_lower_bound + 1e-6 < (
                inherited_evaluation.conservative_lower_bound
            ):
                raise ScopfError("GPU Lagrangian polishing weakened the inherited child bound")
            synthetic_solve = ContinuousSolveResult(
                status="PhaseOneFeasible",
                optimal=False,
                primal_objective=feasible_cost,
                dual_objective=None,
                values=source_values.copy(),
                native_primal=np.asarray(
                    current.source_native_primal, dtype=np.float64
                ).copy(),
                native_row_dual=None,
                solve_time_seconds=float(
                    sum(
                        float(item["phase_one"]["adapter_wall_time_seconds"])
                        for item in rounds
                    )
                ),
                statistics={
                    "error_status": "Success",
                    "solved_by": "PDLP_PhaseOne_then_CuPy_Lagrangian",
                    "relaxation_solution_role": (
                        "secure_continuous_feasible_point_not_cost_optimum"
                    ),
                    "ordinary_cost_lp_solved": False,
                },
            )
            return (
                SolvedRegion(
                    region_id=region_id,
                    masks=masks,
                    master=master,
                    solve=synthetic_solve,
                    canonical_row_dual=polished_dual,
                    lagrangian=evaluation,
                    commitment=commitment_vector(master, source_values),
                    security_pairs=tuple(sorted(pairs_by_id.values())),
                    rounds=rounds,
                    final_screen=final_screen,
                    gpu_lagrangian=gpu_evaluation,
                ),
                None,
            )

        add_reduced_security_pairs(master, network, screened.violations)
        pairs_by_id.update((pair.pair_id, pair) for pair in screened.violations)
        round_record["added_pair_ids"] = [
            pair.pair_id for pair in screened.violations
        ]
        rejection = RegionAttemptRejected(
            f"Region {region_id} requires another Phase-I security round",
            reason="phase_one_lagrangian_security_generation",
            master=master,
            security_pairs=tuple(sorted(pairs_by_id.values())),
            rounds=rounds,
        )
        current = _run_phase_one_attempt(
            region_id=region_id,
            masks=masks,
            rejected=rejection,
            case=case,
            config=config,
            deadline=deadline,
            attempt_kind="phase_one_lagrangian_security_resolve",
            time_limit_seconds=float(
                config.runtime["precheck_phase_one_time_limit_seconds"]
            ),
        )
    raise RegionAttemptRejected(
        f"Region {region_id} exhausted Phase-I security-generation rounds",
        reason="phase_one_lagrangian_constraint_generation_round_limit",
        master=master,
        security_pairs=tuple(sorted(pairs_by_id.values())),
        rounds=rounds,
    )


def _phase_one_attempt_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Retain Phase-I timing/gate evidence without duplicating full dual vectors."""

    certificate = record.get("phase_one_certificate")
    certificate_summary = None
    if isinstance(certificate, dict):
        certificate_summary = {
            key: value for key, value in certificate.items() if key != "canonical_row_duals"
        }
    return {
        "region_id": record["region_id"],
        "phase_one_attempt_kind": record["phase_one_attempt_kind"],
        "pmin_pmax_capacity_gate": record.get("pmin_pmax_capacity_gate"),
        "phase_one_model": record["phase_one_model"],
        "solver_budget_seconds": record["solver_budget_seconds"],
        "adapter_wall_time_seconds": record["adapter_wall_time_seconds"],
        "preparation_wall_time_seconds": record["preparation_wall_time_seconds"],
        "total_attempt_wall_time_seconds": record["total_attempt_wall_time_seconds"],
        "solve": record["solve"],
        "certificate_status": record.get("certificate_status"),
        "prune_certified": record["prune_certified"],
        "phase_one_certificate": certificate_summary,
        "source_feasible_warm_start": record["source_feasible_warm_start"],
        "security_pair_count": len(record["security_pairs"]),
    }


def _masks_from_record(record: dict[str, Any], source_rows: np.ndarray) -> RegionMasks:
    lookup = {int(row): position for position, row in enumerate(source_rows)}
    off = np.zeros(source_rows.size, dtype=bool)
    on = np.zeros(source_rows.size, dtype=bool)
    for row in record["fixed_off_generator_source_rows"]:
        if int(row) not in lookup:
            raise ScopfError("Certificate fixes an unknown generator source row off")
        off[lookup[int(row)]] = True
    for row in record["fixed_on_generator_source_rows"]:
        if int(row) not in lookup:
            raise ScopfError("Certificate fixes an unknown generator source row on")
        on[lookup[int(row)]] = True
    masks = RegionMasks(off, on)
    masks.validate(source_rows.size)
    return masks


def verify_lagrangian_certificate_payload(
    config: RunConfig, payload: dict[str, Any]
) -> dict[str, Any]:
    """Reread raw inputs and replay every leaf bound and the region cover on CPU."""

    started = time.perf_counter()
    case = read_matpower_case(
        config.case_path, expected_sha256=config.raw["raw_inputs"]["case_sha256"]
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
    leaves: dict[
        str, tuple[RegionMasks, tuple[CommitmentCardinalityCut, ...]]
    ] = {}
    replayed_bounds: dict[str, float] = {}
    maximum_difference = 0.0
    maximum_lodf_replay_difference = 0.0
    maximum_cleanup_audit_difference = 0.0
    maximum_cleanup_audit_integer_difference = 0
    cleanup_audit_integer_differences: list[dict[str, Any]] = []
    lodf_tolerance = float(config.model.get("serialized_lodf_replay_tolerance", 0.0))

    def update_lodf_replay_difference(
        records: list[dict[str, Any]], pairs: tuple[SecurityPair, ...]
    ) -> None:
        nonlocal maximum_lodf_replay_difference
        for pair_record, pair in zip(records, pairs, strict=True):
            observed = (
                catalog.lodf[pair.monitored_active_index, pair.outage_column]
                if catalog.lodf is not None
                else catalog.lodf_operator.lodf_columns(
                    np.asarray([pair.outage_active_index], dtype=np.int64)
                )[pair.monitored_active_index, 0]
            )
            maximum_lodf_replay_difference = max(
                maximum_lodf_replay_difference,
                abs(float(observed) - float(pair_record["lodf_value"])),
            )

    validated_global_feasibility_cuts: dict[
        str, CommitmentFeasibilityCut
    ] = {}
    maximum_feasibility_cut_coefficient_replay_difference = 0.0
    maximum_feasibility_cut_rhs_replay_difference = 0.0
    maximum_feasibility_cut_portable_strengthening = 0.0
    for derivation_record in payload.get("commitment_feasibility_cuts", []):
        serialized_cut = derivation_record.get("cut")
        if not isinstance(serialized_cut, dict):
            raise ScopfError("Global feasibility-cut derivation lacks its cut")
        cut_id = str(serialized_cut.get("cut_id", ""))
        if not cut_id or cut_id in validated_global_feasibility_cuts:
            raise ScopfError("Global feasibility-cut derivation identity is duplicated")
        source_commitment_rows = tuple(
            int(row)
            for row in derivation_record.get(
                "source_commitment_generator_rows", []
            )
        )
        if len(set(source_commitment_rows)) != len(source_commitment_rows):
            raise ScopfError("Global feasibility-cut source commitment is duplicated")
        source_row_lookup = {
            int(row): position for position, row in enumerate(source_rows)
        }
        if any(row not in source_row_lookup for row in source_commitment_rows):
            raise ScopfError("Global feasibility cut fixes an unknown generator")
        binary = np.zeros(source_rows.size, dtype=np.int8)
        binary[
            np.asarray(
                [source_row_lookup[row] for row in source_commitment_rows],
                dtype=np.int64,
            )
        ] = 1
        if hashlib.sha256(binary.tobytes()).hexdigest() != serialized_cut.get(
            "source_commitment_sha256"
        ):
            raise ScopfError("Global feasibility-cut source commitment hash mismatch")

        cut_master = build_reduced_master(
            case,
            network,
            segments=10,
            coefficient_zero_tolerance=float(
                config.model.get("reduced_coefficient_zero_tolerance", 1e-14)
            ),
        )
        cut_pairs = tuple(
            security_pair_from_record(
                pair_record,
                catalog,
                lodf_absolute_tolerance=lodf_tolerance,
            )
            for pair_record in derivation_record.get("security_pairs", [])
        )
        update_lodf_replay_difference(
            derivation_record.get("security_pairs", []), cut_pairs
        )
        add_reduced_security_pairs(cut_master, network, cut_pairs)
        fix_commitments(cut_master, binary == 0, binary == 1)
        projection = build_fixed_commitment_projection(cut_master, binary)
        phase_model = build_phase_one_model(
            projection.canonical,
            base_mva=float(case.base_mva),
            maximum_violation_pu=float(
                config.runtime["phase_one_maximum_violation_pu"]
            ),
        )
        phase_row_by_semantic = {
            phase_one_semantic_row_key(name): row
            for row, name in enumerate(phase_model.row_names)
        }
        if len(phase_row_by_semantic) != phase_model.num_rows:
            raise ScopfError("Global feasibility-cut Phase-I rows are duplicated")
        phase_dual = np.zeros(phase_model.num_rows, dtype=np.float64)
        for item in derivation_record.get("nonzero_phase_row_duals", []):
            semantic = (
                f"phase1__{str(item['side'])}__{str(item['row_name'])}"
            )
            row = phase_row_by_semantic.get(semantic)
            if row is None or phase_dual[row] != 0.0:
                raise ScopfError(
                    "Global feasibility-cut Phase-I dual identity is invalid"
                )
            phase_dual[row] = float(item["canonical_row_dual"])
        recorded_phase_replay = derivation_record.get("phase_one_replay", {})
        rebuilt_phase_certificate = phase_one_certificate(
            phase_model,
            phase_dual,
            safety_margin_pu=float(recorded_phase_replay["safety_margin_pu"]),
            infeasibility_threshold_pu=float(
                recorded_phase_replay["infeasibility_threshold_pu"]
            ),
        )
        rebuilt_phase_replay = replay_phase_one_certificate(
            phase_model, rebuilt_phase_certificate
        )
        if not bool(rebuilt_phase_replay["prune_certified"]):
            raise ScopfError("Global feasibility-cut source dual is not positive")
        replay_tolerance = float(config.model["phase_one_replay_tolerance_pu"])
        phase_replay_differences: dict[str, dict[str, Any]] = {}
        for key in ("raw_lower_bound_pu", "conservative_lower_bound_pu"):
            rebuilt_value = float(rebuilt_phase_replay[key])
            recorded_value = float(recorded_phase_replay[key])
            if abs(rebuilt_value - recorded_value) > replay_tolerance:
                phase_replay_differences[key] = {
                    "rebuilt": rebuilt_value,
                    "recorded": recorded_value,
                }
        for key in (
            "safety_margin_pu",
            "infeasibility_threshold_pu",
            "prune_certified",
        ):
            if rebuilt_phase_replay[key] != recorded_phase_replay.get(key):
                phase_replay_differences[key] = {
                    "rebuilt": rebuilt_phase_replay[key],
                    "recorded": recorded_phase_replay.get(key),
                }
        if phase_replay_differences:
            raise ScopfError(
                "Global feasibility-cut Phase-I replay changed: "
                f"{phase_replay_differences}"
            )
        rebuilt_cut, _rebuilt_cut_audit = derive_commitment_feasibility_cut(
            master=cut_master,
            phase_model=phase_model,
            phase_certificate=rebuilt_phase_certificate,
            source_commitment=binary,
            replay_tolerance_pu=float(config.model["phase_one_replay_tolerance_pu"]),
        )
        serialized_cut_object = commitment_upper_cut_from_record(
            serialized_cut, source_rows
        )
        if not isinstance(serialized_cut_object, CommitmentFeasibilityCut):
            raise ScopfError("Global feasibility-cut record changed cut type")
        source_violation_difference = abs(
            serialized_cut_object.violation(binary)
            - serialized_cut_object.conservative_source_violation_pu
        )
        if source_violation_difference > replay_tolerance:
            raise ScopfError("Global feasibility-cut source violation did not replay")
        coefficient_difference = (
            serialized_cut_object.coefficients - rebuilt_cut.coefficients
        )
        rhs_difference = serialized_cut_object.rhs - rebuilt_cut.rhs
        # The rebuilt cut already relaxes its raw dual inequality by the
        # registered Phase-I safety margin.  Bound the worst possible binary
        # strengthening caused by cross-architecture FP64 reconstruction; if
        # it fits inside that margin, the serialized inequality is still no
        # stronger than the independently rebuilt raw dual cut.
        worst_portable_strengthening = max(
            0.0,
            float(np.sum(np.maximum(coefficient_difference, 0.0)))
            - float(rhs_difference),
        )
        safety_margin = float(recorded_phase_replay["safety_margin_pu"])
        if worst_portable_strengthening > safety_margin:
            raise ScopfError(
                "Global feasibility cut consumed its numerical safety margin"
            )
        maximum_feasibility_cut_coefficient_replay_difference = max(
            maximum_feasibility_cut_coefficient_replay_difference,
            float(np.max(np.abs(coefficient_difference))),
        )
        maximum_feasibility_cut_rhs_replay_difference = max(
            maximum_feasibility_cut_rhs_replay_difference,
            abs(float(rhs_difference)),
        )
        maximum_feasibility_cut_portable_strengthening = max(
            maximum_feasibility_cut_portable_strengthening,
            worst_portable_strengthening,
        )
        validated_global_feasibility_cuts[cut_id] = serialized_cut_object

    active_master_cache: dict[
        str,
        tuple[
            ReducedMaster,
            tuple[SecurityPair, ...],
            tuple[CommitmentUpperCut, ...],
        ],
    ] = {}
    active_master_cache_builds = 0
    active_master_cache_hits = 0
    for record in payload["frontier_regions"]:
        region_id = str(record["region_id"])
        if region_id in leaves:
            raise ScopfError("Duplicate frontier region id")
        masks = _masks_from_record(record, source_rows)
        equivalence = record.get("security_row_equivalence", {})
        cache_bytes = json.dumps(
            {
                "security_pairs": record["security_pairs"],
                "security_row_equivalence": equivalence,
                "commitment_upper_cuts": record.get("commitment_upper_cuts", []),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        cache_key = hashlib.sha256(cache_bytes).hexdigest()
        cached = active_master_cache.get(cache_key)
        if cached is None:
            master = build_reduced_master(
                case,
                network,
                segments=10,
                coefficient_zero_tolerance=float(
                    config.model.get("reduced_coefficient_zero_tolerance", 1e-14)
                ),
            )
            pairs = tuple(
                security_pair_from_record(
                    pair_record,
                    catalog,
                    lodf_absolute_tolerance=lodf_tolerance,
                )
                for pair_record in record["security_pairs"]
            )
            update_lodf_replay_difference(record["security_pairs"], pairs)
            add_reduced_security_pairs(
                master,
                network,
                pairs,
                expected_representative_by_pair_id=equivalence.get(
                    "representative_by_pair_id"
                ),
                equivalence_replay_tolerance=float(
                    config.model.get("security_equivalence_replay_tolerance", 0.0)
                ),
            )
            commitment_cuts = tuple(
                commitment_upper_cut_from_record(cut_record, source_rows)
                for cut_record in record.get("commitment_upper_cuts", [])
            )
            for cut in commitment_cuts:
                if isinstance(cut, CommitmentFeasibilityCut):
                    validated = validated_global_feasibility_cuts.get(cut.cut_id)
                    if (
                        validated is None
                        or validated.rhs != cut.rhs
                        or not np.array_equal(
                            validated.coefficients, cut.coefficients
                        )
                    ):
                        raise ScopfError(
                            "Frontier certificate uses an unvalidated feasibility cut"
                        )
            add_commitment_upper_cuts(master, commitment_cuts)
            active_master_cache[cache_key] = (master, pairs, commitment_cuts)
            active_master_cache_builds += 1
        else:
            master, pairs, commitment_cuts = cached
            active_master_cache_hits += 1
        if record.get("security_row_equivalence") != _security_row_equivalence_record(master):
            raise ScopfError(f"Independent region {region_id} security-row audit mismatch")
        if len(pairs) != len(record["security_pairs"]):
            raise ScopfError(f"Independent region {region_id} security-pair cache mismatch")

        cleanup_comparison = _replay_cleanup_audit_comparison(
            record["coefficient_cleanup_audit"],
            master.coefficient_cleanup_audit,
        )
        maximum_cleanup_audit_difference = max(
            maximum_cleanup_audit_difference,
            float(cleanup_comparison["maximum_fp64_difference"]),
        )
        maximum_cleanup_audit_integer_difference = max(
            maximum_cleanup_audit_integer_difference,
            int(cleanup_comparison["maximum_integer_difference"]),
        )
        cleanup_audit_integer_differences.extend(
            {"region_id": region_id, **difference}
            for difference in cleanup_comparison["differing_integer_fields"]
        )
        replayed = replay_lagrangian_certificate(
            master,
            record["lagrangian_certificate"],
            masks,
            commitment_cuts_by_id={cut.cut_id: cut for cut in commitment_cuts},
        )
        recorded = float(record["lagrangian_certificate"]["conservative_lower_bound"])
        difference = abs(replayed.conservative_lower_bound - recorded)
        maximum_difference = max(maximum_difference, difference)
        if difference > float(config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]):
            raise ScopfError(f"Independent region {region_id} replay mismatch")
        leaves[region_id] = (
            masks,
            tuple(
                cut
                for cut in commitment_cuts
                if isinstance(cut, CommitmentCardinalityCut)
            ),
        )
        replayed_bounds[region_id] = replayed.conservative_lower_bound
    maximum_phase_one_difference = 0.0
    pruned_records = payload.get("pruned_regions", [])
    for record in pruned_records:
        region_id = str(record["region_id"])
        if region_id in leaves:
            raise ScopfError("Duplicate active/pruned region id")
        masks = _masks_from_record(record, source_rows)
        master = build_reduced_master(
            case,
            network,
            segments=10,
            coefficient_zero_tolerance=float(
                config.model.get("reduced_coefficient_zero_tolerance", 1e-14)
            ),
        )
        pairs = tuple(
            security_pair_from_record(
                pair_record,
                catalog,
                lodf_absolute_tolerance=lodf_tolerance,
            )
            for pair_record in record["security_pairs"]
        )
        update_lodf_replay_difference(record["security_pairs"], pairs)
        equivalence = record.get("security_row_equivalence", {})
        add_reduced_security_pairs(
            master,
            network,
            pairs,
            expected_representative_by_pair_id=equivalence.get("representative_by_pair_id"),
            equivalence_replay_tolerance=float(
                config.model.get("security_equivalence_replay_tolerance", 0.0)
            ),
        )
        fix_commitments(master, masks.fixed_off, masks.fixed_on)
        commitment_cuts = tuple(
            commitment_upper_cut_from_record(cut_record, source_rows)
            for cut_record in record.get("commitment_upper_cuts", [])
        )
        for cut in commitment_cuts:
            if isinstance(cut, CommitmentFeasibilityCut):
                validated = validated_global_feasibility_cuts.get(cut.cut_id)
                if (
                    validated is None
                    or validated.rhs != cut.rhs
                    or not np.array_equal(validated.coefficients, cut.coefficients)
                ):
                    raise ScopfError(
                        "Pruned certificate uses an unvalidated feasibility cut"
                    )
        add_commitment_upper_cuts(master, commitment_cuts)
        cleanup_comparison = _replay_cleanup_audit_comparison(
            record["coefficient_cleanup_audit"],
            master.coefficient_cleanup_audit,
        )
        maximum_cleanup_audit_difference = max(
            maximum_cleanup_audit_difference,
            float(cleanup_comparison["maximum_fp64_difference"]),
        )
        maximum_cleanup_audit_integer_difference = max(
            maximum_cleanup_audit_integer_difference,
            int(cleanup_comparison["maximum_integer_difference"]),
        )
        cleanup_audit_integer_differences.extend(
            {"region_id": region_id, **difference}
            for difference in cleanup_comparison["differing_integer_fields"]
        )
        if record.get("security_row_equivalence") != _security_row_equivalence_record(master):
            raise ScopfError(f"Independent pruned region {region_id} row audit mismatch")
        phase_model = build_phase_one_model(
            master.canonical,
            base_mva=float(case.base_mva),
            maximum_violation_pu=float(config.runtime["phase_one_maximum_violation_pu"]),
        )
        recorded_phase_model = record["phase_one_model"]
        if (
            phase_model.num_columns != int(recorded_phase_model["columns"])
            or phase_model.num_rows != int(recorded_phase_model["rows"])
            or abs(
                float(phase_model.column_upper[-1])
                - float(recorded_phase_model["box_derived_violation_upper_bound_pu"])
            )
            > float(config.model["phase_one_replay_tolerance_pu"])
        ):
            raise ScopfError(f"Independent Phase-I model mismatch for region {region_id}")
        replayed_phase = replay_phase_one_certificate(phase_model, record["phase_one_certificate"])
        recorded_phase_bound = float(record["phase_one_certificate"]["conservative_lower_bound_pu"])
        phase_difference = abs(
            float(replayed_phase["conservative_lower_bound_pu"]) - recorded_phase_bound
        )
        maximum_phase_one_difference = max(maximum_phase_one_difference, phase_difference)
        if not replayed_phase["prune_certified"] or phase_difference > float(
            config.model["phase_one_replay_tolerance_pu"]
        ):
            raise ScopfError(f"Independent Phase-I replay failed for region {region_id}")
        leaves[region_id] = (
            masks,
            tuple(
                cut
                for cut in commitment_cuts
                if isinstance(cut, CommitmentCardinalityCut)
            ),
        )
    cover_passed = verify_cardinality_disjunctive_cover(
        source_rows,
        payload["disjunctive_splits"],
        leaves,
    )
    if not replayed_bounds:
        raise ScopfError("Lagrangian cover has no active feasible frontier region")
    global_bound = min(replayed_bounds.values())
    recorded_global = float(payload["bound"])
    global_difference = abs(global_bound - recorded_global)
    passed = bool(
        cover_passed
        and global_difference <= float(config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"])
    )
    return {
        "passed": passed,
        "certificate_kind": "independent_raw_input_lagrangian_cover_replay_v1",
        "frontier_region_count": len(leaves),
        "active_frontier_region_count": len(replayed_bounds),
        "phase_one_pruned_region_count": len(pruned_records),
        "validated_global_feasibility_cut_count": len(
            validated_global_feasibility_cuts
        ),
        "maximum_feasibility_cut_coefficient_replay_difference": (
            maximum_feasibility_cut_coefficient_replay_difference
        ),
        "maximum_feasibility_cut_rhs_replay_difference": (
            maximum_feasibility_cut_rhs_replay_difference
        ),
        "maximum_feasibility_cut_portable_strengthening": (
            maximum_feasibility_cut_portable_strengthening
        ),
        "disjunctive_cover_passed": cover_passed,
        "replayed_global_lower_bound": global_bound,
        "recorded_global_lower_bound": recorded_global,
        "global_bound_difference_dollars": global_difference,
        "maximum_region_replay_difference_dollars": maximum_difference,
        "maximum_phase_one_replay_difference_pu": maximum_phase_one_difference,
        "maximum_lodf_replay_difference": maximum_lodf_replay_difference,
        "registered_lodf_replay_tolerance": lodf_tolerance,
        "maximum_cleanup_audit_replay_difference": (maximum_cleanup_audit_difference),
        "maximum_cleanup_audit_integer_difference": (maximum_cleanup_audit_integer_difference),
        "cleanup_audit_integer_differences": cleanup_audit_integer_differences,
        "active_master_cache_builds": active_master_cache_builds,
        "active_master_cache_hits": active_master_cache_hits,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _load_cpu_comparison(config: RunConfig, registration: dict[str, Any]) -> dict[str, Any]:
    reference = registration["cpu_comparison"]
    path = (config.root / str(reference["result_file"])).resolve()
    result = _read_json(path)
    canonical_bytes = json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    if canonical_sha256 != str(reference["canonical_json_sha256"]):
        raise ScopfError("Registered laptop comparison canonical JSON hash mismatch")
    expected_suite = str(reference.get("source_suite", "activsg500-gap-sensitivity-v1"))
    if result.get("experiment_suite_id") != expected_suite:
        raise ScopfError("Registered laptop comparison suite identity changed")
    result_identity = result.get("frozen_identity", result)
    if result_identity.get("tag") != reference.get("source_tag"):
        raise ScopfError("Registered laptop comparison tag changed")
    if result_identity.get("commit") != reference.get("source_commit"):
        raise ScopfError("Registered laptop comparison commit changed")
    hashes = result.get("source_hashes", result.get("source_identity", {}))
    if (
        hashes.get("case_sha256") != config.raw["raw_inputs"]["case_sha256"]
        or hashes.get("contingency_sha256") != config.raw["raw_inputs"]["contingency_sha256"]
    ):
        raise ScopfError("Registered laptop comparison raw-input hashes changed")
    matching = [record for record in result.get("summary", []) if record.get("gap_label") == "1e-3"]
    if len(matching) != 1:
        raise ScopfError("Registered laptop comparison lacks one 1e-3 record")
    summary = matching[0]
    if summary.get("raw_result_sha256") != reference.get("raw_result_sha256"):
        raise ScopfError("Registered laptop raw-result identity changed")
    accepted = result.get("accepted_1e-3", {})
    security_violation = summary.get(
        "final_security_violation_pu",
        accepted.get(
            "final_exhaustive_violation_pu",
            summary.get("last_screen_maximum_violation_pu", float("inf")),
        ),
    )
    model_residual = summary.get(
        "final_model_residual_pu",
        accepted.get("maximum_model_residual_pu", float("inf")),
    )
    independent_passed = summary.get("independent_verification_passed", True)
    if (
        not bool(independent_passed)
        or float(security_violation) > 1e-5
        or float(model_residual) > 1e-6
    ):
        raise ScopfError("Registered laptop comparison did not pass its original gates")
    return {
        "source": str(path.relative_to(config.root)),
        "canonical_json_sha256": canonical_sha256,
        "raw_result_sha256": summary["raw_result_sha256"],
        "frozen_identity": {
            "commit": result_identity.get("commit"),
            "tag": result_identity.get("tag"),
        },
        "status": "optimal_verified",
        "objective": summary.get("objective"),
        "bound": summary.get("bound"),
        "mip_gap": summary.get("achieved_mip_gap"),
        "commitment_count": summary.get("commitment_count"),
        "total_wall_time_seconds": summary.get("total_wall_seconds"),
    }


def run_gpu_lagrangian_experiment(
    config: RunConfig,
    *,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    registration = validate_lagrangian_experiment_config(config)
    deadline = Deadline(
        total_seconds=float(config.runtime["deadline_seconds"]),
        verification_reserve_seconds=float(config.runtime["verification_reserve_seconds"]),
        serialization_reserve_seconds=float(config.runtime["serialization_reserve_seconds"]),
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "case_name": config.case_name,
        "benchmark_id": config.benchmark_id,
        "experiment_policy": registration["experiment_policy"],
        "platform": "dgx_spark",
        "status": "running",
        "deadline_seconds": float(config.runtime["deadline_seconds"]),
        "requested_relative_gap": float(config.model["mip_relative_gap_tolerance"]),
        "integer_solver_used": False,
        "integer_lower_bound_solver_used": False,
        "branch_and_bound_performed": False,
        "custom_cuda_kernel_used": False,
        "disjunctive_splits": [],
        "failed_disjunctive_split_attempts": [],
        "pruned_regions": [],
        "phase_one_prechecks": [],
        "phase_one_fallback_attempts": [],
        "parallel_child_batches": [],
        "frontier_regions": [],
        "primal_repairs": [],
        "commitment_feasibility_cuts": [],
        "secure_incumbent_checkpoint_history": [],
        "timings_seconds": {},
    }
    memory = PeakMemorySampler(sample_gpu=True)

    def save() -> None:
        payload["elapsed_seconds"] = deadline.elapsed
        payload["peak_memory"] = {
            "process_rss_bytes": memory.peak_process_rss_bytes,
            "cupy_pool_used_bytes": memory.peak_gpu_pool_used_bytes,
            "cuda_device_memory_delta_bytes": memory.peak_cuda_device_memory_delta_bytes,
        }
        if checkpoint is not None:
            checkpoint(payload)

    def save_region_progress(record: dict[str, Any]) -> None:
        payload["active_region_progress"] = record
        save()

    with memory:
        guard_runtime_environment(config.root)
        validate_platform(config, "dgx_spark")
        payload["environment"] = environment_manifest("dgx_spark")
        if config.benchmark_id in ACTIVSG2000_V8_PLUS_EXPERIMENT_IDS:
            payload["cpu_comparison_policy"] = (
                "deferred_until_after_gpu_acceptance_status_is_final"
            )
        else:
            payload["cpu_comparison"] = _load_cpu_comparison(config, registration)
        save()

        started = time.perf_counter()
        case = read_matpower_case(
            config.case_path,
            expected_sha256=config.raw["raw_inputs"]["case_sha256"],
        )
        table = read_contingency_table(
            config.contingency_path,
            expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
        )
        payload["source_manifest"] = build_source_manifest(case, table)
        payload["timings_seconds"]["raw_input_loading"] = time.perf_counter() - started

        started = time.perf_counter()
        network = build_network(case)
        catalog = build_contingency_catalog(
            case,
            network,
            table,
            validation_columns=int(config.model["lodf_validation_columns"]),
            validation_tolerance_pu=float(config.model["lodf_validation_tolerance_pu"]),
            chunk_columns=int(config.model["lodf_build_chunk_columns"]),
        )
        base_master = build_reduced_master(
            case,
            network,
            segments=10,
            coefficient_zero_tolerance=float(
                config.model.get("reduced_coefficient_zero_tolerance", 1e-14)
            ),
        )
        payload["timings_seconds"]["network_and_reduced_model_build"] = (
            time.perf_counter() - started
        )
        payload["contingencies"] = contingency_catalog_report(catalog)
        payload["pwl_costs"] = pwl_approximation_report(base_master.costs)
        payload["reduced_model_identity"] = {
            "generator_count": int(base_master.index.generator_source_rows.size),
            "base_rows": base_master.canonical.num_rows,
            "base_columns": base_master.canonical.num_columns,
            "network_elimination": "FP64 reference-bus affine angle/flow map",
            "exact_pmin_changed": False,
            "integer_generator_subproblem": "off_or_exact_pmin_plus_ten_segments",
            "coefficient_cleanup": dict(base_master.coefficient_cleanup_audit),
        }
        screener = ContingencyScreener(
            network,
            catalog,
            backend="cupy",
            chunk_columns=int(config.model["screen_chunk_columns"]),
        )
        save()

        source_rows = base_master.index.generator_source_rows
        compact_lagrangian_certificates = bool(
            config.runtime.get("compact_lagrangian_certificates", False)
        )

        def serialize_region(region: SolvedRegion) -> dict[str, Any]:
            return _region_record(
                region,
                compact_lagrangian_certificate=compact_lagrangian_certificates,
            )

        global_pairs: dict[str, SecurityPair] = {}
        frontier: dict[str, SolvedRegion] = {}
        all_region_records: list[dict[str, Any]] = []
        replay_after_every_split = bool(
            config.runtime.get("frontier_replay_after_every_split", True)
        )
        payload["frontier_replay_policy"] = {
            "replay_after_every_split": replay_after_every_split,
            "root_replay_required": True,
            "final_full_replay_required_before_acceptance": True,
            "intermediate_gpu_bounds_are_pending_independent_replay": (
                not replay_after_every_split
            ),
        }

        def persist_region_evidence() -> None:
            payload["all_solved_region_count"] = len(all_region_records)
            payload["solved_region_history"] = list(all_region_records)
            payload["frontier_regions"] = [
                serialize_region(frontier[region_id]) for region_id in sorted(frontier)
            ]
            if frontier:
                payload["bound"] = min(
                    region.lagrangian.conservative_lower_bound for region in frontier.values()
                )
                payload["bound_status"] = "gpu_generated_pending_independent_replay"
                if payload.get("objective") is not None:
                    payload["relative_gap"] = _relative_gap(
                        objective=float(payload["objective"]),
                        lower_bound=float(payload["bound"]),
                    )
            save()

        def replay_and_checkpoint_frontier(stage: str) -> None:
            payload["active_stage"] = stage
            replay = verify_lagrangian_certificate_payload(config, payload)
            if not replay["passed"]:
                raise ScopfError(f"Independent Lagrangian replay failed during {stage}")
            payload["bound"] = replay["replayed_global_lower_bound"]
            payload["bound_status"] = "independently_replayed_current_frontier"
            if payload.get("objective") is not None:
                payload["relative_gap"] = _relative_gap(
                    objective=float(payload["objective"]),
                    lower_bound=float(payload["bound"]),
                )
            payload.setdefault("lagrangian_replay_history", []).append(
                {
                    "stage": stage,
                    **replay,
                }
            )
            save()

        payload["active_stage"] = "root_lagrangian_relaxation"
        root = _solve_region(
            region_id="r",
            masks=RegionMasks.root(source_rows.size),
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=deadline,
            initial_pairs=(),
            screener=screener,
            checkpoint=save,
            progress=save_region_progress,
        )
        payload.pop("active_region_progress", None)
        frontier[root.region_id] = root
        global_pairs.update((pair.pair_id, pair) for pair in root.security_pairs)
        all_region_records.append(serialize_region(root))
        persist_region_evidence()
        replay_and_checkpoint_frontier("independent_root_lagrangian_replay")
        payload["root_lagrangian_verification"] = payload["lagrangian_replay_history"][-1]
        cardinality_subsets = (
            commitment_branch_subsets(root.master)
            if config.benchmark_id
            in {ACTIVSG2000_V10_EXPERIMENT_ID, ACTIVSG2000_V11_EXPERIMENT_ID}
            else ()
        )
        payload["cardinality_refinement_policy"] = (
            {
                "enabled": True,
                "policy": (
                    "most_fractional_exact_type_then_laminar_cardinality_sum_v1"
                ),
                "subset_count": len(cardinality_subsets),
                "exact_type_subset_count": sum(
                    subset.family == "exact_type" for subset in cardinality_subsets
                ),
                "all_branch_coefficients_are_integer_unit_cardinalities": True,
                "minimum_cardinality_subset_size": 2,
                "binary_completeness_fallback_enabled": True,
                "original_binary_cover_preserved": True,
                "phase_one_precheck_enabled": False,
                "cpu_solution_data_used": False,
            }
            if config.benchmark_id
            in {ACTIVSG2000_V10_EXPERIMENT_ID, ACTIVSG2000_V11_EXPERIMENT_ID}
            else {"enabled": False}
        )
        save()

        best_primal: dict[str, Any] | None = None
        global_feasibility_cuts: dict[str, CommitmentFeasibilityCut] = {}
        tried_commitments: set[str] = set()
        last_tried_commitment: np.ndarray | None = None
        pending_primal_candidates: deque[
            tuple[SolvedRegion, np.ndarray, str, dict[str, Any] | None]
        ] = deque()
        candidate_policy = (
            PrimalCandidatePolicy.from_config(config)
            if (
                config.benchmark_id.endswith(
                    (
                        "-v3",
                        "-v4",
                        "-v5",
                        "-v6",
                        "-v7",
                        "-v8",
                        "-v9",
                        "-v10",
                        "-v11",
                    )
                )
                or config.benchmark_id
                in {
                    ACTIVSG2000_EXPERIMENT_ID,
                    ACTIVSG2000_V2_EXPERIMENT_ID,
                    ACTIVSG2000_V3_EXPERIMENT_ID,
                    ACTIVSG2000_V4_EXPERIMENT_ID,
                    ACTIVSG2000_V5_EXPERIMENT_ID,
                    ACTIVSG2000_V6_EXPERIMENT_ID,
                    ACTIVSG2000_V7_EXPERIMENT_ID,
                    ACTIVSG2000_V8_EXPERIMENT_ID,
                    ACTIVSG2000_V9_EXPERIMENT_ID,
                    ACTIVSG2000_V10_EXPERIMENT_ID,
                    ACTIVSG2000_V11_EXPERIMENT_ID,
                }
            )
            else None
        )
        region_attempt_policy = (
            PrimalCandidatePolicy.from_config(config, scope="disjunctive_region")
            if (
                config.benchmark_id.endswith(
                    ("-v4", "-v5", "-v6", "-v7", "-v8", "-v9", "-v10", "-v11")
                )
                or config.benchmark_id
                in {
                    ACTIVSG2000_EXPERIMENT_ID,
                    ACTIVSG2000_V2_EXPERIMENT_ID,
                    ACTIVSG2000_V3_EXPERIMENT_ID,
                    ACTIVSG2000_V4_EXPERIMENT_ID,
                    ACTIVSG2000_V5_EXPERIMENT_ID,
                    ACTIVSG2000_V6_EXPERIMENT_ID,
                    ACTIVSG2000_V7_EXPERIMENT_ID,
                    ACTIVSG2000_V8_EXPERIMENT_ID,
                    ACTIVSG2000_V9_EXPERIMENT_ID,
                    ACTIVSG2000_V10_EXPERIMENT_ID,
                    ACTIVSG2000_V11_EXPERIMENT_ID,
                }
            )
            else None
        )
        phase_one_first = config.benchmark_id.endswith(
            ("-v6", "-v7", "-v8", "-v9")
        ) or config.benchmark_id in {
            ACTIVSG2000_EXPERIMENT_ID,
            ACTIVSG2000_V2_EXPERIMENT_ID,
            ACTIVSG2000_V3_EXPERIMENT_ID,
            ACTIVSG2000_V4_EXPERIMENT_ID,
            ACTIVSG2000_V5_EXPERIMENT_ID,
            ACTIVSG2000_V6_EXPERIMENT_ID,
            ACTIVSG2000_V7_EXPERIMENT_ID,
            ACTIVSG2000_V8_EXPERIMENT_ID,
            ACTIVSG2000_V9_EXPERIMENT_ID,
        }
        payload["primal_candidate_policy"] = (
            candidate_policy.as_dict() if candidate_policy is not None else None
        )
        payload["primal_candidate_queue"] = []
        payload["initial_commitment_candidate_pipeline"] = (
            {
                "enabled": True,
                "order": [
                    "exact_aggregate_pmin_pmax_capacity_repair",
                    "exact_fixed_commitment_projection_precheck",
                    "gpu_pdlp_phase_one",
                    "exhaustive_cupy_contingency_screen",
                    "exact_cost_lp",
                ],
                "network_aware_repair_after_constant_row_rejection": True,
                "candidate_generation_is_not_feasibility_proof": True,
                "exact_source_pmin_pmax_retained": True,
            }
            if config.benchmark_id in ACTIVSG2000_V4_PLUS_EXPERIMENT_IDS
            else {"enabled": False}
        )
        payload["parallel_child_policy"] = (
            {
                "enabled": int(config.runtime["parallel_child_solver_contexts"]) > 1,
                "solver_contexts": int(config.runtime["parallel_child_solver_contexts"]),
                "minimum_frontier_regions": int(
                    config.runtime["parallel_child_minimum_frontier_regions"]
                ),
                "independent_cupy_screeners": True,
                "native_log_capture": (
                    "console_only_for_concurrent_contexts_file_backed_for_sequential_contexts"
                ),
                "sequential_v11_numerical_guard": (
                    config.benchmark_id == ACTIVSG2000_V11_EXPERIMENT_ID
                ),
            }
            if config.benchmark_id in ACTIVSG2000_V4_PLUS_EXPERIMENT_IDS
            else {"enabled": False}
        )
        payload["disjunctive_region_attempt_policy"] = (
            region_attempt_policy.as_dict() if region_attempt_policy is not None else None
        )
        payload["phase_one_first_policy"] = (
            {
                "enabled": True,
                "precheck_time_limit_seconds": float(
                    config.runtime["precheck_phase_one_time_limit_seconds"]
                ),
                "capacity_gate_is_pruning_authority": False,
                "positive_replayable_phase_one_dual_is_pruning_authority": True,
                "zero_or_uncertain_phase_one_proceeds_to_cost_lp": (
                    config.benchmark_id
                    not in {
                        ACTIVSG2000_V6_EXPERIMENT_ID,
                        ACTIVSG2000_V7_EXPERIMENT_ID,
                        ACTIVSG2000_V8_EXPERIMENT_ID,
                        ACTIVSG2000_V9_EXPERIMENT_ID,
                        ACTIVSG2000_V10_EXPERIMENT_ID,
                        ACTIVSG2000_V11_EXPERIMENT_ID,
                    }
                ),
                "phase_one_source_primal_warm_starts_cost_lp": (
                    config.benchmark_id
                    not in {
                        ACTIVSG2000_V6_EXPERIMENT_ID,
                        ACTIVSG2000_V7_EXPERIMENT_ID,
                        ACTIVSG2000_V8_EXPERIMENT_ID,
                        ACTIVSG2000_V9_EXPERIMENT_ID,
                        ACTIVSG2000_V10_EXPERIMENT_ID,
                        ACTIVSG2000_V11_EXPERIMENT_ID,
                    }
                ),
                "v6_child_policy": (
                    "secure_phase_one_point_plus_parent_inherited_lagrangian_bound"
                    if config.benchmark_id
                    in {
                        ACTIVSG2000_V6_EXPERIMENT_ID,
                        ACTIVSG2000_V7_EXPERIMENT_ID,
                        ACTIVSG2000_V8_EXPERIMENT_ID,
                        ACTIVSG2000_V9_EXPERIMENT_ID,
                        ACTIVSG2000_V10_EXPERIMENT_ID,
                        ACTIVSG2000_V11_EXPERIMENT_ID,
                    }
                    else None
                ),
                "phase_one_row_dual_transferred": (
                    config.benchmark_id in ACTIVSG2000_V4_PLUS_EXPERIMENT_IDS
                ),
            }
            if phase_one_first
            else {"enabled": False}
        )
        payload["fixed_commitment_feasibility_polisher"] = (
            {
                "enabled": True,
                "policy": "exact_dispatch_projection_gpu_phase_one_v1",
                "removes_only_local_fixed_u_and_pwl_feasibility_bookkeeping": True,
                "exact_source_pmin_pmax_retained": True,
                "network_and_security_coupling_rows_retained": True,
                "exhaustive_screen_after_every_feasible_resolve": True,
                "secure_dispatch_checkpointed_before_cost_polish": True,
                "cost_polish_uses_lifted_source_space_primal_start": (
                    config.benchmark_id
                    not in {
                        ACTIVSG2000_V7_EXPERIMENT_ID,
                        ACTIVSG2000_V8_EXPERIMENT_ID,
                        ACTIVSG2000_V9_EXPERIMENT_ID,
                        ACTIVSG2000_V10_EXPERIMENT_ID,
                        ACTIVSG2000_V11_EXPERIMENT_ID,
                    }
                ),
                "cost_polish_uses_exact_convex_pwl_epigraph_projection": (
                    config.benchmark_id
                    in {
                        ACTIVSG2000_V7_EXPERIMENT_ID,
                        ACTIVSG2000_V8_EXPERIMENT_ID,
                        ACTIVSG2000_V9_EXPERIMENT_ID,
                        ACTIVSG2000_V10_EXPERIMENT_ID,
                        ACTIVSG2000_V11_EXPERIMENT_ID,
                    }
                ),
                "cost_polish_uses_mapped_native_row_dual_start": (
                    config.benchmark_id in ACTIVSG2000_V4_PLUS_EXPERIMENT_IDS
                    and config.benchmark_id
                    not in {
                        ACTIVSG2000_V7_EXPERIMENT_ID,
                        ACTIVSG2000_V8_EXPERIMENT_ID,
                        ACTIVSG2000_V9_EXPERIMENT_ID,
                        ACTIVSG2000_V10_EXPERIMENT_ID,
                        ACTIVSG2000_V11_EXPERIMENT_ID,
                    }
                ),
                "cpu_commitment_or_dispatch_seeded": False,
            }
            if config.benchmark_id
            in {
                ACTIVSG2000_V2_EXPERIMENT_ID,
                ACTIVSG2000_V3_EXPERIMENT_ID,
                ACTIVSG2000_V4_EXPERIMENT_ID,
                ACTIVSG2000_V5_EXPERIMENT_ID,
                ACTIVSG2000_V6_EXPERIMENT_ID,
                ACTIVSG2000_V7_EXPERIMENT_ID,
                ACTIVSG2000_V8_EXPERIMENT_ID,
                ACTIVSG2000_V9_EXPERIMENT_ID,
                ACTIVSG2000_V10_EXPERIMENT_ID,
                ACTIVSG2000_V11_EXPERIMENT_ID,
            }
            else {"enabled": False}
        )

        def try_primal(
            parent: SolvedRegion,
            proposed: np.ndarray,
            origin: str,
            generation_audit: dict[str, Any] | None = None,
        ) -> bool:
            nonlocal best_primal, last_tried_commitment
            if len(tried_commitments) >= int(config.runtime["maximum_primal_repairs"]):
                payload["primal_candidate_queue"].append(
                    {"origin": origin, "status": "skipped_repair_limit"}
                )
                save()
                return False
            proposed_binary = np.asarray(proposed >= 0.5, dtype=np.int8)
            try:
                candidate = _capacity_repaired_commitment(
                    case,
                    source_rows,
                    proposed_binary,
                    parent.lagrangian.on_subproblem_values,
                    parent.masks,
                    parent.master.operator.total_demand_mw,
                )
            except ScopfError as exc:
                payload["primal_candidate_queue"].append(
                    {
                        "origin": origin,
                        "status": "rejected_capacity_precheck",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                save()
                return False
            digest = hashlib.sha256(candidate.tobytes()).hexdigest()
            if digest in tried_commitments:
                payload["primal_candidate_queue"].append(
                    {
                        "origin": origin,
                        "commitment_sha256": digest,
                        "status": "skipped_duplicate",
                    }
                )
                save()
                return False
            tried_commitments.add(digest)
            payload["active_stage"] = f"fixed_commitment_pdlp_primal_{origin}"
            fixed = RegionMasks(candidate == 0, candidate == 1)
            repair_started = time.perf_counter()
            capacity_changed = candidate != proposed_binary
            if last_tried_commitment is None:
                transition: dict[str, Any] = {
                    "prior_candidate_available": False,
                    "hamming_distance": None,
                    "turned_on_generator_source_rows": [],
                    "turned_off_generator_source_rows": [],
                }
            else:
                turned_on = (candidate == 1) & (last_tried_commitment == 0)
                turned_off = (candidate == 0) & (last_tried_commitment == 1)
                transition = {
                    "prior_candidate_available": True,
                    "hamming_distance": int(np.count_nonzero(candidate != last_tried_commitment)),
                    "turned_on_generator_source_rows": (source_rows[turned_on] + 1).tolist(),
                    "turned_off_generator_source_rows": (source_rows[turned_off] + 1).tolist(),
                }
            attempt: dict[str, Any] = {
                "origin": origin,
                "commitment_sha256": digest,
                "commitment_count": int(np.count_nonzero(candidate)),
                "committed_generator_source_rows": (source_rows[candidate == 1] + 1).tolist(),
                "candidate_policy": (
                    candidate_policy.as_dict() if candidate_policy is not None else None
                ),
                "capacity_repair": {
                    "proposed_commitment_count": int(np.count_nonzero(proposed_binary)),
                    "hamming_distance": int(np.count_nonzero(capacity_changed)),
                    "turned_on_generator_source_rows": (
                        source_rows[capacity_changed & (candidate == 1)] + 1
                    ).tolist(),
                    "turned_off_generator_source_rows": (
                        source_rows[capacity_changed & (candidate == 0)] + 1
                    ).tolist(),
                },
                "transition_from_prior_candidate": transition,
                "status": "running",
            }
            if generation_audit is not None:
                attempt["network_aware_generation"] = generation_audit
            last_tried_commitment = candidate.copy()
            payload["primal_candidate_queue"].append(attempt)
            payload["primal_repairs"].append(attempt)
            save()

            def save_candidate_progress(record: dict[str, Any]) -> None:
                attempt["solver_progress"] = record
                save_region_progress(record)

            def retain_secure_primal(
                *,
                source_master: ReducedMaster,
                source_values: np.ndarray,
                status: str,
                pricing: dict[str, Any],
                final_screen: dict[str, Any],
            ) -> float:
                nonlocal best_primal
                full, full_values = reconstruct_full_values(
                    case,
                    network,
                    source_master,
                    source_values,
                    exact_commitment=candidate,
                )
                objective = float(np.asarray(full.canonical.objective) @ full_values)
                candidate_payload = {
                    "case_name": config.case_name,
                    "objective": objective,
                    "solution": serialize_solution(full_values, case, network, full),
                }
                canonical_residual = full.canonical.max_row_violation(full_values) / case.base_mva
                attempt.update(
                    {
                        "status": status,
                        "objective": objective,
                        "canonical_model_residual_pu": canonical_residual,
                        "final_screen": final_screen,
                        "wall_time_seconds": time.perf_counter() - repair_started,
                    }
                )
                if canonical_residual > float(config.model["model_residual_tolerance_pu"]):
                    raise ScopfError(
                        "Fixed-commitment feasibility lift exceeded the canonical "
                        "model-residual tolerance"
                    )
                if best_primal is None or objective < float(best_primal["objective"]):
                    best_primal = {
                        **candidate_payload,
                        "commitment": candidate.copy(),
                        "pricing": pricing,
                        "_source_master": source_master,
                        "_source_values": np.asarray(
                            source_values, dtype=np.float64
                        ).copy(),
                    }
                    payload["solution"] = candidate_payload["solution"]
                    payload["objective"] = objective
                    payload["commitment_count"] = int(np.count_nonzero(candidate))
                    payload["pricing"] = pricing
                    payload["secure_incumbent_checkpoint"] = {
                        "origin": origin,
                        "commitment_sha256": digest,
                        "objective": objective,
                        "commitment_count": int(np.count_nonzero(candidate)),
                        "solution": candidate_payload["solution"],
                        "pricing": pricing,
                        "incumbent_source": status,
                        "status": "serialized_pending_independent_verification",
                    }
                    save()
                    incumbent_verification = verify_serialized_solution(config, payload)
                    verification_record = incumbent_verification.as_dict()
                    payload["secure_incumbent_checkpoint"]["verification"] = verification_record
                    payload["secure_incumbent_checkpoint"]["status"] = (
                        "independently_verified"
                        if incumbent_verification.passed
                        else "failed_independent_verification"
                    )
                    payload["secure_incumbent_checkpoint_history"].append(
                        payload["secure_incumbent_checkpoint"]
                    )
                    attempt["immediate_independent_verification"] = verification_record
                    save()
                    if not incumbent_verification.passed:
                        raise ScopfError(
                            "Secure incumbent failed immediate independent verification"
                        )
                return objective

            projected_secure_primal = False
            feasibility: FixedCommitmentFeasibilityResult | None = None
            solved: SolvedRegion | None = None
            projected_cost: FixedCommitmentCostResult | None = None
            try:
                if config.benchmark_id in {
                    ACTIVSG2000_V2_EXPERIMENT_ID,
                    ACTIVSG2000_V3_EXPERIMENT_ID,
                    ACTIVSG2000_V4_EXPERIMENT_ID,
                    ACTIVSG2000_V5_EXPERIMENT_ID,
                    ACTIVSG2000_V6_EXPERIMENT_ID,
                    ACTIVSG2000_V7_EXPERIMENT_ID,
                    ACTIVSG2000_V8_EXPERIMENT_ID,
                    ACTIVSG2000_V9_EXPERIMENT_ID,
                    ACTIVSG2000_V10_EXPERIMENT_ID,
                    ACTIVSG2000_V11_EXPERIMENT_ID,
                }:
                    if candidate_policy is None:
                        raise ScopfError("ACTIVSg2000 v2 requires a bounded candidate policy")
                    attempt["feasibility_polisher"] = {
                        "status": "running",
                        "policy": (
                            "exact_dispatch_projection_gpu_phase_one_then_exhaustive_cupy_screen_v1"
                        ),
                        "cpu_commitment_or_dispatch_seeded": False,
                    }
                    feasibility = _solve_fixed_commitment_feasibility(
                        region_id=f"p{len(tried_commitments)}_feas",
                        commitment=candidate,
                        case=case,
                        network=network,
                        catalog=catalog,
                        config=config,
                        deadline=deadline,
                        initial_pairs=tuple(sorted(global_pairs.values())),
                        screener=screener,
                        checkpoint=save,
                        progress=save_candidate_progress,
                        policy=candidate_policy,
                    )
                    payload.pop("active_region_progress", None)
                    global_pairs.update((pair.pair_id, pair) for pair in feasibility.security_pairs)
                    attempt["feasibility_polisher"].update(
                        {
                            "status": "secure_exhaustive_screen",
                            "constraint_generation_rounds": feasibility.rounds,
                            "final_screen": feasibility.final_screen,
                        }
                    )
                    feasibility_pricing = {
                        "status": "unavailable_feasibility_only",
                        "definition": (
                            "The dispatch-only Phase-I feasibility model has no "
                            "production-cost price dual; pricing awaits the exact "
                            "fixed-commitment cost LP"
                        ),
                        "bus_prices": [],
                    }
                    retain_secure_primal(
                        source_master=feasibility.master,
                        source_values=feasibility.source_values,
                        status=("secure_fixed_commitment_projected_phase_one_pending_cost_polish"),
                        pricing=feasibility_pricing,
                        final_screen=feasibility.final_screen,
                    )
                    projected_secure_primal = True
                    if config.benchmark_id in {
                        ACTIVSG2000_V7_EXPERIMENT_ID,
                        ACTIVSG2000_V8_EXPERIMENT_ID,
                        ACTIVSG2000_V9_EXPERIMENT_ID,
                        ACTIVSG2000_V10_EXPERIMENT_ID,
                        ACTIVSG2000_V11_EXPERIMENT_ID,
                    }:
                        attempt["cost_polish"] = {
                            "status": "running",
                            "formulation": (
                                "exact_fixed_commitment_convex_pwl_cost_epigraph_v1"
                            ),
                            "warm_start_source": (
                                "independently_verified_projected_phase_one_dispatch"
                            ),
                            "phase_one_native_row_dual_available_count": (
                                None
                                if feasibility.source_native_row_dual is None
                                else int(feasibility.source_native_row_dual.size)
                            ),
                            "native_row_dual_transferred": False,
                            "initial_dual_policy": (
                                "phase_one_primal_only_different_objective_v1"
                            ),
                        }
                        save()
                        projected_cost = _solve_fixed_commitment_cost_projection(
                            region_id=f"p{len(tried_commitments)}_cost",
                            commitment=candidate,
                            case=case,
                            network=network,
                            catalog=catalog,
                            config=config,
                            deadline=deadline,
                            prepared_master=feasibility.master,
                            initial_source_values=feasibility.source_values,
                            initial_pairs=feasibility.security_pairs,
                            screener=screener,
                            checkpoint=save,
                            progress=save_candidate_progress,
                            policy=candidate_policy,
                        )
                    else:
                        source_column_scale, _ = native_scaling_vectors(
                            feasibility.master.canonical,
                            mode=str(
                                config.raw["platforms"]["dgx_spark"][
                                    "native_scaling_mode"
                                ]
                            ),
                            base_mva=float(case.base_mva),
                        )
                        initial_native_primal = (
                            feasibility.source_values / source_column_scale
                        )
                        cost_initial_native_dual = feasibility.source_native_row_dual
                        attempt["cost_polish"] = {
                            "status": "running",
                            "warm_start_source": (
                                "independently_verified_projected_phase_one_dispatch"
                            ),
                            "native_primal_count": int(initial_native_primal.size),
                            "native_primal_sha256": hashlib.sha256(
                                initial_native_primal.tobytes()
                            ).hexdigest(),
                            "phase_one_native_row_dual_available_count": (
                                None
                                if feasibility.source_native_row_dual is None
                                else int(feasibility.source_native_row_dual.size)
                            ),
                            "native_row_dual_submitted_count": (
                                None
                                if cost_initial_native_dual is None
                                else int(cost_initial_native_dual.size)
                            ),
                            "native_row_dual_transferred": (
                                cost_initial_native_dual is not None
                            ),
                            "initial_dual_policy": "mapped_phase_one_native_row_dual_v1",
                        }
                        save()
                        solved = _solve_region(
                            region_id=f"p{len(tried_commitments)}_cost",
                            masks=fixed,
                            case=case,
                            network=network,
                            catalog=catalog,
                            config=config,
                            deadline=deadline,
                            initial_pairs=feasibility.security_pairs,
                            screener=screener,
                            checkpoint=save,
                            progress=save_candidate_progress,
                            candidate_policy=candidate_policy,
                            prepared_master=feasibility.master,
                            initial_native_primal=initial_native_primal,
                            initial_native_row_dual=cost_initial_native_dual,
                            initial_warm_start_origin=(
                                "projected_phase_one_secure_dispatch_primal_dual_v2"
                                if feasibility.source_native_row_dual is not None
                                else "projected_phase_one_secure_dispatch_v1"
                            ),
                        )
                else:
                    solved = _solve_region(
                        region_id=f"p{len(tried_commitments)}",
                        masks=fixed,
                        case=case,
                        network=network,
                        catalog=catalog,
                        config=config,
                        deadline=deadline,
                        initial_pairs=tuple(sorted(global_pairs.values())),
                        screener=screener,
                        checkpoint=save,
                        progress=save_candidate_progress,
                        candidate_policy=candidate_policy,
                    )
            except DeadlineExceeded:
                attempt.update(
                    {
                        "status": (
                            "secure_projected_phase_one_cost_polish_global_deadline_exceeded"
                            if projected_secure_primal
                            else "global_deadline_exceeded"
                        ),
                        "wall_time_seconds": time.perf_counter() - repair_started,
                    }
                )
                save()
                raise
            except PrimalCandidateRejected as exc:
                failed_progress = payload.pop("active_region_progress", None)
                if projected_secure_primal:
                    attempt.update(
                        {
                            "status": ("secure_projected_phase_one_cost_polish_rejected"),
                            "cost_polish": {
                                **attempt.get("cost_polish", {}),
                                "status": "rejected_secure_feasibility_retained",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "solver_progress": failed_progress,
                            },
                            "wall_time_seconds": time.perf_counter() - repair_started,
                        }
                    )
                    save()
                    return True
                queued_repairs: list[NetworkCommitmentRepair] = []
                cut_repairs: list[tuple[np.ndarray, dict[str, Any]]] = []
                if (
                    config.benchmark_id in ACTIVSG2000_V8_PLUS_EXPERIMENT_IDS
                    and isinstance(exc, RegionAttemptRejected)
                    and exc.commitment_feasibility_cut is not None
                    and exc.commitment_feasibility_cut_record is not None
                ):
                    cut = exc.commitment_feasibility_cut
                    if not any(
                        record.get("cut", {}).get("cut_id") == cut.cut_id
                        for record in payload["commitment_feasibility_cuts"]
                    ):
                        payload["commitment_feasibility_cuts"].append(
                            exc.commitment_feasibility_cut_record
                        )
                    global_feasibility_cuts[cut.cut_id] = cut
                    cut_repairs = generate_commitment_cut_repairs(
                        cut=cut,
                        commitment=candidate,
                        fixed_off=parent.masks.fixed_off,
                        fixed_on=parent.masks.fixed_on,
                        pmin_mw=case.gen[source_rows, PMIN],
                        pmax_mw=case.gen[source_rows, PMAX],
                        demand_mw=parent.master.operator.total_demand_mw,
                        economic_on_values=parent.lagrangian.on_subproblem_values,
                        maximum_repairs=int(
                            config.runtime[
                                "maximum_commitment_cut_repairs_per_rejection"
                            ]
                        ),
                    )
                    for repair_index, (repair_commitment, repair_audit) in reversed(
                        list(enumerate(cut_repairs, start=1))
                    ):
                        flipped = np.asarray(
                            repair_audit.pop("flipped_generator_positions"),
                            dtype=np.int64,
                        )
                        repair_audit["flipped_generator_source_rows"] = (
                            source_rows[flipped] + 1
                        ).tolist()
                        pending_primal_candidates.appendleft(
                            (
                                parent,
                                repair_commitment,
                                f"phase_one_cut_repair_{repair_index}_after_{origin}",
                                repair_audit,
                            )
                        )
                if (
                    config.benchmark_id in ACTIVSG2000_V4_PLUS_EXPERIMENT_IDS
                    and isinstance(exc, RegionAttemptRejected)
                    and exc.reason == "projected_constant_coupling_row_violation"
                    and exc.rounds
                ):
                    projection_precheck = exc.rounds[-1].get("projection_precheck", {})
                    violated_row_name = projection_precheck.get("row_name")
                    if isinstance(violated_row_name, str):
                        queued_repairs = _network_feasible_commitment_repairs(
                            case=case,
                            master=exc.master,
                            commitment=candidate,
                            masks=parent.masks,
                            violated_row_name=violated_row_name,
                            demand_mw=parent.master.operator.total_demand_mw,
                            on_values=parent.lagrangian.on_subproblem_values,
                            maximum_repairs=int(
                                config.runtime["maximum_network_repair_candidates_per_rejection"]
                            ),
                            pair_search_limit=int(
                                config.runtime["network_repair_pair_search_limit"]
                            ),
                            tolerance_mw=float(config.model["model_residual_tolerance_pu"])
                            * float(case.base_mva),
                        )
                        for repair_index, repair in reversed(
                            list(enumerate(queued_repairs, start=1))
                        ):
                            pending_primal_candidates.appendleft(
                                (
                                    parent,
                                    repair.commitment,
                                    f"network_row_repair_{repair_index}_after_{origin}",
                                    repair.audit,
                                )
                            )
                attempt.update(
                    {
                        "status": "rejected",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "wall_time_seconds": time.perf_counter() - repair_started,
                        "solver_progress": failed_progress,
                        "network_aware_repair_candidates_queued": len(queued_repairs),
                        "phase_one_cut_repair_candidates_queued": len(cut_repairs),
                    }
                )
                save()
                return False
            payload.pop("active_region_progress", None)
            if projected_cost is not None:
                result_master = projected_cost.master
                result_values = projected_cost.source_values
                result_dual = projected_cost.canonical_row_dual
                result_pairs = projected_cost.security_pairs
                result_rounds = projected_cost.rounds
                result_screen = projected_cost.final_screen
                price_status = (
                    "gpu_pdlp_fixed_commitment_exact_cost_epigraph_dual_certified"
                    if projected_cost.pricing_certified
                    else "gpu_pdlp_dual_candidate_uncertified_for_repaired_dispatch"
                )
                price_termination = projected_cost.projected_solve.status
                price_audit = projected_cost.pricing_audit
            elif solved is not None:
                if solved.solve.values is None:
                    raise ScopfError("Fixed-commitment PDLP lost its primal vector")
                result_master = solved.master
                result_values = solved.solve.values
                result_dual = solved.canonical_row_dual
                result_pairs = solved.security_pairs
                result_rounds = solved.rounds
                result_screen = solved.final_screen
                price_status = "gpu_pdlp_fixed_commitment_dual"
                price_termination = solved.solve.status
                price_audit = {
                    "certified_for_returned_dispatch": bool(solved.solve.optimal),
                    "source": "fixed_commitment_cost_pdlp_row_dual",
                }
            else:
                raise ScopfError("Fixed-commitment cost LP result is missing")
            global_pairs.update((pair.pair_id, pair) for pair in result_pairs)
            prices = bus_prices_from_coupling_duals(result_master, result_dual)
            pricing = {
                "status": price_status,
                "definition": (
                    "Demand-derivative fixed-commitment LP dual values; they are "
                    "not MILP duals and are explicitly marked uncertified when "
                    "the returned dispatch required numerical feasibility repair"
                ),
                "pdlp_termination_status": price_termination,
                "audit": price_audit,
                "bus_prices": [
                    {
                        "bus": int(bus),
                        "price_per_mwh": float(price),
                        "price_per_pu_hour": float(price * case.base_mva),
                    }
                    for bus, price in zip(network.bus_ids, prices, strict=True)
                ],
            }
            attempt["constraint_generation_rounds"] = result_rounds
            if projected_secure_primal:
                attempt["cost_polish"] = {
                    **attempt.get("cost_polish", {}),
                    "status": "secure_cost_lp_complete",
                    "constraint_generation_rounds": result_rounds,
                    "final_screen": result_screen,
                }
            retain_secure_primal(
                source_master=result_master,
                source_values=result_values,
                status="secure_fixed_commitment_cost_pdlp",
                pricing=pricing,
                final_screen=result_screen,
            )
            save()
            return True

        def drain_primal_candidate_queue(*, stop_after_first_secure: bool = True) -> None:
            while pending_primal_candidates and (
                not stop_after_first_secure or best_primal is None
            ):
                parent, proposed, origin, generation_audit = pending_primal_candidates.popleft()
                try_primal(parent, proposed, origin, generation_audit)

        target_gap = float(config.model["mip_relative_gap_tolerance"])
        gpu_heuristic_candidate: tuple[np.ndarray, str, dict[str, Any]] | None = None
        if config.benchmark_id in {
            ACTIVSG2000_V10_EXPERIMENT_ID,
            ACTIVSG2000_V11_EXPERIMENT_ID,
        }:
            grouped_candidate, grouped_audit = exact_type_group_rounding(
                root.master, root.commitment
            )
            pending_primal_candidates.append(
                (
                    root,
                    grouped_candidate,
                    "root_pdlp_exact_type_group_rounding",
                    grouped_audit,
                )
            )
            drain_primal_candidate_queue()
        if (
            best_primal is None
            or bool(config.runtime.get("always_run_gpu_primal_heuristics", False))
        ) and config.benchmark_id in {
            ACTIVSG2000_V6_EXPERIMENT_ID,
            ACTIVSG2000_V7_EXPERIMENT_ID,
            ACTIVSG2000_V8_EXPERIMENT_ID,
            ACTIVSG2000_V9_EXPERIMENT_ID,
            ACTIVSG2000_V10_EXPERIMENT_ID,
            ACTIVSG2000_V11_EXPERIMENT_ID,
        }:
            heuristic_pipeline_started = time.perf_counter()
            payload["gpu_primal_heuristics"] = {
                "status": "running",
                "policy": (
                    "short_reduced_gpu_seed_then_sparse_full_gpu_improvement_with_"
                    "independently_verified_complete_gpu_start_v2"
                ),
                "mip_heuristics_only": True,
                "dual_bound_used": False,
                "branch_and_bound_requested": False,
                "partial_mip_start_policy": (
                    "unextended_commitments_are_never_submitted_to_cuopt_26_6"
                ),
                "stages": {},
            }
            payload["integer_solver_used"] = True
            save()

            def heuristic_stage_record(
                result: Any,
                *,
                adapter_wall_time_seconds: float,
                formulation: str,
            ) -> dict[str, Any]:
                return {
                    "formulation": formulation,
                    "status": result.status,
                    "has_incumbent": result.has_incumbent,
                    "objective": result.objective,
                    "native_solve_time_seconds": result.solve_time_seconds,
                    "adapter_wall_time_seconds": adapter_wall_time_seconds,
                    "native_residuals": {
                        name: result.statistics.get(name)
                        for name in (
                            "max_constraint_violation",
                            "max_int_violation",
                            "max_variable_bound_violation",
                        )
                    },
                    "native_log_audit": result.statistics.get("native_log_audit"),
                    "incumbent_commitment_trace": result.statistics.get(
                        "incumbent_commitment_trace"
                    ),
                    "trace_bound_fields_are_diagnostic_only": True,
                    "reported_solver_bound_used": False,
                    "reported_solver_gap_used": False,
                }

            payload["active_stage"] = "gpu_heuristics_reduced_seed"
            deadline.require("cuOpt reduced GPU heuristics seed")
            seed_budget = min(
                deadline.solver_budget(),
                float(config.runtime["gpu_primal_seed_seconds"]),
            )
            seed_started = time.perf_counter()
            seed_result = solve_cuopt(
                root.master.canonical,
                time_limit_seconds=seed_budget,
                mip_relative_gap=target_gap,
                threads=int(config.raw["platforms"]["dgx_spark"]["solver_threads"]),
                native_scaling_mode=str(
                    config.raw["platforms"]["dgx_spark"]["native_scaling_mode"]
                ),
                native_base_mva=float(case.base_mva),
                log_to_console=True,
                track_incumbent_commitments=True,
                mip_certificate_residual_tolerance=float(
                    config.model["model_residual_tolerance_pu"]
                ),
                mip_heuristics_only=True,
            )
            seed_wall = time.perf_counter() - seed_started
            seed_record = heuristic_stage_record(
                seed_result,
                adapter_wall_time_seconds=seed_wall,
                formulation="reduced_affine_network_dc_scopf_v1",
            )
            seed_record["model"] = {
                "columns": root.master.canonical.num_columns,
                "rows": root.master.canonical.num_rows,
                "nonzeros": int(root.master.canonical.matrix_csr().nnz),
                "security_pair_count": len(root.security_pairs),
                "coefficient_cleanup": dict(
                    root.master.coefficient_cleanup_audit
                )
            }
            payload["gpu_primal_heuristics"]["stages"]["reduced_seed"] = (
                seed_record
            )
            seed_values = (
                None
                if seed_result.values is None
                else np.asarray(seed_result.values, dtype=np.float64)
            )
            seed_commitment: np.ndarray | None = None
            seed_start_values: np.ndarray | None = None
            if seed_values is not None:
                seed_residual_pu = (
                    root.master.canonical.max_row_violation(seed_values)
                    / float(case.base_mva)
                )
                seed_commitment_values = commitment_vector(root.master, seed_values)
                rounded_seed_commitment = np.rint(seed_commitment_values)
                seed_integrality_error = float(
                    np.max(
                        np.abs(seed_commitment_values - rounded_seed_commitment)
                    )
                )
                seed_record.update(
                    {
                        "canonical_model_residual_pu": seed_residual_pu,
                        "maximum_commitment_integrality_error": (
                            seed_integrality_error
                        ),
                        "commitment_count": int(
                            np.count_nonzero(rounded_seed_commitment)
                        ),
                    }
                )
                if (
                    seed_integrality_error <= 1e-5
                    and np.all(
                        (rounded_seed_commitment >= 0.0)
                        & (rounded_seed_commitment <= 1.0)
                    )
                ):
                    seed_commitment = rounded_seed_commitment.astype(np.float64)
            save()

            payload["active_stage"] = "gpu_heuristics_sparse_full_model_build"
            sparse_build_started = time.perf_counter()
            secure_gpu_start_used = bool(
                config.benchmark_id == ACTIVSG2000_V11_EXPERIMENT_ID
                and best_primal is not None
            )
            if secure_gpu_start_used:
                assert best_primal is not None
                heuristic_master, seed_start_values = reconstruct_full_values(
                    case,
                    network,
                    best_primal["_source_master"],
                    best_primal["_source_values"],
                    exact_commitment=best_primal["commitment"],
                )
            elif seed_values is not None and seed_commitment is not None:
                heuristic_master, seed_start_values = reconstruct_full_values(
                    case,
                    network,
                    root.master,
                    seed_values,
                    exact_commitment=seed_commitment,
                )
            else:
                heuristic_master = build_master(
                    case,
                    network,
                    segments=int(config.model["pwl_segments"]),
                )
            if not np.array_equal(
                heuristic_master.index.generator_source_rows,
                source_rows,
            ):
                raise ScopfError("Sparse heuristic master changed generator-row identity")
            add_security_pairs(
                heuristic_master.canonical,
                heuristic_master.index,
                network,
                tuple(sorted(global_pairs.values())),
            )
            redundant_bounds = derive_rate_a_angle_bounds(
                network,
                heuristic_master.index.theta_by_bus,
                total_columns=heuristic_master.canonical.num_columns,
            )
            _, original_lower, original_upper, _ = (
                heuristic_master.canonical.column_arrays()
            )
            tightened_lower = np.maximum(original_lower, redundant_bounds.lower)
            tightened_upper = np.minimum(original_upper, redundant_bounds.upper)
            if np.any(tightened_lower > tightened_upper):
                raise ScopfError("Redundant heuristic bounds conflict with canonical bounds")
            tightened_columns = np.flatnonzero(
                (tightened_lower > original_lower)
                | (tightened_upper < original_upper)
            )
            if any(
                not heuristic_master.canonical.variable_names[int(column)].startswith(
                    "theta_"
                )
                for column in tightened_columns
            ):
                raise ScopfError("Heuristic conditioning tightened a non-angle column")
            for column in tightened_columns:
                position = int(column)
                heuristic_master.canonical.column_lower[position] = float(
                    tightened_lower[position]
                )
                heuristic_master.canonical.column_upper[position] = float(
                    tightened_upper[position]
                )
            heuristic_model_record: dict[str, Any] = {
                "formulation": "sparse_full_nodal_dc_scopf_v1",
                "columns": heuristic_master.canonical.num_columns,
                "rows": heuristic_master.canonical.num_rows,
                "nonzeros": int(heuristic_master.canonical.matrix_csr().nnz),
                "security_pair_count": len(global_pairs),
                "complete_start_source": (
                    "independently_verified_gpu_secure_incumbent"
                    if secure_gpu_start_used
                    else "reduced_gpu_heuristic_seed"
                    if seed_start_values is not None
                    else None
                ),
                "cpu_solution_data_used": False,
                "redundant_angle_bounds": {
                    **redundant_bounds.audit,
                    "tightened_column_count": int(tightened_columns.size),
                    "mathematical_feasible_set_changed": False,
                },
                "build_wall_time_seconds": time.perf_counter()
                - sparse_build_started,
            }
            start_audit: dict[str, Any] | None = None
            if seed_start_values is not None:
                column_scale, row_scale = native_scaling_vectors(
                    heuristic_master.canonical,
                    mode=str(
                        config.raw["platforms"]["dgx_spark"][
                            "native_scaling_mode"
                        ]
                    ),
                    base_mva=float(case.base_mva),
                )
                try:
                    start_audit = validate_full_mip_start_feasibility(
                        heuristic_master.canonical,
                        seed_start_values,
                        column_scale=column_scale,
                        row_scale=row_scale,
                        tolerance=float(
                            config.model["model_residual_tolerance_pu"]
                        ),
                    )
                except ScopfError as exc:
                    heuristic_model_record["complete_start_rejection"] = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "native_solver_called": False,
                    }
                    if secure_gpu_start_used:
                        raise ScopfError(
                            "Independently verified GPU incumbent failed the exact full "
                            "MIP-start dimensional/feasibility contract"
                        ) from exc
                    seed_start_values = None
            heuristic_model_record["complete_start_precheck"] = start_audit
            payload["gpu_primal_heuristics"]["sparse_full_model"] = (
                heuristic_model_record
            )
            save()

            total_heuristic_budget = float(
                config.runtime["gpu_primal_heuristics_seconds"]
            )
            heuristic_remaining = total_heuristic_budget - (
                time.perf_counter() - heuristic_pipeline_started
            )
            deadline.require("cuOpt sparse full GPU heuristics improvement")
            if heuristic_remaining <= 1.0:
                raise ScopfError("GPU primal seed exhausted the full heuristic budget")
            full_budget = min(deadline.solver_budget(), heuristic_remaining)
            payload["active_stage"] = "gpu_heuristics_sparse_full_improvement"
            full_started = time.perf_counter()
            full_result = solve_cuopt(
                heuristic_master.canonical,
                time_limit_seconds=full_budget,
                mip_relative_gap=target_gap,
                threads=int(config.raw["platforms"]["dgx_spark"]["solver_threads"]),
                mip_start_values=seed_start_values,
                mip_start_mode=FULL_MIP_START,
                native_scaling_mode=str(
                    config.raw["platforms"]["dgx_spark"]["native_scaling_mode"]
                ),
                native_base_mva=float(case.base_mva),
                log_to_console=True,
                track_incumbent_commitments=True,
                mip_certificate_residual_tolerance=float(
                    config.model["model_residual_tolerance_pu"]
                ),
                mip_heuristics_only=True,
            )
            full_wall = time.perf_counter() - full_started
            full_record = heuristic_stage_record(
                full_result,
                adapter_wall_time_seconds=full_wall,
                formulation="sparse_full_nodal_dc_scopf_v1",
            )
            full_record["complete_gpu_feasible_start_submitted"] = (
                seed_start_values is not None
            )
            full_record["mip_start_contract"] = full_result.statistics.get(
                "mip_start_native_contract"
            )
            payload["gpu_primal_heuristics"]["stages"][
                "sparse_full_improvement"
            ] = full_record

            candidate_values: np.ndarray | None = None
            candidate_master: Any = None
            candidate_formulation: str | None = None
            candidate_objective: float | None = None
            candidate_status: str | None = None
            if full_result.values is not None:
                candidate_values = np.asarray(full_result.values, dtype=np.float64)
                candidate_master = heuristic_master
                candidate_formulation = "sparse_full_nodal_dc_scopf_v1"
                candidate_objective = full_result.objective
                candidate_status = full_result.status
                candidate_commitment_values = np.asarray(
                    [
                        candidate_values[
                            heuristic_master.index.commitment_by_generator[
                                int(source_row)
                            ]
                        ]
                        for source_row in source_rows
                    ],
                    dtype=np.float64,
                )
                candidate_flow = candidate_values[
                    heuristic_master.index.flow_by_active_branch
                ]
            elif seed_values is not None and seed_commitment is not None:
                candidate_values = seed_values
                candidate_master = root.master
                candidate_formulation = "reduced_affine_network_dc_scopf_v1"
                candidate_objective = seed_result.objective
                candidate_status = seed_result.status
                candidate_commitment_values = commitment_vector(
                    root.master, seed_values
                )
                candidate_flow = root.master.operator.flows(
                    reduced_dispatch(root.master, seed_values)
                )
            if candidate_values is not None:
                rounded_commitment = np.rint(candidate_commitment_values)
                maximum_integrality_error = float(
                    np.max(
                        np.abs(candidate_commitment_values - rounded_commitment)
                    )
                )
                canonical_residual_pu = (
                    candidate_master.canonical.max_row_violation(candidate_values)
                    / float(case.base_mva)
                )
                payload["gpu_primal_heuristics"].update(
                    {
                        "status": candidate_status,
                        "candidate_formulation": candidate_formulation,
                        "candidate_objective": candidate_objective,
                        "canonical_model_residual_pu": canonical_residual_pu,
                        "maximum_commitment_integrality_error": (
                            maximum_integrality_error
                        ),
                        "commitment_count": int(
                            np.count_nonzero(rounded_commitment)
                        ),
                    }
                )
                if (
                    maximum_integrality_error <= 1e-5
                    and np.all(
                        (rounded_commitment >= 0.0)
                        & (rounded_commitment <= 1.0)
                    )
                ):
                    heuristic_screen_started = time.perf_counter()
                    heuristic_screen = screener.screen(
                        candidate_flow,
                        tolerance_pu=float(
                            config.model["security_violation_tolerance_pu"]
                        ),
                        already_added=set(global_pairs),
                    )
                    global_pairs.update(
                        (pair.pair_id, pair)
                        for pair in heuristic_screen.violations
                    )
                    payload["gpu_primal_heuristics"][
                        "exhaustive_candidate_screen"
                    ] = {
                        "wall_time_seconds": time.perf_counter()
                        - heuristic_screen_started,
                        "evaluated_sides": heuristic_screen.evaluated_pairs,
                        "new_violated_pairs": len(heuristic_screen.violations),
                        "maximum_violation_pu": (
                            heuristic_screen.maximum_violation_pu
                        ),
                        "maximum_pair_id": heuristic_screen.maximum_pair_id,
                        "candidate_generation_only": True,
                    }
                    gpu_heuristic_candidate = (
                        rounded_commitment.astype(np.float64),
                        "root_cuopt_gpu_heuristics_pipeline",
                        {
                            "solver_status": candidate_status,
                            "heuristic_objective": candidate_objective,
                            "commitment_count": int(
                                np.count_nonzero(rounded_commitment)
                            ),
                            "canonical_model_residual_pu": canonical_residual_pu,
                            "dual_bound_used": False,
                            "complete_gpu_feasible_mip_start_used": (
                                seed_start_values is not None
                            ),
                        },
                    )
                    payload["gpu_primal_heuristics"]["candidate_enqueued"] = True
                else:
                    payload["gpu_primal_heuristics"]["candidate_enqueued"] = False
                    payload["gpu_primal_heuristics"][
                        "candidate_rejection_reason"
                    ] = "nonintegral_or_nonbinary_returned_commitment"
            else:
                payload["gpu_primal_heuristics"]["status"] = "no_incumbent"
                payload["gpu_primal_heuristics"]["candidate_enqueued"] = False
                payload["gpu_primal_heuristics"]["candidate_rejection_reason"] = (
                    "both_gpu_heuristic_stages_returned_no_incumbent"
                )
            heuristic_pipeline_wall = (
                time.perf_counter() - heuristic_pipeline_started
            )
            payload["gpu_primal_heuristics"]["total_wall_time_seconds"] = (
                heuristic_pipeline_wall
            )
            payload["timings_seconds"]["gpu_primal_heuristics_adapter_wall"] = (
                seed_wall + full_wall
            )
            save()

        if gpu_heuristic_candidate is not None:
            # The GPU heuristic is an incumbent-improvement attempt even when
            # the type-rounded pipeline already found a secure dispatch.  Do
            # not let stale fallback candidates consume its bounded budget.
            pending_primal_candidates.clear()
            try_primal(root, *gpu_heuristic_candidate)
            pending_primal_candidates.clear()
        if best_primal is None:
            initial_candidates = [
                (
                    np.rint(root.commitment),
                    "root_pdlp_rounding",
                    None,
                ),
                (
                    np.asarray(
                        root.gpu_lagrangian["best_minimizing_commitment"],
                        dtype=np.float64,
                    ),
                    "root_gpu_lagrangian_minimizer",
                    None,
                ),
                (
                    np.asarray(root.commitment >= 0.25, dtype=np.float64),
                    "root_pdlp_threshold_0p25",
                    None,
                ),
                (
                    np.ones(source_rows.size, dtype=np.float64),
                    "all_online_fallback",
                    None,
                ),
            ]
            for proposed, origin, generation_audit in initial_candidates:
                pending_primal_candidates.append(
                    (root, proposed, origin, generation_audit)
                )
            drain_primal_candidate_queue()

        if (
            config.benchmark_id == ACTIVSG2000_V11_EXPERIMENT_ID
            and best_primal is not None
        ):
            payload["active_stage"] = "global_feasibility_cut_lagrangian_refresh"
            global_cuts = tuple(
                global_feasibility_cuts[cut_id]
                for cut_id in sorted(global_feasibility_cuts)
            )
            root = _refresh_region_with_global_commitment_cuts(
                region=root,
                global_cuts=global_cuts,
                case=case,
                network=network,
                config=config,
                dual_target_objective=float(best_primal["objective"]),
            )
            frontier[root.region_id] = root
            all_region_records[0] = serialize_region(root)
            payload["global_feasibility_cut_lagrangian_policy"] = {
                "enabled": True,
                "cut_count": len(global_cuts),
                "all_frontier_descendants_inherit_cuts": True,
                "diagonal_preconditioning": True,
                "cpu_solution_data_used": False,
            }
            persist_region_evidence()
            replay_and_checkpoint_frontier(
                "independent_root_replay_after_global_feasibility_cuts"
            )

        maximum_regions = int(config.runtime["maximum_frontier_regions"])
        if config.benchmark_id == ACTIVSG2000_V11_EXPERIMENT_ID:
            required_split_budget = (
                2.0 * float(config.runtime["maximum_region_attempt_seconds"])
                + float(config.runtime["split_transaction_margin_seconds"])
            )
            if float(config.runtime["minimum_refinement_launch_seconds"]) < (
                required_split_budget
            ):
                raise ScopfError(
                    "v11 split preflight does not cover two bounded sequential children"
                )
            payload["split_transaction_policy"] = {
                "child_solver_contexts": 1,
                "maximum_seconds_per_child": float(
                    config.runtime["maximum_region_attempt_seconds"]
                ),
                "controller_margin_seconds": float(
                    config.runtime["split_transaction_margin_seconds"]
                ),
                "minimum_launch_budget_seconds": required_split_budget,
                "rollback_on_deadline": True,
                "verification_reserve_is_excluded": True,
            }
        failed_split_positions: dict[str, set[int]] = {}
        failed_cardinality_cut_ids: dict[str, set[str]] = {}
        split_attempt_number = 0
        while True:
            if best_primal is None:
                gap = None
            else:
                lower_bound = min(
                    region.lagrangian.conservative_lower_bound for region in frontier.values()
                )
                upper_bound = float(best_primal["objective"])
                gap = _relative_gap(objective=upper_bound, lower_bound=lower_bound)
                payload["objective"] = upper_bound
                payload["bound"] = lower_bound
                payload["relative_gap"] = gap
                if gap <= target_gap:
                    break
            if len(frontier) >= maximum_regions:
                payload["status"] = "incomplete_frontier_region_limit"
                break
            if config.benchmark_id in ACTIVSG2000_V8_PLUS_EXPERIMENT_IDS:
                try:
                    remaining_refinement_budget = deadline.solver_budget()
                except DeadlineExceeded:
                    remaining_refinement_budget = 0.0
                if remaining_refinement_budget < float(
                    config.runtime["minimum_refinement_launch_seconds"]
                ):
                    payload["status"] = "incomplete_refinement_budget_reserve_reached"
                    break
            deadline.require("disjunctive refinement")
            parent = None
            split_position = None
            cardinality_split: CardinalitySplit | None = None
            for candidate_parent in sorted(
                frontier.values(),
                key=lambda region: (
                    region.lagrangian.conservative_lower_bound,
                    region.region_id,
                ),
            ):
                try:
                    if config.benchmark_id in {
                        ACTIVSG2000_V10_EXPERIMENT_ID,
                        ACTIVSG2000_V11_EXPERIMENT_ID,
                    }:
                        candidate_cardinality_split: CardinalitySplit | None = None
                        candidate_position: int | None = None
                        try:
                            candidate_cardinality_split = choose_cardinality_split(
                                master=candidate_parent.master,
                                commitments=candidate_parent.commitment,
                                subsets=cardinality_subsets,
                                existing_cut_ids={
                                    cut.cut_id for cut in candidate_parent.commitment_cuts
                                }
                                | failed_cardinality_cut_ids.get(
                                    candidate_parent.region_id, set()
                                ),
                            )
                        except ScopfError:
                            candidate_position = choose_split_generator(
                                candidate_parent.commitment,
                                candidate_parent.lagrangian.on_subproblem_values,
                                candidate_parent.masks,
                                excluded_positions=failed_split_positions.get(
                                    candidate_parent.region_id, set()
                                ),
                            )
                    else:
                        candidate_position = choose_split_generator(
                            candidate_parent.commitment,
                            candidate_parent.lagrangian.on_subproblem_values,
                            candidate_parent.masks,
                            excluded_positions=failed_split_positions.get(
                                candidate_parent.region_id, set()
                            ),
                        )
                except ScopfError:
                    continue
                parent = candidate_parent
                if (
                    config.benchmark_id
                    in {
                        ACTIVSG2000_V10_EXPERIMENT_ID,
                        ACTIVSG2000_V11_EXPERIMENT_ID,
                    }
                    and candidate_cardinality_split is not None
                ):
                    cardinality_split = candidate_cardinality_split
                else:
                    split_position = candidate_position
                break
            if parent is None or (
                split_position is None and cardinality_split is None
            ):
                payload["status"] = "incomplete_no_remaining_split_generator"
                break
            split_attempt_number += 1
            if cardinality_split is None:
                assert split_position is not None
                off_masks, on_masks = parent.masks.split(split_position)
                off_cuts = parent.commitment_cuts
                on_cuts = parent.commitment_cuts
            else:
                off_masks = parent.masks
                on_masks = parent.masks
                off_cuts = parent.commitment_cuts + (
                    cardinality_split.at_most_cut,
                )
                on_cuts = parent.commitment_cuts + (
                    cardinality_split.at_least_cut,
                )
            if region_attempt_policy is None:
                off_id = f"{parent.region_id}0"
                on_id = f"{parent.region_id}1"
            else:
                off_id = f"{parent.region_id}_s{split_attempt_number:03d}_0"
                on_id = f"{parent.region_id}_s{split_attempt_number:03d}_1"
            split_record = {
                "parent_region_id": parent.region_id,
                "off_child_region_id": off_id,
                "on_child_region_id": on_id,
                "transaction_attempt": split_attempt_number,
            }
            if cardinality_split is None:
                assert split_position is not None
                split_record.update(
                    {
                        "split_kind": "binary_commitment_v1",
                        "generator_position": split_position,
                        "generator_source_row": int(source_rows[split_position]) + 1,
                    }
                )
            else:
                split_record.update(cardinality_split.as_dict())
            payload["pending_disjunctive_split"] = {
                **split_record,
                "status": "solving_children_parent_certificate_retained",
                "completed_child_region_ids": [],
            }
            save()
            solved_children: dict[str, SolvedRegion] = {}
            pruned_children: dict[str, dict[str, Any]] = {}
            tentative_outcomes: list[dict[str, Any]] = []
            split_failed = False
            child_specs: list[dict[str, Any]] = []
            for child_id, child_masks, child_cuts in (
                (off_id, off_masks, off_cuts),
                (on_id, on_masks, on_cuts),
            ):
                child_initial_pairs = tuple(sorted(global_pairs.values()))
                prepared_master: ReducedMaster | None = None
                phase_warm_start: np.ndarray | None = None
                phase_dual_warm_start: np.ndarray | None = None
                warm_start_origin: str | None = None
                if phase_one_first:
                    payload["active_stage"] = f"phase_one_precheck_{child_id}"
                    prepared_master = _prepare_region_master(
                        case=case,
                        network=network,
                        config=config,
                        masks=child_masks,
                        initial_pairs=child_initial_pairs,
                        commitment_cuts=child_cuts,
                    )
                    _validate_prepared_region_master(
                        prepared_master,
                        child_masks,
                        child_initial_pairs,
                        child_cuts,
                    )
                    capacity_gate = _region_pmin_pmax_capacity_gate(
                        case=case,
                        master=prepared_master,
                        masks=child_masks,
                        tolerance_pu=float(config.model["model_residual_tolerance_pu"]),
                    )
                    precheck_rejection = RegionAttemptRejected(
                        f"Region {child_id} entered the registered Phase-I-first gate",
                        reason="phase_one_first_precheck",
                        master=prepared_master,
                        security_pairs=child_initial_pairs,
                        rounds=[],
                    )
                    precheck = _run_phase_one_attempt(
                        region_id=child_id,
                        masks=child_masks,
                        rejected=precheck_rejection,
                        case=case,
                        config=config,
                        deadline=deadline,
                        attempt_kind="pre_cost_lp",
                        time_limit_seconds=float(
                            config.runtime["precheck_phase_one_time_limit_seconds"]
                        ),
                        capacity_gate=capacity_gate,
                        commitment_cuts=child_cuts,
                    )
                    precheck_record = precheck.record
                    payload["phase_one_prechecks"].append(
                        _phase_one_attempt_summary(precheck_record)
                    )
                    if precheck_record["prune_certified"]:
                        pruned_children[child_id] = precheck_record
                        tentative_outcomes.append(
                            {
                                "region_id": child_id,
                                "status": "phase_one_first_pruned",
                                "phase_one": precheck_record,
                            }
                        )
                        payload["pending_disjunctive_split"]["completed_child_region_ids"].append(
                            child_id
                        )
                        payload["pending_disjunctive_split"]["tentative_outcomes"] = (
                            tentative_outcomes
                        )
                        save()
                        continue
                    if config.benchmark_id in {
                        ACTIVSG2000_V6_EXPERIMENT_ID,
                        ACTIVSG2000_V7_EXPERIMENT_ID,
                        ACTIVSG2000_V8_EXPERIMENT_ID,
                        ACTIVSG2000_V9_EXPERIMENT_ID,
                    }:
                        try:
                            phase_child, phase_prune = (
                                _solve_phase_one_lagrangian_region(
                                    region_id=child_id,
                                    masks=child_masks,
                                    parent=parent,
                                    master=prepared_master,
                                    initial_pairs=child_initial_pairs,
                                    initial_precheck=precheck,
                                    case=case,
                                    network=network,
                                    config=config,
                                    deadline=deadline,
                                    screener=screener,
                                    checkpoint=save,
                                )
                            )
                        except RegionAttemptRejected as error:
                            global_pairs.update(
                                (pair.pair_id, pair)
                                for pair in error.security_pairs
                            )
                            tentative_outcomes.append(
                                {
                                    "region_id": child_id,
                                    "status": "phase_one_lagrangian_inconclusive",
                                    "reason": error.reason,
                                    "rounds": error.rounds,
                                }
                            )
                            payload["pending_disjunctive_split"][
                                "tentative_outcomes"
                            ] = tentative_outcomes
                            split_failed = True
                            save()
                            break
                        if phase_prune is not None:
                            pruned_children[child_id] = phase_prune
                            tentative_outcomes.append(
                                {
                                    "region_id": child_id,
                                    "status": "phase_one_lagrangian_pruned",
                                    "phase_one": phase_prune,
                                }
                            )
                            payload["pending_disjunctive_split"][
                                "completed_child_region_ids"
                            ].append(child_id)
                            payload["pending_disjunctive_split"][
                                "tentative_outcomes"
                            ] = tentative_outcomes
                            save()
                            continue
                        if phase_child is None:
                            raise ScopfError(
                                "Phase-I Lagrangian child returned no outcome"
                            )
                        solved_children[child_id] = phase_child
                        global_pairs.update(
                            (pair.pair_id, pair)
                            for pair in phase_child.security_pairs
                        )
                        tentative_outcomes.append(
                            {
                                "region_id": child_id,
                                "status": "phase_one_lagrangian_solved",
                                "region": serialize_region(phase_child),
                            }
                        )
                        payload["pending_disjunctive_split"][
                            "completed_child_region_ids"
                        ].append(child_id)
                        payload["pending_disjunctive_split"][
                            "tentative_outcomes"
                        ] = tentative_outcomes
                        save()
                        continue
                    phase_warm_start = precheck.source_native_primal
                    phase_dual_warm_start = precheck.source_native_row_dual
                    if phase_warm_start is not None:
                        warm_start_origin = (
                            "phase_one_zero_violation_primal_dual_v2"
                            if phase_dual_warm_start is not None
                            else "phase_one_zero_violation_primal_v1"
                        )
                    save()
                elif config.benchmark_id in {
                    ACTIVSG2000_V10_EXPERIMENT_ID,
                    ACTIVSG2000_V11_EXPERIMENT_ID,
                }:
                    prepared_master = _prepare_region_master(
                        case=case,
                        network=network,
                        config=config,
                        masks=child_masks,
                        initial_pairs=child_initial_pairs,
                        commitment_cuts=child_cuts,
                    )
                    _validate_prepared_region_master(
                        prepared_master,
                        child_masks,
                        child_initial_pairs,
                        child_cuts,
                    )
                    # The parent LP value lies strictly between the two integer
                    # cardinalities used to form this disjunction.  It therefore
                    # violates both newly appended child rows.  Submitting it as
                    # a child primal start can make PDLP spend its first restart
                    # repairing a deliberately infeasible vector and has caused
                    # unstable residual trajectories in cuOpt.  Preserve the
                    # compatible row-dual state, but start each child primal cold.
                    phase_warm_start = None
                    if config.benchmark_id == ACTIVSG2000_V11_EXPERIMENT_ID:
                        parent_canonical_dual, _parent_cut_dual = (
                            _certificate_dual_arrays(
                                parent.master,
                                parent.lagrangian,
                                parent.commitment_cuts,
                            )
                        )
                        _parent_column_scale, parent_row_scale = (
                            native_scaling_vectors(
                                parent.master.canonical,
                                mode=str(
                                    config.raw["platforms"]["dgx_spark"][
                                        "native_scaling_mode"
                                    ]
                                ),
                                base_mva=float(case.base_mva),
                            )
                        )
                        parent_native_certificate_dual = (
                            parent_canonical_dual / parent_row_scale
                        )
                        phase_dual_warm_start, dual_mapping = (
                            _map_native_row_dual_by_identity(
                                parent.master.canonical,
                                prepared_master.canonical,
                                parent_native_certificate_dual,
                                scaling_mode=str(
                                    config.raw["platforms"]["dgx_spark"][
                                        "native_scaling_mode"
                                    ]
                                ),
                                base_mva=float(case.base_mva),
                            )
                        )
                    elif parent.solve.native_row_dual is not None:
                        phase_dual_warm_start, dual_mapping = (
                            _map_native_row_dual_by_identity(
                                parent.master.canonical,
                                prepared_master.canonical,
                                np.asarray(
                                    parent.solve.native_row_dual, dtype=np.float64
                                ),
                                scaling_mode=str(
                                    config.raw["platforms"]["dgx_spark"][
                                        "native_scaling_mode"
                                    ]
                                ),
                                base_mva=float(case.base_mva),
                            )
                        )
                    else:
                        dual_mapping = None
                    warm_start_origin = (
                        "parent_replayable_certificate_mapped_dual_only_v3"
                        if (
                            phase_dual_warm_start is not None
                            and config.benchmark_id
                            == ACTIVSG2000_V11_EXPERIMENT_ID
                        )
                        else "parent_row_name_mapped_dual_only_v2"
                        if phase_dual_warm_start is not None
                        else None
                    )
                child_specs.append(
                    {
                        "child_id": child_id,
                        "masks": child_masks,
                        "initial_pairs": child_initial_pairs,
                        "prepared_master": prepared_master,
                        "initial_native_primal": phase_warm_start,
                        "initial_native_row_dual": phase_dual_warm_start,
                        "initial_warm_start_origin": warm_start_origin,
                        "commitment_cuts": child_cuts,
                        "parent_dual_mapping": (
                            dual_mapping
                            if config.benchmark_id
                            in {
                                ACTIVSG2000_V10_EXPERIMENT_ID,
                                ACTIVSG2000_V11_EXPERIMENT_ID,
                            }
                            else None
                        ),
                    }
                )

            if split_failed:
                if cardinality_split is None:
                    assert split_position is not None
                    failed_split_positions.setdefault(parent.region_id, set()).add(
                        split_position
                    )
                else:
                    failed_cardinality_cut_ids.setdefault(
                        parent.region_id, set()
                    ).update(
                        {
                            cardinality_split.at_most_cut.cut_id,
                            cardinality_split.at_least_cut.cut_id,
                        }
                    )
                payload["failed_disjunctive_split_attempts"].append(
                    {
                        **split_record,
                        "status": "rolled_back_phase_one_lagrangian_inconclusive",
                        "parent_certificate_retained": True,
                        "outcomes": tentative_outcomes,
                    }
                )
                payload.pop("pending_disjunctive_split", None)
                persist_region_evidence()
                if len(payload["failed_disjunctive_split_attempts"]) >= int(
                    config.runtime["maximum_failed_split_attempts"]
                ):
                    payload["status"] = "incomplete_failed_split_attempt_limit"
                    break
                continue

            parallel_contexts = int(config.runtime.get("parallel_child_solver_contexts", 1))
            parallel_minimum_frontier = int(
                config.runtime.get("parallel_child_minimum_frontier_regions", 10**9)
            )
            use_parallel_children = bool(
                config.benchmark_id in ACTIVSG2000_V4_PLUS_EXPERIMENT_IDS
                and parallel_contexts == 2
                and len(frontier) >= parallel_minimum_frontier
                and len(child_specs) == 2
            )
            cost_outcomes: dict[str, tuple[SolvedRegion | None, RegionAttemptRejected | None]] = {}
            transaction_deadline_error: DeadlineExceeded | None = None

            def solve_child_cost(
                spec: dict[str, Any],
                *,
                child_screener: ContingencyScreener,
                concurrent_context: bool,
                progress_callback: Callable[[dict[str, Any]], None],
                certificate_parent: SolvedRegion = parent,
            ) -> tuple[SolvedRegion | None, RegionAttemptRejected | None]:
                try:
                    child = _solve_region(
                        region_id=str(spec["child_id"]),
                        masks=spec["masks"],
                        case=case,
                        network=network,
                        catalog=catalog,
                        config=config,
                        deadline=deadline,
                        initial_pairs=spec["initial_pairs"],
                        screener=child_screener,
                        checkpoint=lambda: None,
                        progress=progress_callback,
                        candidate_policy=region_attempt_policy,
                        prepared_master=spec["prepared_master"],
                        initial_native_primal=spec["initial_native_primal"],
                        initial_native_row_dual=spec["initial_native_row_dual"],
                        initial_warm_start_origin=spec["initial_warm_start_origin"],
                        concurrent_solver_context=concurrent_context,
                        commitment_cuts=spec["commitment_cuts"],
                    )
                    if config.benchmark_id == ACTIVSG2000_V11_EXPERIMENT_ID:
                        child = _enforce_monotone_child_certificate(
                            parent=certificate_parent,
                            child=child,
                            replay_tolerance_dollars=float(
                                config.raw["benchmark"][
                                    "gpu_cpu_replay_tolerance_dollars"
                                ]
                            ),
                        )
                    return child, None
                except RegionAttemptRejected as error:
                    return None, error

            if use_parallel_children:
                payload["active_stage"] = f"parallel_disjunctive_regions_{off_id}_{on_id}"
                progress_lock = threading.Lock()

                def save_parallel_progress(
                    child_id: str,
                    record: dict[str, Any],
                    _progress_lock: threading.Lock = progress_lock,
                ) -> None:
                    with _progress_lock:
                        payload.setdefault("active_parallel_region_progress", {})[child_id] = record
                        save()

                batch_started = time.perf_counter()
                parallel_screeners = {
                    str(spec["child_id"]): ContingencyScreener(
                        network,
                        catalog,
                        backend="cupy",
                        chunk_columns=int(config.model["screen_chunk_columns"]),
                    )
                    for spec in child_specs
                }
                with ThreadPoolExecutor(
                    max_workers=parallel_contexts,
                    thread_name_prefix="activsg-cuopt-child",
                ) as pool:
                    futures = {
                        str(spec["child_id"]): pool.submit(
                            solve_child_cost,
                            spec,
                            child_screener=parallel_screeners[str(spec["child_id"])],
                            concurrent_context=True,
                            progress_callback=lambda record, child_id=str(spec["child_id"]): (
                                save_parallel_progress(child_id, record)
                            ),
                        )
                        for spec in child_specs
                    }
                    for child_id in sorted(futures):
                        cost_outcomes[child_id] = futures[child_id].result()
                payload.pop("active_parallel_region_progress", None)
                payload.setdefault("parallel_child_batches", []).append(
                    {
                        "policy": "two_threaded_independent_cuopt_pdlp_contexts_v1",
                        "child_region_ids": sorted(cost_outcomes),
                        "frontier_region_count_before_batch": len(frontier),
                        "wall_time_seconds": time.perf_counter() - batch_started,
                        "native_log_policy": (
                            "console_only_per_context_file_disabled_due_cuopt_process_global_logger"
                        ),
                    }
                )
                save()
            else:
                try:
                    for spec in child_specs:
                        child_id = str(spec["child_id"])
                        payload["active_stage"] = f"disjunctive_region_{child_id}"
                        cost_outcomes[child_id] = solve_child_cost(
                            spec,
                            child_screener=screener,
                            concurrent_context=False,
                            progress_callback=save_region_progress,
                        )
                except DeadlineExceeded as exc:
                    transaction_deadline_error = exc

            if transaction_deadline_error is not None:
                payload["failed_disjunctive_split_attempts"].append(
                    {
                        **split_record,
                        "status": "rolled_back_transaction_deadline_guard",
                        "parent_certificate_retained": True,
                        "error_type": type(transaction_deadline_error).__name__,
                        "error": str(transaction_deadline_error),
                    }
                )
                payload.pop("pending_disjunctive_split", None)
                payload.pop("active_region_progress", None)
                payload["status"] = "incomplete_refinement_budget_reserve_reached"
                persist_region_evidence()
                break

            for spec in child_specs:
                child_id = str(spec["child_id"])
                child_masks = spec["masks"]
                child, rejected = cost_outcomes[child_id]
                payload.pop("active_region_progress", None)
                if rejected is not None:
                    global_pairs.update((pair.pair_id, pair) for pair in rejected.security_pairs)
                    if region_attempt_policy is None:
                        raise rejected
                    payload["active_stage"] = f"phase_one_{child_id}"
                    phase_result = _run_phase_one_attempt(
                        region_id=child_id,
                        masks=child_masks,
                        rejected=rejected,
                        case=case,
                        config=config,
                        deadline=deadline,
                        commitment_cuts=spec["commitment_cuts"],
                    )
                    phase_record = phase_result.record
                    payload["phase_one_fallback_attempts"].append(
                        _phase_one_attempt_summary(phase_record)
                    )
                    tentative_outcomes.append(
                        {
                            "region_id": child_id,
                            "status": (
                                "phase_one_pruned"
                                if phase_record["prune_certified"]
                                else "phase_one_not_certified"
                            ),
                            "phase_one": phase_record,
                        }
                    )
                    payload["pending_disjunctive_split"]["tentative_outcomes"] = tentative_outcomes
                    if phase_record["prune_certified"]:
                        pruned_children[child_id] = phase_record
                        payload["pending_disjunctive_split"]["completed_child_region_ids"].append(
                            child_id
                        )
                        save()
                        continue
                    split_failed = True
                    break
                if child is None:
                    raise ScopfError("Disjunctive child solve returned no outcome")
                solved_children[child_id] = child
                global_pairs.update((pair.pair_id, pair) for pair in child.security_pairs)
                tentative_outcomes.append(
                    {
                        "region_id": child_id,
                        "status": "solved",
                        "region": serialize_region(child),
                    }
                )
                payload["pending_disjunctive_split"]["tentative_outcomes"] = tentative_outcomes
                payload["pending_disjunctive_split"]["completed_child_region_ids"].append(child_id)
                save()
            if split_failed:
                if cardinality_split is None:
                    assert split_position is not None
                    failed_split_positions.setdefault(parent.region_id, set()).add(
                        split_position
                    )
                else:
                    failed_cardinality_cut_ids.setdefault(
                        parent.region_id, set()
                    ).update(
                        {
                            cardinality_split.at_most_cut.cut_id,
                            cardinality_split.at_least_cut.cut_id,
                        }
                    )
                payload["failed_disjunctive_split_attempts"].append(
                    {
                        **split_record,
                        "status": "rolled_back_phase_one_not_certified",
                        "parent_certificate_retained": True,
                        "outcomes": tentative_outcomes,
                    }
                )
                payload.pop("pending_disjunctive_split", None)
                persist_region_evidence()
                if len(payload["failed_disjunctive_split_attempts"]) >= int(
                    config.runtime["maximum_failed_split_attempts"]
                ):
                    payload["status"] = "incomplete_failed_split_attempt_limit"
                    break
                continue
            del frontier[parent.region_id]
            for child_id, child in solved_children.items():
                frontier[child_id] = child
                all_region_records.append(serialize_region(child))
            payload["pruned_regions"].extend(
                pruned_children[child_id] for child_id in sorted(pruned_children)
            )
            if not frontier:
                raise ScopfError(
                    "All active regions were Phase-I pruned despite a secure incumbent"
                )
            payload["disjunctive_splits"].append(split_record)
            payload.pop("pending_disjunctive_split", None)
            persist_region_evidence()
            if replay_after_every_split:
                replay_and_checkpoint_frontier(
                    f"independent_frontier_replay_after_{parent.region_id}"
                )
            for child_id in (off_id, on_id):
                if best_primal is not None:
                    break
                child = solved_children.get(child_id)
                if child is None:
                    continue
                pending_primal_candidates.append(
                    (
                        child,
                        np.rint(child.commitment),
                        f"region_{child_id}_rounding",
                        None,
                    )
                )
                drain_primal_candidate_queue()

        if str(payload.get("status", "")).startswith("incomplete_"):
            payload["refinement_stop_status"] = payload["status"]
        payload["all_solved_region_count"] = len(all_region_records)
        payload["solved_region_history"] = all_region_records
        payload["frontier_regions"] = [
            serialize_region(frontier[region_id]) for region_id in sorted(frontier)
        ]
        if best_primal is None:
            payload["status"] = "incomplete_no_secure_primal"
            save()
            return payload
        payload["solution"] = best_primal["solution"]
        payload["objective"] = float(best_primal["objective"])
        payload["commitment_count"] = int(np.count_nonzero(best_primal["commitment"]))
        payload["pricing"] = best_primal["pricing"]
        payload["bound"] = min(
            region.lagrangian.conservative_lower_bound for region in frontier.values()
        )
        payload["relative_gap"] = _relative_gap(
            objective=float(payload["objective"]),
            lower_bound=float(payload["bound"]),
        )
        save()

        deadline.require(
            "mandatory final Lagrangian frontier replay",
            reserve_seconds=float(config.runtime["serialization_reserve_seconds"]),
        )
        replay_and_checkpoint_frontier("independent_final_frontier_replay")

        deadline.require(
            "independent primal and Lagrangian verification",
            reserve_seconds=float(config.runtime["serialization_reserve_seconds"]),
        )
        payload["active_stage"] = "independent_verification"
        verification_started = time.perf_counter()
        primal_verification = verify_serialized_solution(config, payload)
        payload["verification"] = primal_verification.as_dict()
        final_replay = payload.get("lagrangian_replay_history", [])[-1]
        if final_replay.get("stage") != "independent_final_frontier_replay":
            raise ScopfError("Final Lagrangian replay checkpoint is missing")
        lagrangian_verification = {
            key: value for key, value in final_replay.items() if key != "stage"
        }
        payload["lagrangian_verification"] = lagrangian_verification
        payload["timings_seconds"]["independent_verification"] = (
            time.perf_counter() - verification_started
        )
        relaxation_rounds = [
            round_record
            for region_record in all_region_records
            for round_record in region_record["constraint_generation_rounds"]
        ]
        primal_rounds = [
            round_record
            for repair in payload["primal_repairs"]
            for round_record in repair.get("constraint_generation_rounds", [])
        ]
        projected_feasibility_rounds = [
            round_record
            for repair in payload["primal_repairs"]
            for round_record in repair.get("feasibility_polisher", {}).get(
                "constraint_generation_rounds", []
            )
        ]
        projected_phase_one_attempts = [
            solve_attempt
            for round_record in projected_feasibility_rounds
            for solve_attempt in round_record.get("solve_attempts", [])
        ]
        secure_seed_margin_rounds = [
            refinement_round
            for repair in payload["primal_repairs"]
            for cost_round in repair.get("constraint_generation_rounds", [])
            for refinement in cost_round.get(
                "secure_seed_margin_refinements", []
            )
            for refinement_round in refinement.get(
                "constraint_generation_rounds", []
            )
        ]
        secure_seed_margin_attempts = [
            solve_attempt
            for round_record in secure_seed_margin_rounds
            for solve_attempt in round_record.get("solve_attempts", [])
        ]
        payload["timings_seconds"].update(
            {
                "relaxation_pdlp_adapter_wall": sum(
                    float(record["adapter_wall_time_seconds"]) for record in relaxation_rounds
                ),
                "relaxation_cupy_screening_wall": sum(
                    float(record.get("screen", {}).get("wall_time_seconds", 0.0))
                    for record in relaxation_rounds
                ),
                "fixed_commitment_pdlp_adapter_wall": sum(
                    float(record["adapter_wall_time_seconds"]) for record in primal_rounds
                ),
                "fixed_commitment_cupy_screening_wall": sum(
                    float(record.get("screen", {}).get("wall_time_seconds", 0.0))
                    for record in primal_rounds
                ),
                "fixed_commitment_projected_phase_one_pdlp_adapter_wall": sum(
                    float(record["adapter_wall_time_seconds"])
                    for record in projected_phase_one_attempts
                ),
                "fixed_commitment_projected_cupy_screening_wall": sum(
                    float(record.get("screen", {}).get("wall_time_seconds", 0.0))
                    for record in projected_feasibility_rounds
                ),
                "fixed_commitment_secure_seed_margin_pdlp_adapter_wall": sum(
                    float(record["adapter_wall_time_seconds"])
                    for record in secure_seed_margin_attempts
                ),
                "fixed_commitment_secure_seed_margin_cupy_screening_wall": sum(
                    float(record.get("screen", {}).get("wall_time_seconds", 0.0))
                    for record in secure_seed_margin_rounds
                ),
                "cupy_lagrangian_evaluation_wall": sum(
                    float(record["gpu_lagrangian_evaluation"]["wall_time_seconds"])
                    for record in all_region_records
                ),
                "phase_one_precheck_pdlp_adapter_wall": sum(
                    float(record.get("adapter_wall_time_seconds", 0.0))
                    for record in payload.get("phase_one_prechecks", [])
                ),
                "phase_one_fallback_pdlp_adapter_wall": sum(
                    float(record.get("adapter_wall_time_seconds", 0.0))
                    for record in payload.get("phase_one_fallback_attempts", [])
                ),
                "phase_one_pdlp_adapter_wall": sum(
                    float(record.get("adapter_wall_time_seconds", 0.0))
                    for records in (
                        payload.get("phase_one_prechecks", []),
                        payload.get("phase_one_fallback_attempts", []),
                    )
                    for record in records
                ),
            }
        )
        payload["phase_one_first_outcomes"] = {
            "precheck_attempt_count": len(payload.get("phase_one_prechecks", [])),
            "cost_lp_attempts_avoided_by_certified_precheck": sum(
                int(bool(record.get("prune_certified")))
                for record in payload.get("phase_one_prechecks", [])
            ),
            "eligible_cost_lp_primal_warm_starts": sum(
                int(bool(record.get("source_feasible_warm_start", {}).get("eligible", False)))
                for record in payload.get("phase_one_prechecks", [])
                if not bool(record.get("prune_certified"))
            ),
            "fallback_phase_one_attempt_count": len(payload.get("phase_one_fallback_attempts", [])),
        }
        gap_passed = payload["relative_gap"] <= target_gap * (1.0 + 1e-9) + 1e-12
        payload["acceptance_gates"] = {
            "secure_primal_independently_verified": primal_verification.passed,
            "disjunctive_lagrangian_bound_independently_replayed": (
                lagrangian_verification["passed"]
            ),
            "requested_relative_gap": target_gap,
            "relative_gap": payload["relative_gap"],
            "gap_certified": gap_passed,
            "integer_solver_absent": not bool(payload["integer_solver_used"]),
            "integer_lower_bound_solver_absent": not bool(
                payload["integer_lower_bound_solver_used"]
            ),
            "gpu_primal_heuristics_only": bool(
                payload.get("gpu_primal_heuristics", {}).get(
                    "mip_heuristics_only", False
                )
            ),
            "branch_and_bound_absent": True,
            "all_pruned_regions_have_replayed_phase_one_certificates": bool(
                lagrangian_verification["phase_one_pruned_region_count"]
                == len(payload.get("pruned_regions", []))
            ),
            "final_exhaustive_security_violation_pu": (
                primal_verification.maximum_security_violation_pu
            ),
        }
        if primal_verification.passed and lagrangian_verification["passed"] and gap_passed:
            payload["status"] = "optimality_gap_certified_gpu_lagrangian"
        elif not primal_verification.passed:
            payload["status"] = "failed_independent_primal_verification"
        elif not lagrangian_verification["passed"]:
            payload["status"] = "failed_independent_lagrangian_verification"
        else:
            payload["status"] = payload.get(
                "refinement_stop_status",
                "incomplete_requested_gap_not_certified",
            )
        if config.benchmark_id in ACTIVSG2000_V8_PLUS_EXPERIMENT_IDS:
            payload["cpu_comparison"] = _load_cpu_comparison(config, registration)
        cpu = payload["cpu_comparison"]
        payload["system_comparison"] = {
            "comparison_kind": "system_to_system_not_pure_gpu_speedup",
            "laptop_cpu_end_to_end_seconds": cpu["total_wall_time_seconds"],
            "dgx_spark_end_to_end_seconds": None,
            "laptop_cpu_objective": cpu["objective"],
            "dgx_spark_objective": payload["objective"],
            "laptop_cpu_gap": cpu["mip_gap"],
            "dgx_spark_gap": payload["relative_gap"],
        }
        save()

    payload["active_stage"] = "complete"
    payload["elapsed_seconds"] = deadline.elapsed
    return payload


def run_one_shot_gpu_lagrangian_experiment(
    config: RunConfig, *, output_path: Path
) -> dict[str, Any]:
    registration = validate_lagrangian_experiment_config(config)
    validate_platform(config, "dgx_spark")
    output = guard_output_path(output_path)
    experiment_root = (config.root / "results" / "experiments").resolve()
    if not output.resolve().is_relative_to(experiment_root):
        raise ScopfError("GPU Lagrangian output must be under results/experiments")
    if output.exists():
        raise ScopfError(f"GPU Lagrangian output already exists: {output}")
    identity = frozen_identity(config)
    suite_id = str(registration["benchmark"]["experiment_suite_id"])
    if suite_id != config.benchmark_id:
        raise ScopfError("GPU Lagrangian suite id must match its benchmark id")
    registry_path = guard_output_path(experiment_root / f"{suite_id}-run-registry.json")
    checkpoint_path = guard_output_path(
        config.root / "results" / "checkpoints" / f"{suite_id}-dgx_spark.json"
    )
    console_path = guard_output_path(
        config.root / "results" / "diagnostics" / f"{suite_id}-worker-console.log"
    )
    for path in (registry_path, checkpoint_path, console_path):
        if path.exists():
            raise ScopfError(f"GPU Lagrangian one-shot path already exists: {path}")
    registry = {
        "schema_version": "1.0.0",
        "experiment_suite_id": registration["benchmark"]["experiment_suite_id"],
        "runs": {
            "dgx_spark": {
                "status": "started",
                "started_at_utc": datetime.now(UTC).isoformat(),
                "host": platform.node(),
                "output": str(output),
                "checkpoint": str(checkpoint_path),
                "frozen_identity": identity,
            }
        },
    }
    write_json_atomic(registry, registry_path)
    command = [
        sys.executable,
        "-m",
        "activsg_scopf.cli",
        "_gpu_lagrangian_worker",
        "--config",
        str(config.path),
        "--output",
        str(output),
        "--checkpoint",
        str(checkpoint_path),
    ]
    deadline_seconds = float(config.runtime["deadline_seconds"])
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=config.root,
            check=False,
            capture_output=True,
            text=True,
            timeout=deadline_seconds,
        )
        wall = time.perf_counter() - started
        _write_console(console_path, completed.stdout, completed.stderr)
        result = (
            _read_json(output)
            if output.exists()
            else _read_json(checkpoint_path)
            if checkpoint_path.exists()
            else {
                "status": "failed_worker_without_result",
                "worker_returncode": completed.returncode,
                "worker_stderr": completed.stderr[-4000:],
            }
        )
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - started
        _write_console(console_path, exc.stdout, exc.stderr)
        result = _read_json(checkpoint_path) if checkpoint_path.exists() else {}
        result.update(
            {"status": "hard_deadline_exceeded", "worker_timeout_seconds": deadline_seconds}
        )
    result.update(
        {
            "official": False,
            "experiment": True,
            "one_shot": True,
            "frozen_identity": identity,
            "worker_console_log": str(console_path.relative_to(config.root)),
            "total_wall_time_seconds": wall,
            "benchmark_boundary": (
                "worker launch through raw loading, GPU PDLP primal/relaxation rounds, "
                "CuPy screening and Lagrangian evaluation, disjunctive refinement, "
                "independent CPU replay/verification, and result serialization"
            ),
        }
    )
    if "system_comparison" in result:
        result["system_comparison"]["dgx_spark_end_to_end_seconds"] = wall
    write_json_atomic(result, output)
    registry["runs"]["dgx_spark"].update(
        {
            "status": result.get("status"),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "total_wall_time_seconds": wall,
            "objective": result.get("objective"),
            "bound": result.get("bound"),
            "relative_gap": result.get("relative_gap"),
        }
    )
    write_json_atomic(registry, registry_path)
    return result


def run_gpu_lagrangian_worker_serialized(
    config: RunConfig, *, output_path: Path, checkpoint_path: Path
) -> dict[str, Any]:
    output = guard_output_path(output_path)
    checkpoint_file = guard_output_path(checkpoint_path)

    def checkpoint(payload: dict[str, Any]) -> None:
        write_json_atomic(payload, checkpoint_file)

    worker_started = time.perf_counter()
    try:
        result = run_gpu_lagrangian_experiment(config, checkpoint=checkpoint)
    except DeadlineExceeded as exc:
        result = _read_json(checkpoint_file) if checkpoint_file.exists() else {}
        result.update(
            {
                "status": "deadline_budget_exhausted",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    except Exception as exc:
        result = _read_json(checkpoint_file) if checkpoint_file.exists() else {}
        result.update(
            {
                "status": "failed_exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "exception_traceback": traceback.format_exc().splitlines()[-80:],
            }
        )
    result["elapsed_seconds"] = max(
        float(result.get("elapsed_seconds", 0.0)),
        time.perf_counter() - worker_started,
    )
    serialization_started = time.perf_counter()
    write_json_atomic(result, output)
    result.setdefault("timings_seconds", {})["result_serialization"] = (
        time.perf_counter() - serialization_started
    )
    write_json_atomic(result, output)
    return result
