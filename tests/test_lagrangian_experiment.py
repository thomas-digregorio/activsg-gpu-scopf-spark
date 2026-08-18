from pathlib import Path

import numpy as np
import pytest

from activsg_scopf import lagrangian_experiment as experiment_module
from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.errors import PrimalCandidateRejected, ScopfError
from activsg_scopf.lagrangian import RegionMasks
from activsg_scopf.lagrangian_experiment import (
    ACTIVSG2000_EXPERIMENT_ID,
    ACTIVSG2000_V4_EXPERIMENT_ID,
    ACTIVSG2000_V5_EXPERIMENT_ID,
    ACTIVSG2000_V6_EXPERIMENT_ID,
    ACTIVSG2000_V7_EXPERIMENT_ID,
    ACTIVSG2000_V8_EXPERIMENT_ID,
    ACTIVSG2000_V9_EXPERIMENT_ID,
    ACTIVSG2000_V10_EXPERIMENT_ID,
    EXPERIMENT_ID,
    EXPERIMENT_TAG,
    PrimalCandidatePolicy,
    RegionAttemptRejected,
    _load_cpu_comparison,
    _map_phase_one_dual_to_source_native,
    _prepare_region_master,
    _region_pmin_pmax_capacity_gate,
    _relative_gap,
    _replay_cleanup_audit_comparison,
    _run_phase_one_attempt,
    _solve_region,
    _validate_prepared_region_master,
    validate_lagrangian_experiment_config,
)
from activsg_scopf.matpower import GEN_STATUS
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import build_reduced_master
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
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v5"
    )
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
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v6"
    )
    assert v6.raw["raw_inputs"] == v5.raw["raw_inputs"]
    assert v6.model == v5.model
    assert v6.runtime["deadline_seconds"] == 990.0
    assert v6.runtime["gpu_primal_heuristics_seconds"] == 105.0
    assert v6.runtime["gpu_primal_seed_seconds"] == 75.0
    assert v6.raw["platforms"]["dgx_spark"]["pdlp_solver_mode_native"] == 1
    assert v6.raw["platforms"]["dgx_spark"]["save_best_primal_so_far"] is True
    fix = registration["benchmark"]["numerical_and_runtime_fix"]
    assert fix["primal_generator"] == (
        "reduced_gpu_heuristics_then_sparse_full_gpu_heuristics_"
        "with_complete_gpu_feasible_start_v3"
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
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v7"
    )
    assert v7.raw["raw_inputs"] == v6.raw["raw_inputs"]
    assert v7.model == v6.model
    assert v7.runtime == v6.runtime
    v7_profile = dict(v7.raw["platforms"]["dgx_spark"])
    assert v7_profile == v6.raw["platforms"]["dgx_spark"]
    fix = registration["benchmark"]["numerical_robustness_fix"]
    assert fix["same_shape_pdlp_continuation"] == (
        "optimal_complete_state_else_raw_primal_dual_v3"
    )
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
    assert fix["pricing_policy"] == (
        "fail_closed_if_repair_breaks_primal_dual_complementarity"
    )
    assert fix["repair_residual_budget_fraction"] == 0.5
    v7.raw["benchmark"]["numerical_robustness_fix"]["failed_v6_run_preserved"] = False
    with pytest.raises(ScopfError, match="numerical-robustness identity changed"):
        validate_lagrangian_experiment_config(v7)


def test_registered_activsg2000_v8_certificate_controller_is_fail_closed() -> None:
    v7 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v7.json")
    v8 = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v8.json")
    registration = validate_lagrangian_experiment_config(v8)

    assert v8.benchmark_id == ACTIVSG2000_V8_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v8"
    )
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
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v9"
    )
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
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v10"
    )
    assert v10.raw["raw_inputs"] == v9.raw["raw_inputs"]
    assert v10.model == v9.model
    assert v10.runtime["maximum_frontier_regions"] == 256
    assert v10.runtime["phase_lagrangian_gpu_iterations"] == 512
    assert registration["benchmark"]["cardinality_refinement"]["child_phase_one"] == (
        "disabled_after_zero_of_83_v9_prunes"
    )
    assert registration["benchmark"]["cardinality_refinement"][
        "minimum_cardinality_subset_size"
    ] == 2
    assert registration["benchmark"]["cardinality_refinement"]["binary_fallback"] == (
        "only_after_no_fractional_multi_unit_sum_remains"
    )
    assert registration["benchmark"]["cardinality_refinement"]["child_warm_start"] == (
        "row_name_mapped_parent_dual_only_because_parent_primal_violates_"
        "the_new_cardinality_branch_v2"
    )
    assert registration["benchmark"]["cardinality_refinement"][
        "cpu_commitment_dispatch_objective_or_bound_seeded"
    ] is False
    v10.raw["benchmark"]["cardinality_refinement"][
        "mathematical_original_integer_optimum_changed"
    ] = True
    with pytest.raises(ScopfError, match="cardinality identity changed"):
        validate_lagrangian_experiment_config(v10)


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
        values = (
            np.zeros(model.num_columns)
            if len(solve_calls) == 1
            else feasible_values(model)
        )
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
    assert solved.rounds[0]["pre_refinement_primal_acceptance"][
        "canonical_model_residual_passed"
    ] is False
    assert solved.rounds[0]["primal_acceptance"][
        "canonical_model_residual_passed"
    ] is True
    assert solved.rounds[0]["pre_refinement_primal_acceptance"][
        "worst_canonical_row"
    ]["row_name"] == "lag_balance"


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
