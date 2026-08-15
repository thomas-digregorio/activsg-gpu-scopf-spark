import json
from pathlib import Path

import pytest

from activsg_scopf.config import load_config
from activsg_scopf.errors import ScopeViolation
from activsg_scopf.seeded_diagnostic import validate_diagnostic_identity

ROOT = Path(__file__).resolve().parents[1]

PDLP_PROFILE = {
    "method": "pdlp",
    "solver_mode": "stable3",
    "precision": "fp64",
    "batch_strong_branching": True,
    "batch_reliability_branching": True,
    "reliability_branching_factor": 1,
}


def test_registered_config_is_single_hour_activsg500() -> None:
    config = load_config(ROOT / "configs" / "activsg500.json")
    assert config.raw["case_name"] == "ACTIVSg500"
    assert config.model["interval_hours"] == 1.0
    assert config.model["pwl_segments"] == 10
    assert config.runtime["deadline_seconds"] == 300.0


def test_registered_activsg10k_config_is_single_hour_and_separately_tagged() -> None:
    config = load_config(ROOT / "configs" / "activsg10k.json")
    assert config.case_name == "ACTIVSg10k"
    assert config.model["interval_hours"] == 1.0
    assert config.model["pwl_segments"] == 10
    assert config.benchmark_id == "activsg10k-v1"
    assert config.raw["benchmark"]["required_git_tag"] == "benchmark-10k-v1"


def test_activsg10k_v2_changes_runtime_identity_not_mathematical_contract() -> None:
    v1 = load_config(ROOT / "configs" / "activsg10k.json")
    v2 = load_config(ROOT / "configs" / "activsg10k-v2.json")
    assert v2.case_name == v1.case_name == "ACTIVSg10k"
    assert v2.model == v1.model
    assert v2.raw["raw_inputs"] == v1.raw["raw_inputs"]
    assert v2.benchmark_id == "activsg10k-v2"
    assert v2.raw["benchmark"]["required_git_tag"] == "benchmark-10k-v2"
    assert v2.runtime["deadline_seconds"] == 300.0
    assert v2.runtime["verification_reserve_seconds"] == 45.0
    assert v2.raw["platforms"]["laptop_cpu"]["solver_session"] == (
        "persistent_incremental"
    )


def test_activsg10k_v3_adds_diagnostics_without_seeding_or_model_changes() -> None:
    v2 = load_config(ROOT / "configs" / "activsg10k-v2.json")
    v3 = load_config(ROOT / "configs" / "activsg10k-v3.json")
    assert v3.case_name == v2.case_name == "ACTIVSg10k"
    assert v3.model == v2.model
    assert v3.runtime == v2.runtime
    assert v3.raw["raw_inputs"] == v2.raw["raw_inputs"]
    assert v3.benchmark_id == "activsg10k-v3"
    assert v3.raw["benchmark"]["required_git_tag"] == "benchmark-10k-v3"
    laptop = v3.raw["platforms"]["laptop_cpu"]
    assert laptop["solver_session"] == "persistent_incremental"
    assert laptop["diagnostics"] == {
        "enabled": True,
        "mip_logging_interval_seconds": 1.0,
    }
    assert not any("seed" in key.casefold() for key in laptop)


def test_activsg10k_gap_experiment_registers_exact_unbounded_levels() -> None:
    expected = ("1e-3", "1e-4", "1e-5", "1e-6", "1e-7")
    for label in expected:
        config = load_config(ROOT / "configs" / f"activsg10k-gap-{label}.json")
        assert config.case_name == "ACTIVSg10k"
        assert config.benchmark_kind == "gap_sensitivity_experiment"
        assert config.runtime["deadline_seconds"] is None
        assert config.raw["benchmark"]["gap_label"] == label
        assert config.raw["benchmark"]["required_git_tag"] == (
            "experiment-10k-gap-v2"
        )
        assert config.raw["benchmark"]["experiment_suite_id"] == (
            "activsg10k-gap-sensitivity-v2"
        )
        assert config.raw["benchmark"]["pricing"]["enabled"] is True
        assert config.raw["platforms"]["laptop_cpu"]["solver_session"] == (
            "persistent_incremental"
        )


def test_activsg500_gap_experiment_registers_exact_bounded_levels() -> None:
    expected = ("1e-3", "1e-4", "1e-5", "1e-6", "1e-7")
    baseline = load_config(ROOT / "configs" / "activsg500-gap-1e-3.json")
    for label in expected:
        config = load_config(ROOT / "configs" / f"activsg500-gap-{label}.json")
        assert config.case_name == "ACTIVSg500"
        assert config.benchmark_kind == "gap_sensitivity_experiment"
        assert config.runtime["deadline_seconds"] == 1800.0
        assert config.runtime["verification_reserve_seconds"] == 120.0
        assert config.runtime["serialization_reserve_seconds"] == 15.0
        assert config.raw["benchmark"]["gap_label"] == label
        assert config.raw["benchmark"]["required_git_tag"] == (
            "experiment-500-gap-v1"
        )
        assert config.raw["benchmark"]["experiment_suite_id"] == (
            "activsg500-gap-sensitivity-v1"
        )
        assert config.raw["benchmark"]["pricing"]["enabled"] is True
        assert config.raw["platforms"]["laptop_cpu"]["solver_session"] == (
            "persistent_incremental"
        )
        assert config.raw["raw_inputs"] == baseline.raw["raw_inputs"]
        assert config.runtime == baseline.runtime
        assert config.raw["platforms"] == baseline.raw["platforms"]
        assert {
            key: value
            for key, value in config.model.items()
            if key != "mip_relative_gap_tolerance"
        } == {
            key: value
            for key, value in baseline.model.items()
            if key != "mip_relative_gap_tolerance"
        }


def test_activsg500_gpu_gap_experiment_registers_exact_bounded_levels() -> None:
    expected = ("1e-3", "1e-4", "1e-5", "1e-6", "1e-7")
    baseline = load_config(ROOT / "configs" / "activsg500-gpu-gap-1e-3.json")
    for label in expected:
        config = load_config(ROOT / "configs" / f"activsg500-gpu-gap-{label}.json")
        assert config.case_name == "ACTIVSg500"
        assert config.benchmark_kind == "gap_sensitivity_experiment"
        assert config.runtime["deadline_seconds"] == 1800.0
        assert config.runtime["verification_reserve_seconds"] == 120.0
        assert config.runtime["serialization_reserve_seconds"] == 15.0
        assert config.raw["benchmark"]["gap_label"] == label
        assert config.raw["benchmark"]["required_git_tag"] == (
            "experiment-500-gpu-gap-v1"
        )
        assert config.raw["benchmark"]["experiment_suite_id"] == (
            "activsg500-gpu-gap-sensitivity-v1"
        )
        assert config.raw["benchmark"]["pricing"] == {
            "enabled": True,
            "solver": "highs",
            "maximum_constraint_generation_rounds": 100,
        }
        assert set(config.raw["platforms"]) == {"dgx_spark"}
        spark = config.raw["platforms"]["dgx_spark"]
        assert spark["solver"] == "cuopt"
        assert spark["screening"] == "cupy"
        assert spark["solver_session"] == (
            "rebuild_each_round_with_partial_mip_start"
        )
        assert spark["pricing_solver"] == "highs"
        assert spark["highspy_version"] == "1.15.1"
        assert config.raw["raw_inputs"] == baseline.raw["raw_inputs"]
        assert config.runtime == baseline.runtime
        assert config.raw["platforms"] == baseline.raw["platforms"]
        assert {
            key: value
            for key, value in config.model.items()
            if key != "mip_relative_gap_tolerance"
        } == {
            key: value
            for key, value in baseline.model.items()
            if key != "mip_relative_gap_tolerance"
        }


def test_activsg2000_gap_experiment_registers_exact_bounded_levels() -> None:
    expected = ("1e-3", "1e-4", "1e-5", "1e-6", "1e-7")
    baseline = load_config(ROOT / "configs" / "activsg2000-gap-1e-3.json")
    for label in expected:
        config = load_config(ROOT / "configs" / f"activsg2000-gap-{label}.json")
        assert config.case_name == "ACTIVSg2000"
        assert config.benchmark_kind == "gap_sensitivity_experiment"
        assert config.runtime["deadline_seconds"] == 1800.0
        assert config.runtime["verification_reserve_seconds"] == 120.0
        assert config.runtime["serialization_reserve_seconds"] == 15.0
        assert config.raw["benchmark"]["gap_label"] == label
        assert config.raw["benchmark"]["required_git_tag"] == (
            "experiment-2000-gap-v1"
        )
        assert config.raw["benchmark"]["experiment_suite_id"] == (
            "activsg2000-gap-sensitivity-v1"
        )
        assert config.raw["benchmark"]["pricing"]["enabled"] is True
        assert config.raw["platforms"]["laptop_cpu"]["solver_session"] == (
            "persistent_incremental"
        )
        assert config.raw["raw_inputs"] == baseline.raw["raw_inputs"]
        assert config.runtime == baseline.runtime
        assert config.raw["platforms"] == baseline.raw["platforms"]
        assert {
            key: value
            for key, value in config.model.items()
            if key != "mip_relative_gap_tolerance"
        } == {
            key: value
            for key, value in baseline.model.items()
            if key != "mip_relative_gap_tolerance"
        }


def test_activsg2000_gpu_experiment_registers_only_1e_3() -> None:
    cpu = load_config(ROOT / "configs" / "activsg2000-gap-1e-3.json")
    gpu = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3.json")
    assert gpu.case_name == cpu.case_name == "ACTIVSg2000"
    assert gpu.model == cpu.model
    assert gpu.runtime == cpu.runtime
    assert gpu.raw["raw_inputs"] == cpu.raw["raw_inputs"]
    assert gpu.benchmark_kind == "gap_sensitivity_experiment"
    assert gpu.benchmark_id == "activsg2000-gpu-gap-v1-1e-3"
    assert gpu.raw["benchmark"]["gap_label"] == "1e-3"
    assert gpu.raw["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-gap-v1"
    )
    assert gpu.raw["benchmark"]["experiment_suite_id"] == (
        "activsg2000-gpu-gap-sensitivity-v1"
    )
    assert set(gpu.raw["platforms"]) == {"dgx_spark"}
    spark = gpu.raw["platforms"]["dgx_spark"]
    assert spark["solver"] == "cuopt"
    assert spark["screening"] == "cupy"
    assert spark["solver_session"] == "rebuild_each_round_with_partial_mip_start"
    assert spark["pricing_solver"] == "highs"
    assert spark["highspy_version"] == "1.15.1"


def test_activsg2000_gpu_v2_changes_only_gap_certificate_identity() -> None:
    cpu = load_config(ROOT / "configs" / "activsg2000-gap-1e-3.json")
    v1 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3.json")
    v2 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v2.json")

    assert v2.case_name == cpu.case_name == "ACTIVSg2000"
    assert v2.model == v1.model == cpu.model
    assert v2.runtime == v1.runtime == cpu.runtime
    assert v2.raw["raw_inputs"] == v1.raw["raw_inputs"] == cpu.raw["raw_inputs"]
    assert v2.benchmark_id == "activsg2000-gpu-gap-v2-1e-3"
    assert v2.raw["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-gap-v2"
    )
    assert v2.raw["benchmark"]["experiment_suite_id"] == (
        "activsg2000-gpu-gap-sensitivity-v2"
    )
    v2_spark = v2.raw["platforms"]["dgx_spark"]
    v1_spark = v1.raw["platforms"]["dgx_spark"]
    assert v2_spark["mip_acceptance_policy"] == (
        "finite_incumbent_bound_gap_and_native_residuals_v1"
    )
    assert v2_spark["mip_certificate_residual_tolerance"] == 1e-6
    assert {
        key: value
        for key, value in v2_spark.items()
        if key not in {"mip_acceptance_policy", "mip_certificate_residual_tolerance"}
    } == v1_spark


def test_activsg2000_gpu_v3_preserves_v2_math_and_policy() -> None:
    v2 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v2.json")
    v3 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v3.json")

    assert v3.case_name == v2.case_name == "ACTIVSg2000"
    assert v3.model == v2.model
    assert v3.runtime == v2.runtime
    assert v3.raw["raw_inputs"] == v2.raw["raw_inputs"]
    assert v3.raw["platforms"] == v2.raw["platforms"]
    assert v3.benchmark_id == "activsg2000-gpu-gap-v3-1e-3"
    assert v3.raw["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-gap-v3"
    )
    assert v3.raw["benchmark"]["experiment_suite_id"] == (
        "activsg2000-gpu-gap-sensitivity-v3"
    )
    assert v3.raw["benchmark"]["pricing"] == v2.raw["benchmark"]["pricing"]


def test_activsg2000_gpu_v4_changes_only_native_numerical_scaling() -> None:
    v3 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v3.json")
    v4 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v4.json")

    assert v4.case_name == v3.case_name == "ACTIVSg2000"
    assert v4.model == v3.model
    assert v4.runtime == v3.runtime
    assert v4.raw["raw_inputs"] == v3.raw["raw_inputs"]
    assert v4.benchmark_id == "activsg2000-gpu-gap-v4-1e-3"
    assert v4.raw["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-gap-v4"
    )
    assert v4.raw["benchmark"]["experiment_suite_id"] == (
        "activsg2000-gpu-gap-sensitivity-v4"
    )
    assert v4.raw["benchmark"]["pricing"] == v3.raw["benchmark"]["pricing"]
    v3_profile = v3.raw["platforms"]["dgx_spark"]
    v4_profile = v4.raw["platforms"]["dgx_spark"]
    assert v4_profile["native_scaling_mode"] == "power_system_per_unit_v1"
    assert {
        key: value
        for key, value in v4_profile.items()
        if key != "native_scaling_mode"
    } == v3_profile


def test_activsg2000_gpu_v5_changes_only_runtime_and_frozen_identity() -> None:
    v4 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v4.json")
    v5 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v5.json")

    assert v5.case_name == v4.case_name == "ACTIVSg2000"
    assert v5.model == v4.model
    assert v5.raw["raw_inputs"] == v4.raw["raw_inputs"]
    assert v5.raw["platforms"] == v4.raw["platforms"]
    assert v5.runtime == {
        "deadline_seconds": 1035.0,
        "verification_reserve_seconds": 120.0,
        "serialization_reserve_seconds": 15.0,
        "maximum_constraint_generation_rounds": 100,
    }
    assert v5.benchmark_id == "activsg2000-gpu-gap-v5-1e-3"
    assert v5.raw["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-gap-v5"
    )
    assert v5.raw["benchmark"]["experiment_suite_id"] == (
        "activsg2000-gpu-gap-sensitivity-v5"
    )
    assert v5.raw["benchmark"]["initialization"] == {
        "external_cpu_mip_start": False,
        "round_1": "cold",
        "later_rounds": "prior_gpu_integer_commitment_only",
    }
    assert v5.raw["benchmark"]["pricing"] == v4.raw["benchmark"]["pricing"]


def test_activsg2000_gpu_v6_changes_only_cuopt_pdlp_policy_and_identity() -> None:
    v5 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v5.json")
    v6 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v6.json")

    assert v6.case_name == v5.case_name == "ACTIVSg2000"
    assert v6.model == v5.model
    assert v6.runtime == v5.runtime
    assert v6.raw["raw_inputs"] == v5.raw["raw_inputs"]
    assert v6.benchmark_id == "activsg2000-gpu-gap-v6-1e-3"
    assert v6.raw["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-gap-v6"
    )
    assert v6.raw["benchmark"]["experiment_suite_id"] == (
        "activsg2000-gpu-gap-sensitivity-v6"
    )
    assert v6.raw["benchmark"]["initialization"] == (
        v5.raw["benchmark"]["initialization"]
    )
    assert v6.raw["benchmark"]["pricing"] == v5.raw["benchmark"]["pricing"]
    v5_profile = v5.raw["platforms"]["dgx_spark"]
    v6_profile = v6.raw["platforms"]["dgx_spark"]
    assert v6_profile["cuopt_pdlp_profile"] == PDLP_PROFILE
    assert {
        key: value
        for key, value in v6_profile.items()
        if key != "cuopt_pdlp_profile"
    } == v5_profile


def test_activsg2000_gpu_v7_changes_only_runtime_and_frozen_identity() -> None:
    v6 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v6.json")
    v7 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v7.json")

    assert v7.case_name == v6.case_name == "ACTIVSg2000"
    assert v7.model == v6.model
    assert v7.raw["raw_inputs"] == v6.raw["raw_inputs"]
    assert v7.raw["platforms"] == v6.raw["platforms"]
    assert v7.runtime == {
        "deadline_seconds": 1935.0,
        "verification_reserve_seconds": 120.0,
        "serialization_reserve_seconds": 15.0,
        "maximum_constraint_generation_rounds": 100,
    }
    assert v7.benchmark_id == "activsg2000-gpu-gap-v7-1e-3"
    assert v7.raw["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-gap-v7"
    )
    assert v7.raw["benchmark"]["experiment_suite_id"] == (
        "activsg2000-gpu-gap-sensitivity-v7"
    )
    assert v7.raw["benchmark"]["initialization"] == (
        v6.raw["benchmark"]["initialization"]
    )
    assert v7.raw["benchmark"]["pricing"] == v6.raw["benchmark"]["pricing"]
    assert v7.raw["benchmark"]["runtime_change"] == {
        "comparison_baseline": "activsg2000-gpu-gap-v6-1e-3",
        "only_change": (
            "double_cumulative_solver_allowance_from_900_to_1800_seconds"
        ),
    }


def test_activsg2000_gpu_v8_changes_only_bugfix_identity() -> None:
    v7 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v7.json")
    v8 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v8.json")

    assert v8.case_name == v7.case_name == "ACTIVSg2000"
    assert v8.model == v7.model
    assert v8.runtime == v7.runtime
    assert v8.raw["raw_inputs"] == v7.raw["raw_inputs"]
    assert v8.raw["platforms"] == v7.raw["platforms"]
    assert v8.benchmark_id == "activsg2000-gpu-gap-v8-1e-3"
    assert v8.raw["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-gap-v8"
    )
    assert v8.raw["benchmark"]["experiment_suite_id"] == (
        "activsg2000-gpu-gap-sensitivity-v8"
    )
    assert v8.raw["benchmark"]["initialization"] == (
        v7.raw["benchmark"]["initialization"]
    )
    assert v8.raw["benchmark"]["pricing"] == v7.raw["benchmark"]["pricing"]
    assert v8.raw["benchmark"]["bugfix_change"] == {
        "comparison_baseline": "activsg2000-gpu-gap-v7-1e-3",
        "mip_start_policy": "presolve_off_original_space_readback_v1",
        "native_log_policy": "fail_on_mip_start_rejection_or_native_error_v1",
        "constraint_generation_policy": (
            "add_violations_and_resolve_even_when_current_gap_is_uncertified"
        ),
    }


def test_1935_second_outer_deadline_is_authorized_only_for_v7_and_v8(
    tmp_path: Path,
) -> None:
    v6 = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v6.json")
    forged = json.loads(json.dumps(v6.raw))
    forged["runtime"]["deadline_seconds"] = 1935.0
    path = tmp_path / "configs" / "forged-v6.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(ScopeViolation, match=r"\(0, 1800\]"):
        load_config(path)


def test_activsg2000_seeded_round2_diagnostic_is_exactly_registered() -> None:
    config = load_config(
        ROOT / "configs" / "activsg2000-gpu-round2-cpu-seed-diagnostic-v1.json"
    )

    validate_diagnostic_identity(config)
    assert config.benchmark_kind == "seeded_round2_diagnostic"
    assert config.runtime["deadline_seconds"] == 1800.0
    assert config.raw["diagnostic"]["mip_start_mode"] == "all_columns"
    assert config.raw["diagnostic"]["console_logging"] is True
    assert config.raw["diagnostic"]["constraint_generation_enabled"] is False


def test_activsg2000_seeded_round2_v2_registers_exact_bound_projection() -> None:
    config = load_config(
        ROOT / "configs" / "activsg2000-gpu-round2-cpu-seed-diagnostic-v2.json"
    )

    validate_diagnostic_identity(config)
    assert config.benchmark_id == (
        "activsg2000-gpu-round2-cpu-seed-diagnostic-v2"
    )
    assert config.raw["diagnostic"]["mip_start_bound_policy"] == (
        "project_numerical_excess_to_exact_bound"
    )


def test_activsg2000_seeded_round2_v3_registers_per_unit_native_scaling() -> None:
    config = load_config(
        ROOT / "configs" / "activsg2000-gpu-round2-cpu-seed-diagnostic-v3.json"
    )

    validate_diagnostic_identity(config)
    assert config.benchmark_id == (
        "activsg2000-gpu-round2-cpu-seed-diagnostic-v3"
    )
    assert config.raw["diagnostic"]["mip_start_bound_policy"] == (
        "project_numerical_excess_to_exact_bound"
    )
    assert config.raw["diagnostic"]["native_scaling_mode"] == (
        "power_system_per_unit_v1"
    )
