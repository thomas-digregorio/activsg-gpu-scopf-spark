from pathlib import Path

from activsg_scopf.config import load_config
from activsg_scopf.seeded_diagnostic import validate_diagnostic_identity

ROOT = Path(__file__).resolve().parents[1]


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
