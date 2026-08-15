import copy
import json
import subprocess
from pathlib import Path

import pytest

from activsg_scopf.config import RunConfig, load_config
from activsg_scopf.errors import ScopfError
from activsg_scopf.experiments import (
    _experiment_identity,
    _require_prior_success,
    run_one_shot_gap_experiment,
)

ROOT = Path(__file__).resolve().parents[1]


def test_activsg500_gap_identity_is_bounded_and_registered() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gap-1e-3.json")
    assert _experiment_identity(config) == ("activsg500-gap-sensitivity-v1", "1e-3")


def test_activsg500_gpu_gap_identity_is_bounded_and_registered() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-gap-1e-3.json")
    assert _experiment_identity(config) == (
        "activsg500-gpu-gap-sensitivity-v1",
        "1e-3",
    )


def test_activsg2000_gap_identity_is_bounded_and_registered() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gap-1e-3.json")
    assert _experiment_identity(config) == (
        "activsg2000-gap-sensitivity-v1",
        "1e-3",
    )


def test_activsg2000_gpu_gap_identity_authorizes_only_1e_3() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3.json")
    assert _experiment_identity(config) == (
        "activsg2000-gpu-gap-sensitivity-v1",
        "1e-3",
    )
    forged_raw = copy.deepcopy(config.raw)
    forged_raw["model"]["mip_relative_gap_tolerance"] = 1e-4
    forged_raw["benchmark"]["gap_label"] = "1e-4"
    forged_raw["benchmark"]["id"] = "activsg2000-gpu-gap-v1-1e-4"
    forged = RunConfig(path=config.path, root=config.root, raw=forged_raw)
    with pytest.raises(ScopfError, match="does not authorize gap"):
        _experiment_identity(forged)


def test_activsg2000_gpu_v2_authorizes_only_1e_3() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v2.json")
    assert _experiment_identity(config) == (
        "activsg2000-gpu-gap-sensitivity-v2",
        "1e-3",
    )
    forged_raw = copy.deepcopy(config.raw)
    forged_raw["model"]["mip_relative_gap_tolerance"] = 1e-4
    forged_raw["benchmark"]["gap_label"] = "1e-4"
    forged_raw["benchmark"]["id"] = "activsg2000-gpu-gap-v2-1e-4"
    forged = RunConfig(path=config.path, root=config.root, raw=forged_raw)
    with pytest.raises(ScopfError, match="does not authorize gap"):
        _experiment_identity(forged)


def test_activsg2000_gpu_v3_authorizes_only_1e_3() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v3.json")
    assert _experiment_identity(config) == (
        "activsg2000-gpu-gap-sensitivity-v3",
        "1e-3",
    )
    forged_raw = copy.deepcopy(config.raw)
    forged_raw["model"]["mip_relative_gap_tolerance"] = 1e-4
    forged_raw["benchmark"]["gap_label"] = "1e-4"
    forged_raw["benchmark"]["id"] = "activsg2000-gpu-gap-v3-1e-4"
    forged = RunConfig(path=config.path, root=config.root, raw=forged_raw)
    with pytest.raises(ScopfError, match="does not authorize gap"):
        _experiment_identity(forged)


def test_activsg2000_gpu_v5_registers_900_second_solve_budget() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v5.json")

    assert _experiment_identity(config) == (
        "activsg2000-gpu-gap-sensitivity-v5",
        "1e-3",
    )
    runtime = config.runtime
    available = (
        runtime["deadline_seconds"]
        - runtime["verification_reserve_seconds"]
        - runtime["serialization_reserve_seconds"]
    )
    assert available == 900.0
    assert config.raw["benchmark"]["initialization"]["external_cpu_mip_start"] is False


def test_activsg2000_gpu_v6_registers_pdlp_and_same_900_second_budget() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v6.json")

    assert _experiment_identity(config) == (
        "activsg2000-gpu-gap-sensitivity-v6",
        "1e-3",
    )
    runtime = config.runtime
    available = (
        runtime["deadline_seconds"]
        - runtime["verification_reserve_seconds"]
        - runtime["serialization_reserve_seconds"]
    )
    assert available == 900.0
    profile = config.raw["platforms"]["dgx_spark"]
    assert profile["cuopt_pdlp_profile"] == {
        "method": "pdlp",
        "solver_mode": "stable3",
        "precision": "fp64",
        "batch_strong_branching": True,
        "batch_reliability_branching": True,
        "reliability_branching_factor": 1,
    }


def test_activsg2000_gpu_v7_registers_1800_second_solve_budget() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v7.json")

    assert _experiment_identity(config) == (
        "activsg2000-gpu-gap-sensitivity-v7",
        "1e-3",
    )
    runtime = config.runtime
    available = (
        runtime["deadline_seconds"]
        - runtime["verification_reserve_seconds"]
        - runtime["serialization_reserve_seconds"]
    )
    assert available == 1800.0
    assert config.raw["platforms"]["dgx_spark"]["cuopt_pdlp_profile"] == {
        "method": "pdlp",
        "solver_mode": "stable3",
        "precision": "fp64",
        "batch_strong_branching": True,
        "batch_reliability_branching": True,
        "reliability_branching_factor": 1,
    }


def test_activsg2000_gpu_v8_registers_fixed_start_and_same_budget() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v8.json")

    assert _experiment_identity(config) == (
        "activsg2000-gpu-gap-sensitivity-v8",
        "1e-3",
    )
    runtime = config.runtime
    available = (
        runtime["deadline_seconds"]
        - runtime["verification_reserve_seconds"]
        - runtime["serialization_reserve_seconds"]
    )
    assert available == 1800.0
    assert config.raw["benchmark"]["bugfix_change"]["mip_start_policy"] == (
        "presolve_off_original_space_readback_v1"
    )


def test_next_gap_requires_prior_success_and_pricing() -> None:
    successful = {
        "runs": {
            "1e-3": {
                "status": "optimal_verified",
                "pricing_status": "optimal_secure_fixed_commitment_lp",
            }
        }
    }
    _require_prior_success(successful, "1e-4")

    with pytest.raises(ScopfError, match="was not run"):
        _require_prior_success({"runs": {}}, "1e-4")
    with pytest.raises(ScopfError, match="hard_deadline_exceeded"):
        _require_prior_success(
            {
                "runs": {
                    "1e-3": {
                        "status": "hard_deadline_exceeded",
                        "pricing_status": None,
                    }
                }
            },
            "1e-4",
        )
    with pytest.raises(ScopfError, match="does not have accepted"):
        _require_prior_success(
            {"runs": {"1e-3": {"status": "optimal_verified"}}},
            "1e-4",
        )


def test_bounded_gap_controller_serializes_hard_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = load_config(ROOT / "configs" / "activsg500-gap-1e-3.json")
    config_path = tmp_path / "configs" / "activsg500-gap-1e-3.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(source.raw), encoding="utf-8")
    config = RunConfig(path=config_path, root=tmp_path, raw=source.raw)
    output = tmp_path / "results" / "experiments" / "result.json"
    output.parent.mkdir(parents=True)

    monkeypatch.setattr("activsg_scopf.experiments.validate_platform", lambda *_: None)
    monkeypatch.setattr(
        "activsg_scopf.experiments.frozen_identity",
        lambda *_: {"commit": "frozen", "tag": "experiment-500-gap-v1", "config_sha256": "x"},
    )

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="worker", timeout=1800.0, output="partial")

    monkeypatch.setattr("activsg_scopf.experiments.subprocess.run", timeout)
    result = run_one_shot_gap_experiment(config, output_path=output)
    serialized = json.loads(output.read_text(encoding="utf-8"))
    registry = json.loads(
        (
            tmp_path
            / "results"
            / "experiments"
            / "activsg500-gap-sensitivity-v1-run-registry.json"
        ).read_text(encoding="utf-8")
    )

    assert result["status"] == "hard_deadline_exceeded"
    assert serialized["status"] == "hard_deadline_exceeded"
    assert serialized["worker_timeout_seconds"] == 1800.0
    assert serialized["worker_stdout"] == "partial"
    worker_console = (
        tmp_path
        / "results"
        / "diagnostics"
        / "activsg500-gap-v1-1e-3-laptop_cpu-worker-console.log"
    )
    assert worker_console.read_text(encoding="utf-8") == "partial"
    assert serialized["worker_console_log"] == str(
        worker_console.relative_to(tmp_path)
    )
    assert registry["runs"]["1e-3"]["status"] == "hard_deadline_exceeded"


def test_gpu_gap_controller_launches_registered_spark_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = load_config(ROOT / "configs" / "activsg500-gpu-gap-1e-3.json")
    config_path = tmp_path / "configs" / "activsg500-gpu-gap-1e-3.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(source.raw), encoding="utf-8")
    config = RunConfig(path=config_path, root=tmp_path, raw=source.raw)
    output = (
        tmp_path
        / "results"
        / "experiments"
        / "activsg500-gpu-gap-v1-1e-3-dgx-spark.json"
    )
    output.parent.mkdir(parents=True)
    validated_platforms: list[str] = []
    worker_commands: list[list[str]] = []

    def validate(_config: RunConfig, platform_name: str) -> None:
        validated_platforms.append(platform_name)

    monkeypatch.setattr("activsg_scopf.experiments.validate_platform", validate)
    monkeypatch.setattr(
        "activsg_scopf.experiments.frozen_identity",
        lambda *_: {
            "commit": "frozen",
            "tag": "experiment-500-gpu-gap-v1",
            "config_sha256": "x",
        },
    )

    def timeout(command: list[str], **kwargs: object) -> None:
        worker_commands.append(command)
        raise subprocess.TimeoutExpired(cmd=command, timeout=1800.0)

    monkeypatch.setattr("activsg_scopf.experiments.subprocess.run", timeout)
    result = run_one_shot_gap_experiment(config, output_path=output)

    assert validated_platforms == ["dgx_spark"]
    assert result["platform"] == "dgx_spark"
    assert result["status"] == "hard_deadline_exceeded"
    assert worker_commands
    platform_index = worker_commands[0].index("--platform")
    assert worker_commands[0][platform_index + 1] == "dgx_spark"
    checkpoint_index = worker_commands[0].index("--checkpoint")
    assert worker_commands[0][checkpoint_index + 1].endswith(
        "activsg500-gpu-gap-v1-1e-3-dgx_spark.json"
    )
