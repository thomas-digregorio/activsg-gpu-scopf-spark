"""One-shot, frozen gap-sensitivity experiment controller."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import RunConfig
from .environment import validate_platform
from .errors import ProvenanceError, ScopfError
from .official import frozen_identity
from .paths import guard_output_path
from .provenance import write_json_atomic

GAP_LABELS = {
    1e-3: "1e-3",
    1e-4: "1e-4",
    1e-5: "1e-5",
    1e-6: "1e-6",
    1e-7: "1e-7",
}

EXPERIMENT_SUITES: dict[str, dict[str, Any]] = {
    "activsg10k-gap-sensitivity-v2": {
        "case_name": "ACTIVSg10k",
        "benchmark_prefix": "activsg10k-gap-v2",
        "required_git_tag": "experiment-10k-gap-v2",
        "platform": "laptop_cpu",
        "required_profile": {
            "solver": "highs",
            "screening": "numpy",
            "solver_session": "persistent_incremental",
        },
        "deadline_seconds": None,
        "verification_reserve_seconds": 0.0,
        "serialization_reserve_seconds": 0.0,
    },
    "activsg500-gap-sensitivity-v1": {
        "case_name": "ACTIVSg500",
        "benchmark_prefix": "activsg500-gap-v1",
        "required_git_tag": "experiment-500-gap-v1",
        "platform": "laptop_cpu",
        "required_profile": {
            "solver": "highs",
            "screening": "numpy",
            "solver_session": "persistent_incremental",
        },
        "deadline_seconds": 1800.0,
        "verification_reserve_seconds": 120.0,
        "serialization_reserve_seconds": 15.0,
    },
    "activsg2000-gap-sensitivity-v1": {
        "case_name": "ACTIVSg2000",
        "benchmark_prefix": "activsg2000-gap-v1",
        "required_git_tag": "experiment-2000-gap-v1",
        "platform": "laptop_cpu",
        "required_profile": {
            "solver": "highs",
            "screening": "numpy",
            "solver_session": "persistent_incremental",
        },
        "deadline_seconds": 1800.0,
        "verification_reserve_seconds": 120.0,
        "serialization_reserve_seconds": 15.0,
    },
    "activsg500-gpu-gap-sensitivity-v1": {
        "case_name": "ACTIVSg500",
        "benchmark_prefix": "activsg500-gpu-gap-v1",
        "required_git_tag": "experiment-500-gpu-gap-v1",
        "platform": "dgx_spark",
        "required_profile": {
            "solver": "cuopt",
            "screening": "cupy",
            "solver_session": "rebuild_each_round_with_partial_mip_start",
            "pricing_solver": "highs",
            "highspy_version": "1.15.1",
        },
        "deadline_seconds": 1800.0,
        "verification_reserve_seconds": 120.0,
        "serialization_reserve_seconds": 15.0,
    },
    "activsg2000-gpu-gap-sensitivity-v1": {
        "case_name": "ACTIVSg2000",
        "benchmark_prefix": "activsg2000-gpu-gap-v1",
        "required_git_tag": "experiment-2000-gpu-gap-v1",
        "allowed_gap_labels": ("1e-3",),
        "platform": "dgx_spark",
        "required_profile": {
            "solver": "cuopt",
            "screening": "cupy",
            "solver_session": "rebuild_each_round_with_partial_mip_start",
            "pricing_solver": "highs",
            "highspy_version": "1.15.1",
        },
        "deadline_seconds": 1800.0,
        "verification_reserve_seconds": 120.0,
        "serialization_reserve_seconds": 15.0,
    },
    "activsg2000-gpu-gap-sensitivity-v2": {
        "case_name": "ACTIVSg2000",
        "benchmark_prefix": "activsg2000-gpu-gap-v2",
        "required_git_tag": "experiment-2000-gpu-gap-v2",
        "allowed_gap_labels": ("1e-3",),
        "platform": "dgx_spark",
        "required_profile": {
            "solver": "cuopt",
            "screening": "cupy",
            "solver_session": "rebuild_each_round_with_partial_mip_start",
            "pricing_solver": "highs",
            "highspy_version": "1.15.1",
            "mip_acceptance_policy": (
                "finite_incumbent_bound_gap_and_native_residuals_v1"
            ),
            "mip_certificate_residual_tolerance": 1e-6,
        },
        "deadline_seconds": 1800.0,
        "verification_reserve_seconds": 120.0,
        "serialization_reserve_seconds": 15.0,
    },
    "activsg2000-gpu-gap-sensitivity-v3": {
        "case_name": "ACTIVSg2000",
        "benchmark_prefix": "activsg2000-gpu-gap-v3",
        "required_git_tag": "experiment-2000-gpu-gap-v3",
        "allowed_gap_labels": ("1e-3",),
        "platform": "dgx_spark",
        "required_profile": {
            "solver": "cuopt",
            "screening": "cupy",
            "solver_session": "rebuild_each_round_with_partial_mip_start",
            "pricing_solver": "highs",
            "highspy_version": "1.15.1",
            "mip_acceptance_policy": (
                "finite_incumbent_bound_gap_and_native_residuals_v1"
            ),
            "mip_certificate_residual_tolerance": 1e-6,
        },
        "deadline_seconds": 1800.0,
        "verification_reserve_seconds": 120.0,
        "serialization_reserve_seconds": 15.0,
    },
    "activsg2000-gpu-gap-sensitivity-v4": {
        "case_name": "ACTIVSg2000",
        "benchmark_prefix": "activsg2000-gpu-gap-v4",
        "required_git_tag": "experiment-2000-gpu-gap-v4",
        "allowed_gap_labels": ("1e-3",),
        "platform": "dgx_spark",
        "required_profile": {
            "solver": "cuopt",
            "screening": "cupy",
            "solver_session": "rebuild_each_round_with_partial_mip_start",
            "pricing_solver": "highs",
            "highspy_version": "1.15.1",
            "mip_acceptance_policy": (
                "finite_incumbent_bound_gap_and_native_residuals_v1"
            ),
            "mip_certificate_residual_tolerance": 1e-6,
            "native_scaling_mode": "power_system_per_unit_v1",
        },
        "deadline_seconds": 1800.0,
        "verification_reserve_seconds": 120.0,
        "serialization_reserve_seconds": 15.0,
    },
    "activsg2000-gpu-gap-sensitivity-v5": {
        "case_name": "ACTIVSg2000",
        "benchmark_prefix": "activsg2000-gpu-gap-v5",
        "required_git_tag": "experiment-2000-gpu-gap-v5",
        "allowed_gap_labels": ("1e-3",),
        "platform": "dgx_spark",
        "required_profile": {
            "solver": "cuopt",
            "screening": "cupy",
            "solver_session": "rebuild_each_round_with_partial_mip_start",
            "pricing_solver": "highs",
            "highspy_version": "1.15.1",
            "mip_acceptance_policy": (
                "finite_incumbent_bound_gap_and_native_residuals_v1"
            ),
            "mip_certificate_residual_tolerance": 1e-6,
            "native_scaling_mode": "power_system_per_unit_v1",
        },
        # 900 seconds remain for loading plus all cuOpt rounds after preserving
        # the existing independent-verification and serialization reserves.
        "deadline_seconds": 1035.0,
        "verification_reserve_seconds": 120.0,
        "serialization_reserve_seconds": 15.0,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read experiment evidence {path}: {exc}") from exc


def _experiment_identity(config: RunConfig) -> tuple[str, str]:
    suite_id = str(config.raw["benchmark"].get("experiment_suite_id", ""))
    suite = EXPERIMENT_SUITES.get(suite_id)
    if suite is None:
        raise ScopfError(f"Unregistered experiment suite {suite_id!r}")
    if config.case_name != suite["case_name"]:
        raise ScopfError(
            f"Experiment suite {suite_id!r} requires {suite['case_name']}, "
            f"observed {config.case_name}"
        )
    if config.benchmark_kind != "gap_sensitivity_experiment":
        raise ScopfError("Configuration is not a gap-sensitivity experiment")
    expected_deadline = suite["deadline_seconds"]
    observed_deadline = config.runtime.get("deadline_seconds")
    if expected_deadline is None:
        if observed_deadline is not None:
            raise ScopfError("This gap-sensitivity suite must have no deadline")
    elif observed_deadline is None or float(observed_deadline) != expected_deadline:
        raise ScopfError(
            "Gap-sensitivity deadline changed: expected "
            f"{expected_deadline}, observed {observed_deadline!r}"
        )
    for key in ("verification_reserve_seconds", "serialization_reserve_seconds"):
        expected = suite[key]
        if float(config.runtime.get(key, -1.0)) != expected:
            raise ScopfError(
                f"Gap experiment changed registered runtime value {key}: "
                f"expected {expected!r}, observed {config.runtime.get(key)!r}"
            )
    gap = float(config.model["mip_relative_gap_tolerance"])
    label = str(config.raw["benchmark"].get("gap_label", ""))
    if gap not in GAP_LABELS or label != GAP_LABELS[gap]:
        raise ScopfError(f"Unregistered gap experiment identity: gap={gap}, label={label!r}")
    allowed_labels = suite.get("allowed_gap_labels")
    if allowed_labels is not None and label not in allowed_labels:
        raise ScopfError(
            f"Experiment suite {suite_id!r} does not authorize gap {label!r}"
        )
    expected_benchmark_id = f"{suite['benchmark_prefix']}-{label}"
    if config.benchmark_id != expected_benchmark_id:
        raise ScopfError(
            f"Expected benchmark id {expected_benchmark_id!r}, "
            f"observed {config.benchmark_id!r}"
        )
    required_tag = str(config.raw["benchmark"].get("required_git_tag", ""))
    if required_tag != suite["required_git_tag"]:
        raise ScopfError(
            f"Expected frozen tag {suite['required_git_tag']!r}, observed {required_tag!r}"
        )
    pricing = config.raw["benchmark"].get("pricing", {})
    if not bool(pricing.get("enabled", False)):
        raise ScopfError("Gap sensitivity requires fixed-commitment pricing")
    required_model_values = {
        "interval_hours": 1.0,
        "pwl_segments": 10,
        "model_residual_tolerance_pu": 1e-6,
        "security_violation_tolerance_pu": 1e-5,
        "lodf_validation_tolerance_pu": 1e-9,
        "lodf_validation_columns": 3,
        "lodf_build_chunk_columns": 256,
        "screen_chunk_columns": 256,
    }
    for key, expected in required_model_values.items():
        if config.model.get(key) != expected:
            raise ScopfError(
                f"Gap experiment changed registered model value {key}: "
                f"expected {expected!r}, observed {config.model.get(key)!r}"
            )
    platform_name = str(suite["platform"])
    profile = config.raw["platforms"].get(platform_name, {})
    required_profile = suite["required_profile"]
    for key, expected in required_profile.items():
        if profile.get(key) != expected:
            raise ScopfError(
                f"Gap experiment changed {platform_name} profile {key}: "
                f"expected {expected!r}, observed {profile.get(key)!r}"
            )
    expected_pricing_solver = required_profile.get("pricing_solver", "highs")
    observed_pricing_solver = pricing.get("solver", "highs")
    if observed_pricing_solver != expected_pricing_solver:
        raise ScopfError(
            "Gap experiment changed the fixed-commitment pricing solver: "
            f"expected {expected_pricing_solver!r}, observed "
            f"{observed_pricing_solver!r}"
        )
    return suite_id, label


def _require_prior_success(registry: dict[str, Any], gap_label: str) -> None:
    """Fail closed if an earlier gap is absent, incomplete, or unpriced."""

    ordered_labels = list(GAP_LABELS.values())
    requested_index = ordered_labels.index(gap_label)
    for prior_label in ordered_labels[:requested_index]:
        prior = registry["runs"].get(prior_label)
        if prior is None:
            raise ScopfError(
                f"Gap {gap_label} is blocked because prior gap {prior_label} was not run"
            )
        if prior.get("status") != "optimal_verified":
            raise ScopfError(
                f"Gap {gap_label} is blocked because prior gap {prior_label} ended as "
                f"{prior.get('status', 'unknown')}"
            )
        if prior.get("pricing_status") != "optimal_secure_fixed_commitment_lp":
            raise ScopfError(
                f"Gap {gap_label} is blocked because prior gap {prior_label} "
                "does not have accepted fixed-commitment pricing"
            )


def _registry_path(config: RunConfig, suite_id: str) -> Path:
    return guard_output_path(
        config.root / "results" / "experiments" / f"{suite_id}-run-registry.json"
    )


def _write_worker_console(
    path: Path,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> None:
    """Persist captured worker output so native solver logging is not discarded."""

    def decode(value: str | bytes | None) -> str:
        return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(decode(stdout) + decode(stderr), encoding="utf-8")


def run_one_shot_gap_experiment(
    config: RunConfig,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Launch one immutable gap-level experiment under its registered deadline."""

    suite_id, gap_label = _experiment_identity(config)
    suite = EXPERIMENT_SUITES[suite_id]
    platform_name = str(suite["platform"])
    validate_platform(config, platform_name)
    output = guard_output_path(output_path)
    experiment_root = (config.root / "results" / "experiments").resolve()
    if not output.resolve().is_relative_to(experiment_root):
        raise ScopfError("Gap-experiment output must be under results/experiments")
    if output.exists():
        raise ScopfError(f"Gap-experiment output already exists: {output}")

    identity = frozen_identity(config)
    registry_path = _registry_path(config, suite_id)
    registry = (
        _read_json(registry_path)
        if registry_path.exists()
        else {
            "schema_version": "1.0.0",
            "experiment_suite_id": suite_id,
            "runs": {},
        }
    )
    _require_prior_success(registry, gap_label)
    if gap_label in registry["runs"]:
        prior = registry["runs"][gap_label]
        raise ScopfError(
            f"Gap {gap_label} already has a one-shot record with status "
            f"{prior.get('status', 'unknown')}; rerun is forbidden"
        )

    checkpoint = guard_output_path(
        config.root
        / "results"
        / "checkpoints"
        / f"{config.benchmark_id}-{platform_name}.json"
    )
    if checkpoint.exists():
        raise ScopfError(f"Gap-experiment checkpoint already exists: {checkpoint}")
    diagnostic_root = config.root / "results" / "diagnostics"
    worker_console_path = guard_output_path(
        diagnostic_root
        / f"{config.benchmark_id}-{platform_name}-worker-console.log"
    )
    stale_diagnostics = [
        diagnostic_root / f"{config.benchmark_id}-{platform_name}-events.jsonl",
        diagnostic_root / f"{config.benchmark_id}-{platform_name}-highs.log",
        worker_console_path,
    ]
    if any(path.exists() for path in stale_diagnostics):
        raise ScopfError("Gap-experiment diagnostic output already exists")
    registry["runs"][gap_label] = {
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "output": str(output),
        "checkpoint": str(checkpoint),
        "frozen_identity": identity,
        "mip_relative_gap_tolerance": float(
            config.model["mip_relative_gap_tolerance"]
        ),
    }
    write_json_atomic(registry, registry_path)

    command = [
        sys.executable,
        "-m",
        "activsg_scopf.cli",
        "_worker",
        "--config",
        str(config.path),
        "--output",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--platform",
        platform_name,
    ]
    deadline_value = config.runtime.get("deadline_seconds")
    deadline_seconds = None if deadline_value is None else float(deadline_value)
    if deadline_seconds is not None:
        command.extend(["--deadline-seconds", str(deadline_seconds)])
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
        _write_worker_console(
            worker_console_path, completed.stdout, completed.stderr
        )
        total_wall = time.perf_counter() - started
        if output.exists():
            result = _read_json(output)
        else:
            result = _read_json(checkpoint) if checkpoint.exists() else {}
            result.update(
                {
                    "status": "failed_worker_without_result",
                    "worker_returncode": completed.returncode,
                    "worker_stdout": completed.stdout[-4000:],
                    "worker_stderr": completed.stderr[-4000:],
                }
            )
    except subprocess.TimeoutExpired as exc:
        total_wall = time.perf_counter() - started
        _write_worker_console(worker_console_path, exc.stdout, exc.stderr)
        result = _read_json(output) if output.exists() else (
            _read_json(checkpoint) if checkpoint.exists() else {}
        )

        def tail(value: str | bytes | None) -> str:
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            return (value or "")[-4000:]

        result.update(
            {
                "status": "hard_deadline_exceeded",
                "worker_timeout_seconds": deadline_seconds,
                "worker_stdout": tail(exc.stdout),
                "worker_stderr": tail(exc.stderr),
            }
        )
    result.update(
        {
            "official": False,
            "experiment": True,
            "one_shot": True,
            "platform": platform_name,
            "experiment_suite_id": suite_id,
            "gap_label": gap_label,
            "frozen_identity": identity,
            "deadline_seconds": deadline_seconds,
            "worker_console_log": str(worker_console_path.relative_to(config.root)),
            "total_wall_time_seconds": total_wall,
            "benchmark_boundary": (
                "worker launch through raw loading, all MIP solve/screen rounds, "
                "independent verification, fixed-commitment pricing, and result "
                "serialization; "
                + (
                    "no wall-clock deadline"
                    if deadline_seconds is None
                    else f"hard {deadline_seconds:g}-second wall-clock deadline"
                )
            ),
        }
    )
    if (
        deadline_seconds is not None
        and total_wall > deadline_seconds
        and result.get("status") == "optimal_verified"
    ):
        result["status"] = "failed_end_to_end_deadline"
    write_json_atomic(result, output)

    record = registry["runs"][gap_label]
    record.update(
        {
            "status": str(result.get("status", "unknown")),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "total_wall_time_seconds": total_wall,
            "objective": result.get("objective"),
            "bound": result.get("bound"),
            "mip_gap": result.get("mip_gap"),
            "pricing_status": result.get("pricing", {}).get("status"),
        }
    )
    write_json_atomic(registry, registry_path)
    return result
