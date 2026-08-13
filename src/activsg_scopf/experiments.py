"""One-shot, frozen, unbounded laptop experiment controller."""

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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read experiment evidence {path}: {exc}") from exc


def _experiment_identity(config: RunConfig) -> tuple[str, str]:
    if config.case_name != "ACTIVSg10k":
        raise ScopfError("Gap sensitivity is registered only for ACTIVSg10k")
    if config.benchmark_kind != "gap_sensitivity_experiment":
        raise ScopfError("Configuration is not a gap-sensitivity experiment")
    if config.runtime.get("deadline_seconds") is not None:
        raise ScopfError("Gap-sensitivity experiment must have no deadline")
    gap = float(config.model["mip_relative_gap_tolerance"])
    label = str(config.raw["benchmark"].get("gap_label", ""))
    if gap not in GAP_LABELS or label != GAP_LABELS[gap]:
        raise ScopfError(f"Unregistered gap experiment identity: gap={gap}, label={label!r}")
    suite_id = str(config.raw["benchmark"].get("experiment_suite_id", ""))
    if suite_id != "activsg10k-gap-sensitivity-v1":
        raise ScopfError(f"Unregistered experiment suite {suite_id!r}")
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
    profile = config.raw["platforms"].get("laptop_cpu", {})
    required_profile = {
        "solver": "highs",
        "screening": "numpy",
        "solver_session": "persistent_incremental",
    }
    for key, expected in required_profile.items():
        if profile.get(key) != expected:
            raise ScopfError(
                f"Gap experiment changed laptop profile {key}: "
                f"expected {expected!r}, observed {profile.get(key)!r}"
            )
    return suite_id, label


def _registry_path(config: RunConfig, suite_id: str) -> Path:
    return guard_output_path(
        config.root / "results" / "experiments" / f"{suite_id}-run-registry.json"
    )


def run_one_shot_gap_experiment(
    config: RunConfig,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Launch one immutable gap-level experiment with no wall-clock timeout."""

    suite_id, gap_label = _experiment_identity(config)
    validate_platform(config, "laptop_cpu")
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
        / f"{config.benchmark_id}-laptop_cpu.json"
    )
    if checkpoint.exists():
        raise ScopfError(f"Gap-experiment checkpoint already exists: {checkpoint}")
    diagnostic_root = config.root / "results" / "diagnostics"
    stale_diagnostics = [
        diagnostic_root / f"{config.benchmark_id}-laptop_cpu-events.jsonl",
        diagnostic_root / f"{config.benchmark_id}-laptop_cpu-highs.log",
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
        "laptop_cpu",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=config.root,
        check=False,
        capture_output=True,
        text=True,
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
    result.update(
        {
            "official": False,
            "experiment": True,
            "one_shot": True,
            "platform": "laptop_cpu",
            "experiment_suite_id": suite_id,
            "gap_label": gap_label,
            "frozen_identity": identity,
            "total_wall_time_seconds": total_wall,
            "benchmark_boundary": (
                "worker launch through raw loading, all MIP solve/screen rounds, "
                "independent verification, fixed-commitment pricing, and result "
                "serialization; no wall-clock deadline"
            ),
        }
    )
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
