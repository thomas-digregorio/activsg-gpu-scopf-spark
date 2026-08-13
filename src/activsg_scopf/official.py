"""Immutable benchmark identity, one-run registry, and hard subprocess controller."""

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
from .matpower import sha256_file
from .paths import guard_input_path, guard_output_path
from .provenance import write_json_atomic


def _git(config: RunConfig, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=config.root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ScopfError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def frozen_identity(config: RunConfig) -> dict[str, str]:
    root = Path(_git(config, "rev-parse", "--show-toplevel")).resolve()
    if root != config.root.resolve():
        raise ScopfError(f"Configuration root {config.root} is not Git root {root}")
    dirty = _git(config, "status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise ScopfError("Official benchmark requires a completely clean worktree")
    commit = _git(config, "rev-parse", "HEAD")
    tag = str(config.raw["benchmark"]["required_git_tag"])
    try:
        tag_commit = _git(config, "rev-list", "-n", "1", tag)
    except ScopfError as exc:
        raise ScopfError(f"Required frozen benchmark tag {tag!r} does not exist") from exc
    if tag_commit != commit:
        raise ScopfError(
            f"Official benchmark requires HEAD {commit} to equal {tag} ({tag_commit})"
        )
    return {
        "commit": commit,
        "tag": tag,
        "config_sha256": sha256_file(config.path),
    }


def _registry_path(config: RunConfig) -> Path:
    name = f"{config.benchmark_id}-official-run-registry.json"
    return guard_output_path(config.root / "results" / name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read JSON evidence {path}: {exc}") from exc


def validate_laptop_gate(
    path: Path, identity: dict[str, str], *, case_name: str, benchmark_id: str
) -> None:
    path = guard_input_path(path)
    result = _read_json(path)
    if result.get("official") is not True or result.get("platform") != "laptop_cpu":
        raise ScopfError("Spark gate requires the official laptop_cpu result")
    if result.get("status") != "optimal_verified":
        raise ScopfError("Spark gate is closed because the laptop result did not pass")
    if float(result.get("total_wall_time_seconds", float("inf"))) > 300:
        raise ScopfError("Spark gate is closed because the laptop exceeded 300 seconds")
    if result.get("case_name") != case_name or result.get("benchmark_id") != benchmark_id:
        raise ScopfError("Spark gate laptop evidence is for a different case or benchmark")
    observed = result.get("frozen_identity", {})
    for key in ("commit", "tag", "config_sha256"):
        if observed.get(key) != identity[key]:
            raise ScopfError(f"Spark gate laptop evidence has a different {key}")


def register_start(
    config: RunConfig, platform_name: str, identity: dict[str, str], output: Path
) -> None:
    path = _registry_path(config)
    registry = _read_json(path) if path.exists() else {"schema_version": "1.0.0", "runs": {}}
    if platform_name in registry["runs"]:
        previous = registry["runs"][platform_name]
        raise ScopfError(
            f"Official {platform_name} run is already registered as {previous['status']}; "
            "automatic retry/replacement is forbidden"
        )
    registry["runs"][platform_name] = {
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "output": str(output),
        "frozen_identity": identity,
    }
    write_json_atomic(registry, path)


def register_finish(config: RunConfig, platform_name: str, result: dict[str, Any]) -> None:
    path = _registry_path(config)
    registry = _read_json(path)
    record = registry["runs"][platform_name]
    record.update(
        {
            "status": str(result.get("status", "unknown")),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "total_wall_time_seconds": result.get("total_wall_time_seconds"),
            "objective": result.get("objective"),
            "bound": result.get("bound"),
            "mip_gap": result.get("mip_gap"),
        }
    )
    write_json_atomic(registry, path)


def run_controlled(
    config: RunConfig,
    *,
    platform_name: str,
    output_path: Path,
    official: bool,
    laptop_result: Path | None = None,
) -> dict[str, Any]:
    if config.runtime.get("deadline_seconds") is None:
        raise ScopfError(
            "Unbounded gap configurations must use the gap-experiment command"
        )
    validate_platform(config, platform_name)
    output = guard_output_path(output_path)
    if official and not output.is_relative_to((config.root / "results").resolve()):
        raise ScopfError("Official output must be inside the repository results directory")
    if official and output.exists():
        raise ScopfError(f"Official output already exists and will not be overwritten: {output}")
    identity = frozen_identity(config) if official else {
        "commit": _git(config, "rev-parse", "HEAD"),
        "tag": "unfrozen-nonofficial",
        "config_sha256": sha256_file(config.path),
    }
    if official and platform_name == "dgx_spark":
        if laptop_result is None:
            raise ScopfError("Spark official benchmark requires --laptop-result evidence")
        validate_laptop_gate(
            laptop_result,
            identity,
            case_name=config.case_name,
            benchmark_id=config.benchmark_id,
        )
    if official:
        register_start(config, platform_name, identity, output)
    checkpoint = guard_output_path(
        config.root
        / "results"
        / "checkpoints"
        / f"{config.benchmark_id}-{platform_name}.json"
    )
    deadline_seconds = float(config.runtime["deadline_seconds"])
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
        "--deadline-seconds",
        str(deadline_seconds),
    ]
    if official:
        command.append("--official")
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
        wall_time = time.perf_counter() - started
        if output.exists():
            result = _read_json(output)
        else:
            result = _read_json(checkpoint) if checkpoint.exists() else {}
            result.update(
                {
                    "status": "failed_worker_without_result",
                    "worker_returncode": completed.returncode,
                    "worker_stderr": completed.stderr[-4000:],
                }
            )
    except subprocess.TimeoutExpired as exc:
        wall_time = time.perf_counter() - started
        result = _read_json(checkpoint) if checkpoint.exists() else {}

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
            "official": official,
            "platform": platform_name,
            "frozen_identity": identity,
            "total_wall_time_seconds": wall_time,
            "benchmark_boundary": (
                "worker process launch through first complete result serialization; "
                "raw loading, model build, all solve/screen rounds, and verification included"
            ),
        }
    )
    if wall_time > deadline_seconds and result.get("status") == "optimal_verified":
        result["status"] = "failed_end_to_end_deadline"
    write_json_atomic(result, output)
    if official:
        register_finish(config, platform_name, result)
    return result
