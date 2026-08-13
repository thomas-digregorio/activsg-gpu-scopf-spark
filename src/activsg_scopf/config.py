"""Versioned JSON configuration loading with semantic fail-closed checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ProvenanceError, ScopeViolation
from .matpower import EXPECTED_CASE_SHA256, EXPECTED_CONTINGENCY_SHA256
from .paths import assert_activsg500_name, guard_input_path


@dataclass(frozen=True)
class RunConfig:
    path: Path
    root: Path
    raw: dict[str, Any]

    @property
    def case_path(self) -> Path:
        return guard_input_path(self.root / self.raw["raw_inputs"]["case_file"])

    @property
    def contingency_path(self) -> Path:
        return guard_input_path(self.root / self.raw["raw_inputs"]["contingency_file"])

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.raw["runtime"]


def load_config(path: str | Path) -> RunConfig:
    config_path = guard_input_path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ProvenanceError(f"Cannot read JSON configuration {config_path}: {exc}") from exc
    if payload.get("schema_version") != "1.0.0":
        raise ProvenanceError("Only configuration schema_version 1.0.0 is accepted")
    if payload.get("case_name") != "ACTIVSg500":
        raise ScopeViolation("Only case_name ACTIVSg500 is approved")
    assert_activsg500_name(json.dumps(payload))
    required = {"raw_inputs", "model", "runtime", "platforms", "benchmark"}
    missing = required - payload.keys()
    if missing:
        raise ProvenanceError(f"Configuration is missing keys: {sorted(missing)}")
    inputs = payload["raw_inputs"]
    if inputs.get("case_sha256") != EXPECTED_CASE_SHA256:
        raise ProvenanceError("Configuration case hash is not the registered ACTIVSg500 hash")
    if inputs.get("contingency_sha256") != EXPECTED_CONTINGENCY_SHA256:
        raise ProvenanceError("Configuration contingency hash is not registered")
    if payload["model"].get("interval_hours") != 1.0:
        raise ScopeViolation("Version 1 is exactly one one-hour interval")
    if payload["model"].get("pwl_segments") != 10:
        raise ScopeViolation("Version 1 requires exactly 10 equal-MW PWL segments")
    deadline = float(payload["runtime"].get("deadline_seconds", 0))
    if deadline <= 0 or deadline > 300:
        raise ScopeViolation("The end-to-end deadline must be in (0, 300] seconds")
    root = config_path.parent.parent
    return RunConfig(path=config_path, root=root, raw=payload)

