"""Versioned JSON configuration loading with semantic fail-closed checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cases import registered_case
from .errors import ProvenanceError, ScopeViolation
from .paths import assert_approved_activsg_name, guard_input_path


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

    @property
    def case_name(self) -> str:
        return str(self.raw["case_name"])

    @property
    def benchmark_id(self) -> str:
        return str(self.raw["benchmark"]["id"])

    @property
    def benchmark_kind(self) -> str:
        return str(self.raw["benchmark"].get("kind", "deadline_benchmark"))


def load_config(path: str | Path) -> RunConfig:
    config_path = guard_input_path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ProvenanceError(f"Cannot read JSON configuration {config_path}: {exc}") from exc
    if payload.get("schema_version") != "1.0.0":
        raise ProvenanceError("Only configuration schema_version 1.0.0 is accepted")
    registration = registered_case(str(payload.get("case_name", "")))
    assert_approved_activsg_name(json.dumps(payload))
    required = {"raw_inputs", "model", "runtime", "platforms", "benchmark"}
    missing = required - payload.keys()
    if missing:
        raise ProvenanceError(f"Configuration is missing keys: {sorted(missing)}")
    inputs = payload["raw_inputs"]
    if Path(str(inputs.get("case_file", ""))).name != registration.case_file:
        raise ProvenanceError("Configuration case filename is not registered for case_name")
    if Path(str(inputs.get("contingency_file", ""))).name != registration.contingency_file:
        raise ProvenanceError("Configuration contingency filename is not registered for case_name")
    if inputs.get("case_sha256") != registration.case_sha256:
        raise ProvenanceError("Configuration case hash is not the registered source hash")
    if inputs.get("contingency_sha256") != registration.contingency_sha256:
        raise ProvenanceError("Configuration contingency hash is not registered")
    if payload["model"].get("interval_hours") != 1.0:
        raise ScopeViolation("Version 1 is exactly one one-hour interval")
    if payload["model"].get("pwl_segments") != 10:
        raise ScopeViolation("Version 1 requires exactly 10 equal-MW PWL segments")
    deadline_value = payload["runtime"].get("deadline_seconds", 0)
    if deadline_value is None:
        if payload["benchmark"].get("kind") != "gap_sensitivity_experiment":
            raise ScopeViolation(
                "Only a registered gap-sensitivity experiment may omit the deadline"
            )
    else:
        deadline = float(deadline_value)
        maximum_deadline = (
            1800.0
            if payload["benchmark"].get("kind")
            in {"gap_sensitivity_experiment", "seeded_round2_diagnostic"}
            else 300.0
        )
        if deadline <= 0 or deadline > maximum_deadline:
            raise ScopeViolation(
                f"The end-to-end deadline must be in (0, {maximum_deadline:g}] seconds"
            )
    root = config_path.parent.parent
    return RunConfig(path=config_path, root=root, raw=payload)
