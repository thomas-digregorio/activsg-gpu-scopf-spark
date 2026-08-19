"""Versioned JSON configuration loading with semantic fail-closed checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cases import registered_case
from .errors import ProvenanceError, ScopeViolation
from .paths import assert_approved_activsg_name, guard_input_path

EXTENDED_DEADLINE_SUITES = {
    "activsg2000-gpu-gap-sensitivity-v7": 1935.0,
    "activsg2000-gpu-gap-sensitivity-v8": 1935.0,
    "activsg2000-gpu-gap-sensitivity-v9": 1935.0,
}


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
        value = self.raw["raw_inputs"].get("contingency_file")
        if value is None:
            raise ProvenanceError(
                f"{self.case_name} derives branch contingencies from its case topology"
            )
        return guard_input_path(self.root / value)

    @property
    def contingency_mode(self) -> str:
        return str(self.raw["raw_inputs"].get("contingency_mode", "source_table"))

    @property
    def source_archive_path(self) -> Path | None:
        value = self.raw["raw_inputs"].get("source_archive_file")
        return None if value is None else guard_input_path(self.root / value)

    @property
    def reference_case_path(self) -> Path:
        value = self.raw["raw_inputs"].get("reference_case_file")
        if value is None:
            raise ProvenanceError("Mapped contingencies require a reference case file")
        return guard_input_path(self.root / value)

    @property
    def reference_contingency_path(self) -> Path:
        value = self.raw["raw_inputs"].get("reference_contingency_file")
        if value is None:
            raise ProvenanceError(
                "Mapped contingencies require a reference contingency file"
            )
        return guard_input_path(self.root / value)

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
    if inputs.get("case_sha256") != registration.case_sha256:
        raise ProvenanceError("Configuration case hash is not the registered source hash")
    observed_contingency_mode = str(inputs.get("contingency_mode", "source_table"))
    if observed_contingency_mode not in registration.accepted_contingency_modes:
        raise ProvenanceError(
            "Configuration contingency mode is not registered for case_name"
        )
    if observed_contingency_mode == "source_table":
        if (
            Path(str(inputs.get("contingency_file", ""))).name
            != registration.contingency_file
        ):
            raise ProvenanceError(
                "Configuration contingency filename is not registered for case_name"
            )
        if inputs.get("contingency_sha256") != registration.contingency_sha256:
            raise ProvenanceError("Configuration contingency hash is not registered")
    elif observed_contingency_mode == "enumerate_in_service_branches":
        if "contingency_file" in inputs or "contingency_sha256" in inputs:
            raise ProvenanceError(
                "Topology-derived contingency configurations cannot name a contingency file"
            )
    elif observed_contingency_mode == "mapped_reference_branch_table":
        expected_reference_fields = {
            "reference_case_file": registration.reference_case_file,
            "reference_case_sha256": registration.reference_case_sha256,
            "reference_contingency_file": registration.reference_contingency_file,
            "reference_contingency_sha256": (
                registration.reference_contingency_sha256
            ),
            "branch_mapping_method": registration.branch_mapping_method,
        }
        for field, expected in expected_reference_fields.items():
            observed = inputs.get(field)
            if field.endswith("_file"):
                observed = Path(str(observed)).name
            if observed != expected:
                raise ProvenanceError(
                    f"Configuration {field} is not registered for {registration.case_name}"
                )
        if "contingency_file" in inputs or "contingency_sha256" in inputs:
            raise ProvenanceError(
                "Mapped contingency configurations must use reference contingency fields"
            )
    else:
        raise ProvenanceError(
            f"Unsupported registered contingency mode {observed_contingency_mode!r}"
        )
    if registration.source_archive_file is not None:
        if (
            Path(str(inputs.get("source_archive_file", ""))).name
            != registration.source_archive_file
        ):
            raise ProvenanceError("Configuration source archive filename is not registered")
        if inputs.get("source_archive_sha256") != registration.source_archive_sha256:
            raise ProvenanceError("Configuration source archive hash is not registered")
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
        benchmark = payload["benchmark"]
        maximum_deadline = EXTENDED_DEADLINE_SUITES.get(
            str(benchmark.get("experiment_suite_id", "")),
            (
                1800.0
                if benchmark.get("kind")
                in {
                    "gap_sensitivity_experiment",
                    "series24_scenario_experiment",
                    "seeded_round2_diagnostic",
                    "gpu_lp_relaxation_certificate",
                    "gpu_lagrangian_disjunctive_experiment",
                }
                else 300.0
            ),
        )
        if deadline <= 0 or deadline > maximum_deadline:
            raise ScopeViolation(
                f"The end-to-end deadline must be in (0, {maximum_deadline:g}] seconds"
            )
    benchmark = payload["benchmark"]
    if benchmark.get("kind") == "series24_scenario_experiment":
        if not str(payload["case_name"]).startswith("Texas2kSeries24Case"):
            raise ScopeViolation("Series24 scenario experiments require a Series24 case")
        if float(payload["model"].get("mip_relative_gap_tolerance", -1.0)) != 1e-3:
            raise ScopeViolation("Series24 scenario experiments require a 1e-3 MIP gap")
        if float(payload["runtime"].get("deadline_seconds", -1.0)) != 1800.0:
            raise ScopeViolation("Series24 scenario experiments require a 1,800-second cap")
        expected_initialization = {
            "cross_scenario_mip_start": False,
            "round_1": "cold",
            "later_rounds": "prior_round_commitment_within_same_scenario",
        }
        if benchmark.get("initialization") != expected_initialization:
            raise ScopeViolation(
                "Series24 scenarios must start cold and may reuse only same-scenario rounds"
            )
    root = config_path.parent.parent
    return RunConfig(path=config_path, root=root, raw=payload)
