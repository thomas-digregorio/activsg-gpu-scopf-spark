from __future__ import annotations

import json
from pathlib import Path

import pytest

from activsg_scopf.config import load_config
from activsg_scopf.errors import ScopfError
from activsg_scopf.lp_certificate import (
    allocate_lp_round_budget,
    validate_lp_certificate_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("version", ["v12", "v13", "v14", "v15"])
def test_activsg2000_gpu_lp_certificate_is_registered_for_one_600_second_run(
    version: str,
) -> None:
    config = load_config(ROOT / "configs" / f"activsg2000-gpu-lp-certificate-{version}.json")

    registration = validate_lp_certificate_config(config)

    assert config.runtime["deadline_seconds"] == (900.0 if version == "v15" else 600.0)
    assert registration["profile"]["lp_method"] == "pdlp"
    assert registration["profile"]["pdlp_precision"] == "fp64"
    assert registration["reference_incumbent"]["objective"] == 1133047.8684341211
    if version in {"v14", "v15"}:
        assert registration["profile"]["per_constraint_residual"] is True
        assert registration["profile"]["redundant_angle_bounds"] == ("rate_a_dc_shortest_path_v1")
    if version == "v15":
        assert registration["profile"]["presolve"] == 0
        assert registration["profile"]["recover_time_limit_vectors"] is True


def test_lp_round_budget_uses_cap_and_preserves_followup() -> None:
    assert allocate_lp_round_budget(
        825.0,
        maximum_round_seconds=480.0,
        followup_reserve_seconds=90.0,
        minimum_round_seconds=30.0,
    ) == 480.0
    assert allocate_lp_round_budget(
        255.0,
        maximum_round_seconds=480.0,
        followup_reserve_seconds=90.0,
        minimum_round_seconds=30.0,
    ) == 165.0


def test_lp_round_budget_spends_last_small_window() -> None:
    assert allocate_lp_round_budget(
        90.0,
        maximum_round_seconds=480.0,
        followup_reserve_seconds=90.0,
        minimum_round_seconds=30.0,
    ) == 90.0


def test_lp_certificate_rejects_unregistered_identity(tmp_path: Path) -> None:
    source = ROOT / "configs" / "activsg2000-gpu-lp-certificate-v13.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["benchmark"]["id"] = "activsg2000-gpu-lp-certificate-v99"
    raw["benchmark"]["experiment_suite_id"] = "activsg2000-gpu-lp-certificate-v99"
    candidate = tmp_path / "unregistered.json"
    candidate.write_text(json.dumps(raw), encoding="utf-8")

    config = load_config(candidate)

    with pytest.raises(ScopfError, match="Unregistered GPU LP-certificate benchmark id"):
        validate_lp_certificate_config(config)
