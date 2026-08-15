from __future__ import annotations

import json
from pathlib import Path

import pytest

from activsg_scopf.config import load_config
from activsg_scopf.errors import ScopfError
from activsg_scopf.lp_certificate import validate_lp_certificate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("version", ["v12", "v13"])
def test_activsg2000_gpu_lp_certificate_is_registered_for_one_600_second_run(
    version: str,
) -> None:
    config = load_config(ROOT / "configs" / f"activsg2000-gpu-lp-certificate-{version}.json")

    registration = validate_lp_certificate_config(config)

    assert config.runtime["deadline_seconds"] == 600.0
    assert registration["profile"]["lp_method"] == "pdlp"
    assert registration["profile"]["pdlp_precision"] == "fp64"
    assert registration["reference_incumbent"]["objective"] == 1133047.8684341211


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
