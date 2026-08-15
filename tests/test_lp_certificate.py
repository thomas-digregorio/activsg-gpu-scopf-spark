from __future__ import annotations

from pathlib import Path

from activsg_scopf.config import load_config
from activsg_scopf.lp_certificate import validate_lp_certificate_config

ROOT = Path(__file__).resolve().parents[1]


def test_activsg2000_gpu_lp_certificate_is_registered_for_one_600_second_run() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lp-certificate-v12.json")

    registration = validate_lp_certificate_config(config)

    assert config.runtime["deadline_seconds"] == 600.0
    assert registration["profile"]["lp_method"] == "pdlp"
    assert registration["profile"]["pdlp_precision"] == "fp64"
    assert registration["reference_incumbent"]["objective"] == 1133047.8684341211
