from pathlib import Path

from activsg_scopf.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_registered_config_is_single_hour_activsg500() -> None:
    config = load_config(ROOT / "configs" / "activsg500.json")
    assert config.raw["case_name"] == "ACTIVSg500"
    assert config.model["interval_hours"] == 1.0
    assert config.model["pwl_segments"] == 10
    assert config.runtime["deadline_seconds"] == 300.0


def test_registered_activsg10k_config_is_single_hour_and_separately_tagged() -> None:
    config = load_config(ROOT / "configs" / "activsg10k.json")
    assert config.case_name == "ACTIVSg10k"
    assert config.model["interval_hours"] == 1.0
    assert config.model["pwl_segments"] == 10
    assert config.benchmark_id == "activsg10k-v1"
    assert config.raw["benchmark"]["required_git_tag"] == "benchmark-10k-v1"
