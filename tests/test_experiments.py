import json
import subprocess
from pathlib import Path

import pytest

from activsg_scopf.config import RunConfig, load_config
from activsg_scopf.errors import ScopfError
from activsg_scopf.experiments import (
    _experiment_identity,
    _require_prior_success,
    run_one_shot_gap_experiment,
)

ROOT = Path(__file__).resolve().parents[1]


def test_activsg500_gap_identity_is_bounded_and_registered() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gap-1e-3.json")
    assert _experiment_identity(config) == ("activsg500-gap-sensitivity-v1", "1e-3")


def test_activsg2000_gap_identity_is_bounded_and_registered() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gap-1e-3.json")
    assert _experiment_identity(config) == (
        "activsg2000-gap-sensitivity-v1",
        "1e-3",
    )


def test_next_gap_requires_prior_success_and_pricing() -> None:
    successful = {
        "runs": {
            "1e-3": {
                "status": "optimal_verified",
                "pricing_status": "optimal_secure_fixed_commitment_lp",
            }
        }
    }
    _require_prior_success(successful, "1e-4")

    with pytest.raises(ScopfError, match="was not run"):
        _require_prior_success({"runs": {}}, "1e-4")
    with pytest.raises(ScopfError, match="hard_deadline_exceeded"):
        _require_prior_success(
            {
                "runs": {
                    "1e-3": {
                        "status": "hard_deadline_exceeded",
                        "pricing_status": None,
                    }
                }
            },
            "1e-4",
        )
    with pytest.raises(ScopfError, match="does not have accepted"):
        _require_prior_success(
            {"runs": {"1e-3": {"status": "optimal_verified"}}},
            "1e-4",
        )


def test_bounded_gap_controller_serializes_hard_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = load_config(ROOT / "configs" / "activsg500-gap-1e-3.json")
    config_path = tmp_path / "configs" / "activsg500-gap-1e-3.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(source.raw), encoding="utf-8")
    config = RunConfig(path=config_path, root=tmp_path, raw=source.raw)
    output = tmp_path / "results" / "experiments" / "result.json"
    output.parent.mkdir(parents=True)

    monkeypatch.setattr("activsg_scopf.experiments.validate_platform", lambda *_: None)
    monkeypatch.setattr(
        "activsg_scopf.experiments.frozen_identity",
        lambda *_: {"commit": "frozen", "tag": "experiment-500-gap-v1", "config_sha256": "x"},
    )

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="worker", timeout=1800.0, output="partial")

    monkeypatch.setattr("activsg_scopf.experiments.subprocess.run", timeout)
    result = run_one_shot_gap_experiment(config, output_path=output)
    serialized = json.loads(output.read_text(encoding="utf-8"))
    registry = json.loads(
        (
            tmp_path
            / "results"
            / "experiments"
            / "activsg500-gap-sensitivity-v1-run-registry.json"
        ).read_text(encoding="utf-8")
    )

    assert result["status"] == "hard_deadline_exceeded"
    assert serialized["status"] == "hard_deadline_exceeded"
    assert serialized["worker_timeout_seconds"] == 1800.0
    assert serialized["worker_stdout"] == "partial"
    assert registry["runs"]["1e-3"]["status"] == "hard_deadline_exceeded"
