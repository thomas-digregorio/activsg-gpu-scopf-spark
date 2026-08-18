import json
from pathlib import Path

import pytest

from activsg_scopf.errors import ScopfError
from activsg_scopf.official import validate_laptop_gate


def test_spark_gate_rejects_laptop_evidence_for_a_different_case(tmp_path: Path) -> None:
    evidence = tmp_path / "activsg10k-laptop.json"
    identity = {"commit": "abc", "tag": "benchmark-10k-v1", "config_sha256": "def"}
    evidence.write_text(
        json.dumps(
            {
                "official": True,
                "platform": "laptop_cpu",
                "status": "optimal_verified",
                "total_wall_time_seconds": 10,
                "case_name": "ACTIVSg500",
                "benchmark_id": "activsg10k-v1",
                "frozen_identity": identity,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ScopfError, match="different case or benchmark"):
        validate_laptop_gate(
            evidence,
            identity,
            case_name="ACTIVSg10k",
            benchmark_id="activsg10k-v1",
        )
