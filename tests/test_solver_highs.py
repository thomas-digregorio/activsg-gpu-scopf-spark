from pathlib import Path

import pytest

from activsg_scopf.config import RunConfig
from activsg_scopf.errors import ScopfError
from activsg_scopf.matpower import sha256_file
from activsg_scopf.model import build_master
from activsg_scopf.network import build_network
from activsg_scopf.solution import serialize_solution
from activsg_scopf.solvers import solve_canonical
from activsg_scopf.solvers.highs import _require_run_not_error
from activsg_scopf.verify import verify_serialized_solution

from .helpers import triangle_case, write_triangle_matpower


def test_highs_warning_is_preserved_for_model_status_extraction() -> None:
    import highspy

    _require_run_not_error(highspy.HighsStatus.kWarning)
    with pytest.raises(ScopfError, match="HiGHS"):
        _require_run_not_error(highspy.HighsStatus.kError)


def test_highs_adapter_and_independent_checker_use_one_tiny_solve(tmp_path: Path) -> None:
    case, _ = triangle_case()
    network = build_network(case)
    master = build_master(case, network)
    result = solve_canonical(
        master.canonical,
        solver="highs",
        time_limit_seconds=5,
        mip_relative_gap=1e-6,
    )
    assert result.optimal
    assert result.values is not None
    u = result.values[master.index.commitment_by_generator[0]]
    pg = result.values[master.index.dispatch_by_generator[0]]
    assert u == pytest.approx(1.0)
    assert pg == pytest.approx(62.0)
    case_path, contingency_path = write_triangle_matpower(tmp_path)
    config = RunConfig(
        path=tmp_path / "configs" / "activsg500.json",
        root=tmp_path,
        raw={
            "raw_inputs": {
                "case_file": str(case_path.relative_to(tmp_path)),
                "case_sha256": sha256_file(case_path),
                "contingency_file": str(contingency_path.relative_to(tmp_path)),
                "contingency_sha256": sha256_file(contingency_path),
            },
            "model": {
                "pwl_segments": 10,
                "model_residual_tolerance_pu": 1e-6,
                "security_violation_tolerance_pu": 1e-5,
                "lodf_validation_columns": 3,
                "lodf_validation_tolerance_pu": 1e-9,
            },
        },
    )
    payload = {
        "objective": result.objective,
        "solution": serialize_solution(result.values, case, network, master),
    }
    verification = verify_serialized_solution(config, payload)
    assert verification.passed
    assert verification.checked_valid_outages == 3
