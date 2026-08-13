from pathlib import Path

import pytest

from activsg_scopf.config import RunConfig
from activsg_scopf.matpower import sha256_file
from activsg_scopf.model import build_master
from activsg_scopf.network import build_network
from activsg_scopf.solution import serialize_solution
from activsg_scopf.solvers import solve_canonical
from activsg_scopf.verify import verify_serialized_solution

from .helpers import triangle_case


def _write_raw_triangle(root: Path) -> tuple[Path, Path]:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    case_path = raw / "case_ACTIVSg500.m"
    case_path.write_text(
        """function mpc = case_ACTIVSg500
mpc.baseMVA = 100;
mpc.bus = [
1 3 0 0 0 0 1 1 0 230 1 1.1 0.9;
2 1 40 0 2 0 1 1 0 230 1 1.1 0.9;
3 1 20 0 0 0 1 1 0 230 1 1.1 0.9;
];
mpc.gen = [
1 60 0 100 -100 1 100 1 100 25;
3 0 0 100 -100 1 100 0 50 10;
];
mpc.branch = [
1 2 0 0.1 0 100 0 0 0 0 1 0 0;
2 3 0 0.1 0 100 0 0 0 0 1 0 0;
1 3 0 0.2 0 100 0 0 1.05 2 1 0 0;
];
mpc.gencost = [
2 0 0 3 0.01 10 100;
2 0 0 3 0.02 12 80;
];
""",
        encoding="utf-8",
    )
    contingency_path = raw / "contab_ACTIVSg500.m"
    contingency_path.write_text(
        """function chgtab = contab_ACTIVSg500
chgtab = [
1 0 CT_TBRCH 1 BR_STATUS CT_REP 0;
2 0 CT_TBRCH 2 BR_STATUS CT_REP 0;
3 0 CT_TBRCH 3 BR_STATUS CT_REP 0;
4 0 CT_TGEN 1 GEN_STATUS CT_REP 0;
];
""",
        encoding="utf-8",
    )
    return case_path, contingency_path


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
    case_path, contingency_path = _write_raw_triangle(tmp_path)
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
