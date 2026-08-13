from pathlib import Path

import pytest

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.config import RunConfig
from activsg_scopf.errors import ScopfError
from activsg_scopf.matpower import sha256_file
from activsg_scopf.model import build_master
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.pricing import run_fixed_commitment_pricing
from activsg_scopf.solution import serialize_solution
from activsg_scopf.solvers import create_solver_session, solve_canonical
from activsg_scopf.solvers.highs import _require_run_not_error
from activsg_scopf.verify import verify_serialized_solution

from .helpers import triangle_case, write_triangle_matpower


def test_highs_warning_is_preserved_for_model_status_extraction() -> None:
    import highspy

    _require_run_not_error(highspy.HighsStatus.kWarning)
    with pytest.raises(ScopfError, match="HiGHS"):
        _require_run_not_error(highspy.HighsStatus.kError)


def test_persistent_highs_session_appends_rows_logs_and_resolves(
    tmp_path: Path,
) -> None:
    model = CanonicalMILP()
    x = model.add_variable("x", objective=1.0, lower=0.0, upper=1.0, integer=True)
    events: list[tuple[str, dict[str, object]]] = []

    def record(event: str, **fields: object) -> None:
        events.append((event, fields))

    native_log = tmp_path / "highs.log"
    session = create_solver_session(
        model,
        solver="highs",
        mip_relative_gap=1e-6,
        threads=0,
        diagnostic_event=record,
        native_log_path=native_log,
        mip_logging_interval_seconds=0.0,
    )
    first = session.solve(time_limit_seconds=5.0)
    assert first.optimal
    assert first.values is not None
    assert first.values[x] == pytest.approx(0.0)
    model.add_row("force_x_on", {x: 1.0}, lower=1.0)
    second = session.solve(time_limit_seconds=5.0)
    assert second.optimal
    assert second.values is not None
    assert second.values[x] == pytest.approx(1.0)
    assert second.statistics["session_mode"] == "persistent_incremental"
    assert second.statistics["session_solve_number"] == 2
    assert second.statistics["incremental_rows_added"] == 1
    assert second.statistics["partial_integer_mip_start_return_status"] is not None
    assert 0.0 < second.statistics["native_time_limit_seconds"] <= 5.0
    _, configured_time_limit = session.highs.getOptionValue("time_limit")
    assert 0.0 < configured_time_limit <= 5.0
    event_names = [event for event, _ in events]
    assert event_names.count("highs_row_sync_started") == 2
    assert event_names.count("highs_run_started") == 2
    assert event_names.count("highs_run_finished") == 2
    assert "highs_mip_start_started" in event_names
    assert "highs_native_log" in event_names
    assert "highs_mip_progress" in event_names
    assert "Running HiGHS" in native_log.read_text(encoding="utf-8")


def test_highs_session_can_run_without_a_time_limit() -> None:
    model = CanonicalMILP()
    x = model.add_variable("x", objective=1.0, lower=0.0, upper=1.0, integer=True)
    model.add_row("force_x_on", {x: 1.0}, lower=1.0)
    session = create_solver_session(
        model,
        solver="highs",
        mip_relative_gap=1e-6,
    )
    result = session.solve(time_limit_seconds=None)
    assert result.optimal
    assert result.values is not None
    assert result.values[x] == pytest.approx(1.0)
    assert result.statistics["requested_call_budget_seconds"] is None
    assert result.statistics["native_time_limit_seconds"] is None


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
    generator = payload["solution"]["generators"][0]
    assert generator["pmin_mw"] == pytest.approx(25.0)
    assert generator["pmin_pu"] == pytest.approx(0.25)
    assert generator["dispatch_pu"] == pytest.approx(0.62)


def test_fixed_commitment_pricing_returns_mw_and_per_unit_prices() -> None:
    case, contingencies = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(
        case,
        network,
        contingencies,
        validation_columns=3,
        validation_tolerance_pu=1e-9,
    )
    master = build_master(case, network)
    result = solve_canonical(
        master.canonical,
        solver="highs",
        time_limit_seconds=5,
        mip_relative_gap=1e-6,
    )
    assert result.optimal and result.values is not None
    pricing = run_fixed_commitment_pricing(
        case,
        network,
        catalog,
        master,
        result.values,
        already_added_pair_ids=set(),
        security_tolerance_pu=1e-5,
        screen_chunk_columns=2,
        solver_threads=0,
        maximum_rounds=5,
    )
    assert pricing["status"] == "optimal_secure_fixed_commitment_lp"
    assert len(pricing["bus_prices"]) == 3
    assert len(pricing["generators"]) == 2
    for record in pricing["bus_prices"]:
        assert record["price_per_pu_hour"] == pytest.approx(
            record["price_per_mwh"] * case.base_mva
        )
    online = pricing["generators"][0]
    assert online["pricing_dispatch_pu"] == pytest.approx(
        online["pricing_dispatch_mw"] / case.base_mva
    )
