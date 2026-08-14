import json
from pathlib import Path

import numpy as np

from activsg_scopf.config import RunConfig
from activsg_scopf.errors import MipStartSolveError
from activsg_scopf.matpower import sha256_file
from activsg_scopf.runner import _solve_with_mip_start_fallback, run_end_to_end
from activsg_scopf.screening import ScreenResult, SecurityPair
from activsg_scopf.solvers import SolveResult

from .helpers import write_triangle_matpower


def test_tiny_end_to_end_run_persists_diagnostics(tmp_path: Path) -> None:
    case_path, contingency_path = write_triangle_matpower(tmp_path)
    config = RunConfig(
        path=tmp_path / "configs" / "activsg500-diagnostics.json",
        root=tmp_path,
        raw={
            "case_name": "ACTIVSg500",
            "raw_inputs": {
                "case_file": str(case_path.relative_to(tmp_path)),
                "case_sha256": sha256_file(case_path),
                "contingency_file": str(contingency_path.relative_to(tmp_path)),
                "contingency_sha256": sha256_file(contingency_path),
            },
            "model": {
                "interval_hours": 1.0,
                "pwl_segments": 10,
                "mip_relative_gap_tolerance": 1e-6,
                "model_residual_tolerance_pu": 1e-6,
                "security_violation_tolerance_pu": 1e-5,
                "lodf_validation_tolerance_pu": 1e-9,
                "lodf_validation_columns": 3,
                "lodf_build_chunk_columns": 2,
                "screen_chunk_columns": 2,
            },
            "runtime": {
                "deadline_seconds": 10.0,
                "verification_reserve_seconds": 1.0,
                "serialization_reserve_seconds": 0.5,
                "maximum_constraint_generation_rounds": 5,
            },
            "platforms": {
                "laptop_cpu": {
                    "solver": "highs",
                    "screening": "numpy",
                    "solver_threads": 0,
                    "solver_session": "persistent_incremental",
                    "diagnostics": {
                        "enabled": True,
                        "mip_logging_interval_seconds": 0.0,
                    },
                }
            },
            "benchmark": {
                "id": "activsg500-diagnostics-fixture",
                "required_git_tag": "unused",
                "official": False,
            },
        },
    )
    result = run_end_to_end(config, platform_name="laptop_cpu")
    assert result["status"] == "optimal_verified"
    event_path = tmp_path / result["diagnostics"]["event_log"]
    native_path = tmp_path / result["diagnostics"]["native_solver_log"]
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    names = [record["event"] for record in events]
    assert "constraint_generation_round_started" in names
    assert "highs_mip_progress" in names
    assert "exhaustive_contingency_screen_finished" in names
    assert names.count("constraint_generation_round_started") == names.count(
        "exhaustive_contingency_screen_started"
    )
    assert names.count("constraint_generation_round_started") == names.count(
        "exhaustive_contingency_screen_finished"
    )
    assert result["acceptance_gates"]["requested_mip_gap_certified"] is True
    assert (
        result["acceptance_gates"][
            "final_exhaustive_screen_zero_above_tolerance"
        ]
        is True
    )
    assert names[-1] == "end_to_end_finished"
    assert "Running HiGHS" in native_path.read_text(encoding="utf-8")


def test_tiny_unbounded_gap_run_reaches_fixed_commitment_prices(tmp_path: Path) -> None:
    case_path, contingency_path = write_triangle_matpower(tmp_path)
    config = RunConfig(
        path=tmp_path / "configs" / "activsg500-gap-fixture.json",
        root=tmp_path,
        raw={
            "case_name": "ACTIVSg500",
            "raw_inputs": {
                "case_file": str(case_path.relative_to(tmp_path)),
                "case_sha256": sha256_file(case_path),
                "contingency_file": str(contingency_path.relative_to(tmp_path)),
                "contingency_sha256": sha256_file(contingency_path),
            },
            "model": {
                "interval_hours": 1.0,
                "pwl_segments": 10,
                "mip_relative_gap_tolerance": 1e-3,
                "model_residual_tolerance_pu": 1e-6,
                "security_violation_tolerance_pu": 1e-5,
                "lodf_validation_tolerance_pu": 1e-9,
                "lodf_validation_columns": 3,
                "lodf_build_chunk_columns": 2,
                "screen_chunk_columns": 2,
            },
            "runtime": {
                "deadline_seconds": None,
                "verification_reserve_seconds": 0.0,
                "serialization_reserve_seconds": 0.0,
                "maximum_constraint_generation_rounds": 5,
            },
            "platforms": {
                "laptop_cpu": {
                    "solver": "highs",
                    "screening": "numpy",
                    "solver_threads": 0,
                    "solver_session": "persistent_incremental",
                }
            },
            "benchmark": {
                "id": "activsg500-gap-fixture",
                "kind": "gap_sensitivity_experiment",
                "pricing": {
                    "enabled": True,
                    "maximum_constraint_generation_rounds": 5,
                },
            },
        },
    )
    result = run_end_to_end(config, platform_name="laptop_cpu")
    assert result["deadline_seconds"] is None
    assert result["status"] == "optimal_verified"
    assert result["pricing"]["status"] == "optimal_secure_fixed_commitment_lp"
    assert len(result["pricing"]["bus_prices"]) == 3
    assert result["constraint_generation_rounds"][0]["solver_budget_seconds"] is None


def test_each_security_row_resolve_is_followed_by_another_exhaustive_screen(
    tmp_path: Path, monkeypatch
) -> None:
    case_path, contingency_path = write_triangle_matpower(tmp_path)
    config = RunConfig(
        path=tmp_path / "configs" / "activsg500-rescreen.json",
        root=tmp_path,
        raw={
            "case_name": "ACTIVSg500",
            "raw_inputs": {
                "case_file": str(case_path.relative_to(tmp_path)),
                "case_sha256": sha256_file(case_path),
                "contingency_file": str(contingency_path.relative_to(tmp_path)),
                "contingency_sha256": sha256_file(contingency_path),
            },
            "model": {
                "interval_hours": 1.0,
                "pwl_segments": 10,
                "mip_relative_gap_tolerance": 1e-3,
                "model_residual_tolerance_pu": 1e-6,
                "security_violation_tolerance_pu": 1e-5,
                "lodf_validation_tolerance_pu": 1e-9,
                "lodf_validation_columns": 3,
                "lodf_build_chunk_columns": 2,
                "screen_chunk_columns": 2,
            },
            "runtime": {
                "deadline_seconds": 10.0,
                "verification_reserve_seconds": 1.0,
                "serialization_reserve_seconds": 0.5,
                "maximum_constraint_generation_rounds": 5,
            },
            "platforms": {
                "laptop_cpu": {
                    "solver": "highs",
                    "screening": "numpy",
                    "solver_threads": 0,
                    "solver_session": "persistent_incremental",
                }
            },
            "benchmark": {
                "id": "activsg500-rescreen-fixture",
                "official": False,
            },
        },
    )
    pair = SecurityPair(
        contingency_label=1,
        monitored_branch_source_row=1,
        side="upper",
        outage_column=0,
        monitored_active_index=0,
        outage_active_index=1,
        lodf_value=1.0,
    )

    class TwoSolveSession:
        mode = "persistent_incremental"

        def __init__(self, model) -> None:
            self.model = model
            self.solve_count = 0

        def solve(self, *, time_limit_seconds: float | None) -> SolveResult:
            self.solve_count += 1
            values = np.zeros(self.model.num_columns, dtype=np.float64)
            return SolveResult(
                solver="highs",
                solver_version="test",
                status="Optimal",
                optimal=True,
                requested_gap_certified=True,
                has_incumbent=True,
                objective=float(self.solve_count),
                bound=float(self.solve_count),
                mip_gap=0.0,
                solve_time_seconds=0.0,
                values=values,
                statistics={},
            )

    class TwoScreenScreener:
        def __init__(self, *args, **kwargs) -> None:
            self.screen_count = 0

        def screen(self, *args, already_added=None, **kwargs) -> ScreenResult:
            self.screen_count += 1
            if self.screen_count == 1:
                assert already_added == set()
                return ScreenResult((pair,), 0.1, pair.pair_id, 2)
            assert already_added == {pair.pair_id}
            return ScreenResult((), 0.0, None, 2)

    class PassedVerification:
        passed = True
        maximum_model_residual_pu = 0.0
        maximum_security_violation_pu = 0.0

        def as_dict(self) -> dict[str, object]:
            return {"passed": True}

    session_holder: list[TwoSolveSession] = []

    def build_session(model, **kwargs) -> TwoSolveSession:
        session = TwoSolveSession(model)
        session_holder.append(session)
        return session

    monkeypatch.setattr("activsg_scopf.runner.validate_platform", lambda *_: None)
    monkeypatch.setattr("activsg_scopf.runner.environment_manifest", lambda *_: {})
    monkeypatch.setattr("activsg_scopf.runner.create_solver_session", build_session)
    monkeypatch.setattr(
        "activsg_scopf.runner.ContingencyScreener", TwoScreenScreener
    )
    monkeypatch.setattr(
        "activsg_scopf.runner.verify_serialized_solution",
        lambda *_: PassedVerification(),
    )

    result = run_end_to_end(config, platform_name="laptop_cpu")

    assert result["status"] == "optimal_verified"
    assert len(session_holder) == 1
    assert session_holder[0].solve_count == 2
    assert result["constraint_generation_round_count"] == 2
    assert result["added_security_pairs"] == 1
    assert [
        record["screen"]["new_violated_pairs"]
        for record in result["constraint_generation_rounds"]
    ] == [1, 0]
    assert result["acceptance_gates"]["requested_mip_gap_certified"] is True
    assert (
        result["acceptance_gates"][
            "final_exhaustive_screen_zero_above_tolerance"
        ]
        is True
    )


def test_partial_mip_start_internal_error_rebuilds_same_master_cold() -> None:
    class FailedStartSession:
        mode = "persistent_incremental"

        def solve(self, *, time_limit_seconds: float | None) -> SolveResult:
            raise MipStartSolveError("synthetic partial-start failure")

    class ColdSession:
        mode = "persistent_incremental"

        def solve(self, *, time_limit_seconds: float | None) -> SolveResult:
            return SolveResult(
                solver="highs",
                solver_version="test",
                status="Optimal",
                optimal=True,
                requested_gap_certified=True,
                has_incumbent=True,
                objective=1.0,
                bound=1.0,
                mip_gap=0.0,
                solve_time_seconds=0.0,
                values=None,
                statistics={},
            )

    cold = ColdSession()
    events: list[str] = []
    session, result, fallback = _solve_with_mip_start_fallback(
        FailedStartSession(),
        time_limit_seconds=None,
        rebuild_session=lambda: cold,
        emit_diagnostic=lambda event, **fields: events.append(event),
    )
    assert session is cold
    assert result.optimal
    assert fallback is not None and fallback["used"] is True
    assert result.statistics["mip_start_cold_fallback"] == fallback
    assert events == [
        "mip_start_cold_fallback_started",
        "mip_start_cold_fallback_finished",
    ]
