import json
from pathlib import Path

from activsg_scopf.config import RunConfig
from activsg_scopf.errors import MipStartSolveError
from activsg_scopf.matpower import sha256_file
from activsg_scopf.runner import _solve_with_mip_start_fallback, run_end_to_end
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
