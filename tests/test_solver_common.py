import numpy as np

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.solvers.common import (
    HIGHS_FIXED_COMMITMENT_PRECHECK,
    RebuildingSolverSession,
    SolveResult,
)


def test_rebuilding_session_reuses_only_prior_solution_as_partial_start(
    monkeypatch,
) -> None:
    model = CanonicalMILP()
    model.add_variable("u", lower=0.0, upper=1.0, integer=True)
    model.add_variable("pg", lower=0.0, upper=100.0)
    returned_values = [
        np.asarray([1.0, 50.0], dtype=np.float64),
        np.asarray([1.0, 55.0], dtype=np.float64),
    ]
    observed_starts: list[np.ndarray | None] = []
    observed_options: list[dict[str, object]] = []

    def solve_stub(*args, mip_start_values=None, **kwargs):
        observed_starts.append(
            None if mip_start_values is None else mip_start_values.copy()
        )
        observed_options.append(kwargs)
        values = returned_values[len(observed_starts) - 1]
        return SolveResult(
            solver="cuopt",
            solver_version="test",
            status="Optimal",
            optimal=True,
            requested_gap_certified=True,
            has_incumbent=True,
            objective=float(values[1]),
            bound=float(values[1]),
            mip_gap=0.0,
            solve_time_seconds=0.01,
            values=values,
            statistics={},
        )

    monkeypatch.setattr(
        "activsg_scopf.solvers.common.solve_canonical",
        solve_stub,
    )
    session = RebuildingSolverSession(
        model=model,
        solver="cuopt",
        mip_relative_gap=1e-3,
        threads=0,
        native_scaling_mode="power_system_per_unit_v1",
        native_base_mva=100.0,
        log_to_console=True,
        cuopt_pdlp_profile={"method": "pdlp"},
    )

    first = session.solve(time_limit_seconds=5.0)
    first.values[0] = 0.0
    second = session.solve(time_limit_seconds=5.0)

    assert session.mode == "rebuild_each_round_with_partial_mip_start"
    assert observed_starts[0] is None
    assert observed_starts[1] is not None
    np.testing.assert_array_equal(observed_starts[1], np.asarray([1.0, 50.0]))
    np.testing.assert_array_equal(second.values, np.asarray([1.0, 55.0]))
    assert observed_options[0]["native_scaling_mode"] == (
        "power_system_per_unit_v1"
    )
    assert observed_options[0]["native_base_mva"] == 100.0
    assert observed_options[0]["log_to_console"] is True
    assert observed_options[0]["cuopt_pdlp_profile"] == {"method": "pdlp"}


def test_rebuilding_session_submits_only_feasibility_completed_full_start(
    monkeypatch,
) -> None:
    model = CanonicalMILP()
    model.add_variable("u", lower=0.0, upper=1.0, integer=True)
    dispatch = model.add_variable("pg", lower=0.0, upper=100.0)
    model.add_row("balance", {dispatch: 1.0}, lower=50.0, upper=50.0)
    starts: list[np.ndarray | None] = []
    options: list[dict[str, object]] = []

    def solve_stub(*args, mip_start_values=None, **kwargs):
        starts.append(
            None if mip_start_values is None else mip_start_values.copy()
        )
        options.append(kwargs)
        values = np.asarray([1.0, 50.0])
        return SolveResult(
            solver="cuopt",
            solver_version="test",
            status="Optimal",
            optimal=True,
            requested_gap_certified=True,
            has_incumbent=True,
            objective=50.0,
            bound=50.0,
            mip_gap=0.0,
            solve_time_seconds=0.01,
            values=values,
            statistics={},
        )

    def completion_stub(*args, **kwargs):
        values = np.asarray([1.0, 50.0])
        return SolveResult(
            solver="highs",
            solver_version="test",
            status="Optimal",
            optimal=True,
            requested_gap_certified=True,
            has_incumbent=True,
            objective=50.0,
            bound=50.0,
            mip_gap=0.0,
            solve_time_seconds=0.01,
            values=values,
            statistics={},
        )

    monkeypatch.setattr(
        "activsg_scopf.solvers.common.solve_canonical", solve_stub
    )
    monkeypatch.setattr(
        "activsg_scopf.solvers.highs.complete_fixed_integer_start",
        completion_stub,
    )
    session = RebuildingSolverSession(
        model=model,
        solver="cuopt",
        mip_relative_gap=1e-3,
        threads=0,
        mip_start_precheck=HIGHS_FIXED_COMMITMENT_PRECHECK,
    )

    session.solve(time_limit_seconds=5.0)
    second = session.solve(time_limit_seconds=5.0)

    assert session.mode == (
        "rebuild_each_round_with_feasibility_checked_mip_start"
    )
    assert starts[0] is None
    np.testing.assert_array_equal(starts[1], np.asarray([1.0, 50.0]))
    assert options[1]["mip_start_mode"] == "all_columns"
    assert options[1]["clip_mip_start_to_bounds"] is True
    precheck = second.statistics["mip_start_feasibility_precheck"]
    assert precheck["prior_commitment_extendable"] is True
    assert precheck["decision"] == "submit_completed_full_start"


def test_rebuilding_session_skips_unextendable_prior_commitment(
    monkeypatch,
) -> None:
    model = CanonicalMILP()
    model.add_variable("u", lower=0.0, upper=1.0, integer=True)
    starts: list[np.ndarray | None] = []

    def solve_stub(*args, mip_start_values=None, **kwargs):
        starts.append(
            None if mip_start_values is None else mip_start_values.copy()
        )
        values = np.asarray([1.0])
        return SolveResult(
            solver="cuopt",
            solver_version="test",
            status="Optimal",
            optimal=True,
            requested_gap_certified=True,
            has_incumbent=True,
            objective=0.0,
            bound=0.0,
            mip_gap=0.0,
            solve_time_seconds=0.01,
            values=values,
            statistics={},
        )

    def infeasible_completion(*args, **kwargs):
        return SolveResult(
            solver="highs",
            solver_version="test",
            status="Infeasible",
            optimal=False,
            requested_gap_certified=False,
            has_incumbent=False,
            objective=None,
            bound=None,
            mip_gap=None,
            solve_time_seconds=0.01,
            values=None,
            statistics={},
        )

    monkeypatch.setattr(
        "activsg_scopf.solvers.common.solve_canonical", solve_stub
    )
    monkeypatch.setattr(
        "activsg_scopf.solvers.highs.complete_fixed_integer_start",
        infeasible_completion,
    )
    session = RebuildingSolverSession(
        model=model,
        solver="cuopt",
        mip_relative_gap=1e-3,
        threads=0,
        mip_start_precheck=HIGHS_FIXED_COMMITMENT_PRECHECK,
    )

    session.solve(time_limit_seconds=5.0)
    second = session.solve(time_limit_seconds=5.0)

    assert starts == [None, None]
    precheck = second.statistics["mip_start_feasibility_precheck"]
    assert precheck["prior_commitment_extendable"] is False
    assert precheck["decision"] == "solve_cold"
