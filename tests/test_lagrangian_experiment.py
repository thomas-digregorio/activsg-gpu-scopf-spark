from pathlib import Path

import numpy as np
import pytest

from activsg_scopf import lagrangian_experiment as experiment_module
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.errors import PrimalCandidateRejected, ScopfError
from activsg_scopf.lagrangian import RegionMasks
from activsg_scopf.lagrangian_experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_TAG,
    PrimalCandidatePolicy,
    RegionAttemptRejected,
    _load_cpu_comparison,
    _solve_region,
    validate_lagrangian_experiment_config,
)
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import build_reduced_master
from activsg_scopf.screening import (
    ContingencyScreener,
    ScreenResult,
    SecurityPair,
)
from activsg_scopf.solvers.cuopt_lp import ContinuousSolveResult

from .helpers import triangle_case

ROOT = Path(__file__).resolve().parents[1]


def test_registered_activsg500_lagrangian_config_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v1.json")
    registration = validate_lagrangian_experiment_config(config)
    assert config.benchmark_id == EXPERIMENT_ID
    assert config.raw["benchmark"]["required_git_tag"] == EXPERIMENT_TAG
    assert config.runtime["deadline_seconds"] == 600.0
    assert config.model["mip_relative_gap_tolerance"] == 1e-3
    assert registration["profile"]["integer_solver"] == "none"
    assert registration["profile"]["branch_and_bound"] is False
    assert registration["profile"]["console_logging"] is True
    comparison = _load_cpu_comparison(config, registration)
    assert comparison["status"] == "optimal_verified"
    assert comparison["objective"] == pytest.approx(79410.651432182)
    assert comparison["total_wall_time_seconds"] == pytest.approx(2.702380099988659)


def test_registered_v2_bugfix_config_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v2.json")
    registration = validate_lagrangian_experiment_config(config)
    assert config.benchmark_id == "activsg500-gpu-lagrangian-v2"
    assert registration["benchmark"]["required_git_tag"] == ("experiment-500-gpu-lagrangian-v2")
    assert config.model["reduced_coefficient_zero_tolerance"] == 1e-14
    assert registration["benchmark"]["bugfix_change"]["pdlp_primal_gate"] == (
        "never_screen_or_add_rows_from_primal_infeasible_vector"
    )


def test_registered_v3_controller_config_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v3.json")
    registration = validate_lagrangian_experiment_config(config)
    assert config.benchmark_id == "activsg500-gpu-lagrangian-v3"
    assert registration["benchmark"]["required_git_tag"] == ("experiment-500-gpu-lagrangian-v3")
    assert registration["benchmark"]["controller_change"]["candidate_budget"] == (
        "15_seconds_total_with_5_second_pdlp_slices"
    )
    policy = PrimalCandidatePolicy.from_config(config)
    assert policy.total_seconds == 15.0
    assert policy.maximum_round_seconds == 5.0
    assert policy.stagnation_window_rounds == 2
    assert policy.dual_divergence_multiple == 1e6
    config.raw["runtime"]["maximum_primal_candidate_seconds"] = 16.0
    with pytest.raises(ScopfError, match="candidate policy changed"):
        validate_lagrangian_experiment_config(config)


def test_registered_v4_phase_one_controller_config_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v4.json")
    registration = validate_lagrangian_experiment_config(config)
    assert config.benchmark_id == "activsg500-gpu-lagrangian-v4"
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-500-gpu-lagrangian-v4"
    )
    assert registration["benchmark"]["controller_change"]["infeasible_leaf_gate"] == (
        "replayable_gpu_phase_one_box_dual_certificate"
    )
    assert config.model["serialized_lodf_replay_tolerance"] == 1e-12
    candidate = PrimalCandidatePolicy.from_config(config)
    region = PrimalCandidatePolicy.from_config(
        config, scope="disjunctive_region"
    )
    assert candidate.cold_restart_attempts == 1
    assert region.total_seconds == 15.0
    assert region.cold_restart_attempts == 1
    config.raw["runtime"]["phase_one_time_limit_seconds"] = 16.0
    with pytest.raises(ScopfError, match="bounded-region policy changed"):
        validate_lagrangian_experiment_config(config)


def test_lagrangian_config_rejects_non_500_identity() -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v1.json")
    config.raw["benchmark"]["id"] = "activsg2000-gpu-lagrangian-v1"
    with pytest.raises(ScopfError, match="Unregistered"):
        validate_lagrangian_experiment_config(config)


def test_tiny_region_flow_reaches_exhaustive_screen_without_integer_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v1.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    template = build_reduced_master(case, network)

    def fake_solve(model, **_kwargs):
        values = np.zeros(model.num_columns)
        values[template.index.commitment_by_generator[0]] = 1.0
        values[template.index.dispatch_by_generator[0]] = 62.0
        remaining = 37.0
        for column, width in zip(
            template.index.segments_by_generator[0],
            template.costs[0].segment_widths_mw,
            strict=True,
        ):
            if column is not None:
                values[column] = min(remaining, width)
                remaining -= values[column]
        return ContinuousSolveResult(
            status="Optimal",
            optimal=True,
            primal_objective=float(np.asarray(model.objective) @ values),
            dual_objective=0.0,
            values=values,
            native_primal=values.copy(),
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {"passed": True, "primal_feasible": True},
            },
        )

    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    monkeypatch.setattr(
        experiment_module,
        "canonical_row_duals",
        lambda master, native_row_dual, **_kwargs: np.asarray(native_row_dual),
    )
    monkeypatch.setattr(
        experiment_module,
        "optimize_lagrangian_bound_cupy",
        lambda master, row_dual, region, **_kwargs: (
            np.asarray(row_dual),
            {
                "backend": "fixture",
                "best_raw_lower_bound": 0.0,
                "best_minimizing_commitment": np.asarray([0], dtype=np.int8),
            },
        ),
    )
    solved = _solve_region(
        region_id="r",
        masks=RegionMasks.root(1),
        case=case,
        network=network,
        catalog=catalog,
        config=config,
        deadline=Deadline(10.0, 0.0, 0.0),
        initial_pairs=(),
        screener=ContingencyScreener(network, catalog, backend="numpy"),
        checkpoint=lambda: None,
    )
    assert solved.solve.statistics["native_integer_columns"] == 0
    assert solved.final_screen["new_violated_pairs"] == 0
    assert solved.lagrangian.conservative_lower_bound == -0.01


def test_region_continues_without_screening_a_primal_infeasible_pdlp_iterate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v2.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    template = build_reduced_master(case, network)
    solve_calls: list[dict[str, object]] = []

    def feasible_values(model) -> np.ndarray:
        values = np.zeros(model.num_columns)
        values[template.index.commitment_by_generator[0]] = 1.0
        values[template.index.dispatch_by_generator[0]] = 62.0
        remaining = 37.0
        for column, width in zip(
            template.index.segments_by_generator[0],
            template.costs[0].segment_widths_mw,
            strict=True,
        ):
            if column is not None:
                values[column] = min(remaining, width)
                remaining -= values[column]
        return values

    def fake_solve(model, **kwargs):
        solve_calls.append(kwargs)
        first = len(solve_calls) == 1
        values = np.zeros(model.num_columns) if first else feasible_values(model)
        return ContinuousSolveResult(
            status="TimeLimit" if first else "Optimal",
            optimal=not first,
            primal_objective=float(np.asarray(model.objective) @ values),
            dual_objective=0.0,
            values=values,
            native_primal=values.copy(),
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {
                    "passed": not first,
                    "primal_feasible": not first,
                },
            },
        )

    class CountingScreener:
        def __init__(self) -> None:
            self.calls = 0
            self.delegate = ContingencyScreener(network, catalog, backend="numpy")

        def screen(self, *args, **kwargs):
            self.calls += 1
            return self.delegate.screen(*args, **kwargs)

    screener = CountingScreener()
    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    monkeypatch.setattr(
        experiment_module,
        "canonical_row_duals",
        lambda master, native_row_dual, **_kwargs: np.asarray(native_row_dual),
    )
    monkeypatch.setattr(
        experiment_module,
        "optimize_lagrangian_bound_cupy",
        lambda master, row_dual, region, **_kwargs: (
            np.asarray(row_dual),
            {
                "backend": "fixture",
                "best_raw_lower_bound": 0.0,
                "best_minimizing_commitment": np.asarray([0], dtype=np.int8),
            },
        ),
    )
    solved = _solve_region(
        region_id="r",
        masks=RegionMasks.root(1),
        case=case,
        network=network,
        catalog=catalog,
        config=config,
        deadline=Deadline(10.0, 0.0, 0.0),
        initial_pairs=(),
        screener=screener,  # type: ignore[arg-type]
        checkpoint=lambda: None,
    )
    assert len(solve_calls) == 2
    assert screener.calls == 1
    assert solved.rounds[0]["screen"]["reason"] == "pdlp_primal_infeasible"
    assert solve_calls[1]["initial_native_primal"] is not None
    assert solve_calls[1]["initial_native_row_dual"] is not None


def test_candidate_rejects_divergent_dual_without_screening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v3.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    solve_calls = 0

    def fake_solve(model, **_kwargs):
        nonlocal solve_calls
        solve_calls += 1
        values = np.zeros(model.num_columns)
        return ContinuousSolveResult(
            status="TimeLimit",
            optimal=False,
            primal_objective=1.0,
            dual_objective=1e9,
            values=values,
            native_primal=values.copy(),
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {"passed": False, "primal_feasible": False},
            },
        )

    class RejectIfScreened:
        def screen(self, *_args, **_kwargs):
            raise AssertionError("primal-infeasible candidate must not be screened")

    progress: list[dict[str, object]] = []
    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    with pytest.raises(PrimalCandidateRejected, match="dual_objective_divergence"):
        _solve_region(
            region_id="p1",
            masks=RegionMasks.root(1),
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(30.0, 0.0, 0.0),
            initial_pairs=(),
            screener=RejectIfScreened(),  # type: ignore[arg-type]
            checkpoint=lambda: None,
            progress=progress.append,
            candidate_policy=PrimalCandidatePolicy.from_config(config),
        )
    assert solve_calls == 1
    final_round = progress[-1]["constraint_generation_rounds"][-1]  # type: ignore[index]
    assert final_round["candidate_gate"]["reason"] == "dual_objective_divergence"


def test_candidate_rejects_two_round_residual_stagnation_and_uses_warm_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v3.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    solve_calls: list[dict[str, object]] = []

    def fake_solve(model, **kwargs):
        solve_calls.append(kwargs)
        values = np.zeros(model.num_columns)
        return ContinuousSolveResult(
            status="TimeLimit",
            optimal=False,
            primal_objective=1.0,
            dual_objective=0.0,
            values=values,
            native_primal=values.copy(),
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {"passed": False, "primal_feasible": False},
            },
        )

    class RejectIfScreened:
        def screen(self, *_args, **_kwargs):
            raise AssertionError("primal-infeasible candidate must not be screened")

    progress: list[dict[str, object]] = []
    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    with pytest.raises(PrimalCandidateRejected, match="primal_residual_stagnation"):
        _solve_region(
            region_id="p1",
            masks=RegionMasks.root(1),
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(30.0, 0.0, 0.0),
            initial_pairs=(),
            screener=RejectIfScreened(),  # type: ignore[arg-type]
            checkpoint=lambda: None,
            progress=progress.append,
            candidate_policy=PrimalCandidatePolicy.from_config(config),
        )
    assert len(solve_calls) == 2
    assert solve_calls[1]["initial_native_primal"] is not None
    assert solve_calls[1]["initial_native_row_dual"] is not None
    final_round = progress[-1]["constraint_generation_rounds"][-1]  # type: ignore[index]
    assert final_round["candidate_gate"]["reason"] == "primal_residual_stagnation"


def test_candidate_does_not_relabel_adapter_error_as_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v3.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)

    def fake_solve(_model, **_kwargs):
        return ContinuousSolveResult(
            status="Error",
            optimal=False,
            primal_objective=None,
            dual_objective=None,
            values=None,
            native_primal=None,
            native_row_dual=None,
            solve_time_seconds=0.001,
            statistics={
                "error_status": "InternalError",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {"passed": False, "primal_feasible": False},
            },
        )

    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    with pytest.raises(ScopfError, match="did not return usable vectors") as caught:
        _solve_region(
            region_id="p1",
            masks=RegionMasks.root(1),
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(30.0, 0.0, 0.0),
            initial_pairs=(),
            screener=ContingencyScreener(network, catalog, backend="numpy"),
            checkpoint=lambda: None,
            candidate_policy=PrimalCandidatePolicy.from_config(config),
        )
    assert type(caught.value) is ScopfError


def test_v4_region_uses_warm_attempt_then_exactly_one_cold_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg500-gpu-lagrangian-v4.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    template = build_reduced_master(case, network)
    solve_calls: list[dict[str, object]] = []

    def feasible_values(model) -> np.ndarray:
        values = np.zeros(model.num_columns)
        values[template.index.commitment_by_generator[0]] = 1.0
        values[template.index.dispatch_by_generator[0]] = 62.0
        remaining = 37.0
        for column, width in zip(
            template.index.segments_by_generator[0],
            template.costs[0].segment_widths_mw,
            strict=True,
        ):
            if column is not None:
                values[column] = min(remaining, width)
                remaining -= values[column]
        return values

    def fake_solve(model, **kwargs):
        solve_calls.append(kwargs)
        if len(solve_calls) == 1:
            values = feasible_values(model)
            primal_feasible = True
            dual_objective = 0.0
        else:
            values = np.zeros(model.num_columns)
            primal_feasible = False
            dual_objective = 1e9
        return ContinuousSolveResult(
            status="TimeLimit",
            optimal=False,
            primal_objective=1.0,
            dual_objective=dual_objective,
            values=values,
            native_primal=values.copy(),
            native_row_dual=np.zeros(model.num_rows),
            solve_time_seconds=0.001,
            statistics={
                "error_status": "Success",
                "solved_by": "PDLP",
                "solved_by_pdlp": True,
                "native_integer_columns": 0,
                "dual_certificate": {
                    "passed": primal_feasible,
                    "primal_feasible": primal_feasible,
                },
            },
        )

    outage = catalog.valid[0]
    pair = SecurityPair(
        outage.contingency_label,
        int(network.active_branch_source_rows[1]) + 1,
        "upper",
        0,
        1,
        outage.active_branch_index,
        float(catalog.lodf[1, 0]),
    )

    class OneViolation:
        calls = 0

        def screen(self, *_args, **_kwargs):
            self.calls += 1
            return ScreenResult((pair,), 0.1, pair.pair_id, 1)

    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    with pytest.raises(RegionAttemptRejected, match="cold_restart_failed"):
        _solve_region(
            region_id="r0",
            masks=RegionMasks.root(1),
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(30.0, 0.0, 0.0),
            initial_pairs=(),
            screener=OneViolation(),  # type: ignore[arg-type]
            checkpoint=lambda: None,
            candidate_policy=PrimalCandidatePolicy.from_config(
                config, scope="disjunctive_region"
            ),
        )
    assert len(solve_calls) == 3
    assert solve_calls[1]["initial_native_primal"] is not None
    assert solve_calls[1]["initial_native_row_dual"] is not None
    assert solve_calls[2]["initial_native_primal"] is None
    assert solve_calls[2]["initial_native_row_dual"] is None
