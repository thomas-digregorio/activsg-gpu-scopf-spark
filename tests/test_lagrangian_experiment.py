from pathlib import Path

import numpy as np
import pytest

from activsg_scopf import lagrangian_experiment as experiment_module
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.errors import ScopfError
from activsg_scopf.lagrangian import RegionMasks
from activsg_scopf.lagrangian_experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_TAG,
    _load_cpu_comparison,
    _solve_region,
    validate_lagrangian_experiment_config,
)
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import build_reduced_master
from activsg_scopf.screening import ContingencyScreener
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
