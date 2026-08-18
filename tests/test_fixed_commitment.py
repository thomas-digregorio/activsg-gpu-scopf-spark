from pathlib import Path

import numpy as np

from activsg_scopf import lagrangian_experiment as experiment_module
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.fixed_commitment import build_fixed_commitment_projection
from activsg_scopf.lagrangian import RegionMasks
from activsg_scopf.lagrangian_experiment import (
    ACTIVSG2000_V2_EXPERIMENT_ID,
    PrimalCandidatePolicy,
    _prepare_region_master,
    _solve_fixed_commitment_feasibility,
    validate_lagrangian_experiment_config,
)
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.screening import ContingencyScreener
from activsg_scopf.solvers.cuopt import native_scaling_vectors
from activsg_scopf.solvers.cuopt_lp import ContinuousSolveResult

from .helpers import triangle_case

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_commitment_projection_lifts_exact_pmin_and_pwl_segments() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v2.json")
    case, _ = triangle_case()
    network = build_network(case)
    masks = RegionMasks(np.asarray([False]), np.asarray([True]))
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
    )

    projection = build_fixed_commitment_projection(master, np.asarray([1]))

    assert projection.canonical.variable_names == ["pg_g0001"]
    assert all(not name.startswith(("u_", "pseg_")) for name in projection.canonical.variable_names)
    assert projection.canonical.column_lower == [25.0]
    assert projection.canonical.column_upper == [100.0]
    assert projection.audit["exact_source_pmin_pmax_retained"] is True
    assert projection.audit["removed_commitment_column_count"] == 1
    assert projection.audit["removed_pwl_segment_column_count"] == 10

    lifted = projection.lift(np.asarray([62.0]))
    assert lifted[master.index.commitment_by_generator[0]] == 1.0
    assert lifted[master.index.dispatch_by_generator[0]] == 62.0
    segments = np.asarray(
        [
            lifted[column]
            for column in master.index.segments_by_generator[0]
            if column is not None
        ]
    )
    assert np.sum(segments) == 37.0
    assert projection.validate_lift(
        np.asarray([62.0]), tolerance_mw=1e-10
    )["passed"]


def test_registered_activsg2000_v2_feasibility_fix_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v2.json")
    registration = validate_lagrangian_experiment_config(config)

    assert config.benchmark_id == ACTIVSG2000_V2_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == (
        "experiment-2000-gpu-lagrangian-v2"
    )
    fix = registration["benchmark"]["feasibility_fix"]
    assert fix["exact_source_pmin_pmax"] == (
        "retained_without_clipping_relaxation_or_replacement"
    )
    assert fix["cpu_commitment_or_dispatch_seeded"] is False
    assert config.runtime["deadline_seconds"] == 1800.0


def test_projected_phase_one_returns_lifted_secure_fixture_dispatch(
    monkeypatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v2.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)

    def fake_solve(model, **_kwargs):
        assert model.variable_names == ["pg_g0001", "phase1_violation_pu"]
        values = np.asarray([62.0, 0.0])
        column_scale, _ = native_scaling_vectors(
            model,
            mode="power_system_per_unit_v1",
            base_mva=case.base_mva,
        )
        return ContinuousSolveResult(
            status="Optimal",
            optimal=True,
            primal_objective=0.0,
            dual_objective=0.0,
            values=values,
            native_primal=values / column_scale,
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

    monkeypatch.setattr(
        experiment_module, "solve_cuopt_continuous_pdlp", fake_solve
    )
    result = _solve_fixed_commitment_feasibility(
        region_id="fixture",
        commitment=np.asarray([1], dtype=np.int8),
        case=case,
        network=network,
        catalog=catalog,
        config=config,
        deadline=Deadline(10.0, 0.0, 0.0),
        initial_pairs=(),
        screener=ContingencyScreener(network, catalog, backend="numpy"),
        checkpoint=lambda: None,
        progress=None,
        policy=PrimalCandidatePolicy.from_config(config),
    )

    assert result.final_screen["maximum_violation_pu"] == 0.0
    assert result.final_screen["new_violated_pairs"] == 0
    assert result.rounds[0]["projection"]["projected_column_count"] == 1
    assert result.source_values[result.master.index.commitment_by_generator[0]] == 1.0
    assert result.source_values[result.master.index.dispatch_by_generator[0]] == 62.0
