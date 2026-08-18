from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from activsg_scopf import lagrangian_experiment as experiment_module
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.errors import ScopfError
from activsg_scopf.fixed_commitment import (
    FixedCommitmentProjectionInfeasible,
    build_fixed_commitment_projection,
)
from activsg_scopf.lagrangian import RegionMasks
from activsg_scopf.lagrangian_experiment import (
    ACTIVSG2000_V2_EXPERIMENT_ID,
    ACTIVSG2000_V3_EXPERIMENT_ID,
    PrimalCandidatePolicy,
    RegionAttemptRejected,
    _network_feasible_commitment_repairs,
    _prepare_region_master,
    _solve_fixed_commitment_feasibility,
    validate_lagrangian_experiment_config,
)
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import CouplingRow, fix_commitments
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
        [lifted[column] for column in master.index.segments_by_generator[0] if column is not None]
    )
    assert np.sum(segments) == 37.0
    assert projection.validate_lift(np.asarray([62.0]), tolerance_mw=1e-10)["passed"]


def test_registered_activsg2000_v2_feasibility_fix_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v2.json")
    registration = validate_lagrangian_experiment_config(config)

    assert config.benchmark_id == ACTIVSG2000_V2_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v2")
    fix = registration["benchmark"]["feasibility_fix"]
    assert fix["exact_source_pmin_pmax"] == ("retained_without_clipping_relaxation_or_replacement")
    assert fix["cpu_commitment_or_dispatch_seeded"] is False
    assert config.runtime["deadline_seconds"] == 1800.0


def test_constant_coupling_violation_rejects_only_fixed_candidate(
    monkeypatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v2.json")
    case, table = triangle_case()
    network = build_network(case)
    masks = RegionMasks(np.asarray([True]), np.asarray([False]))
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
    )
    with pytest.raises(FixedCommitmentProjectionInfeasible, match="balance"):
        build_fixed_commitment_projection(master, np.asarray([0]))

    monkeypatch.setattr(
        experiment_module,
        "solve_cuopt_continuous_pdlp",
        lambda *_args, **_kwargs: pytest.fail("constant-row candidate rejection must precede PDLP"),
    )
    catalog = build_contingency_catalog(case, network, table)
    with pytest.raises(RegionAttemptRejected) as rejected:
        _solve_fixed_commitment_feasibility(
            region_id="constant_row_candidate",
            commitment=np.asarray([0], dtype=np.int8),
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
    assert rejected.value.reason == "projected_constant_coupling_row_violation"
    precheck = rejected.value.rounds[0]["projection_precheck"]
    assert precheck["pruning_scope"] == "fixed_commitment_candidate_only"
    assert precheck["full_model_infeasibility_claimed"] is False


def test_network_aware_repair_turns_off_fixed_output_row_trigger() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v2.json")
    original, _ = triangle_case()
    gen = original.gen.copy()
    gen[1, 7] = 1.0
    gen[1, 8] = 10.0
    gen[1, 9] = 10.0
    case = replace(original, gen=gen)
    network = build_network(case)
    masks = RegionMasks.root(2)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=(),
    )
    row_index = master.canonical.add_row(
        "synthetic_constant_security_upper",
        {master.index.dispatch_by_generator[1]: 1.0},
        upper=9.0,
    )
    master.coupling_rows.append(
        CouplingRow(
            row_index=row_index,
            row_name="synthetic_constant_security_upper",
            rhs=9.0,
            generator_coefficients=np.asarray([0.0, 1.0]),
            bus_coefficients=np.zeros(3),
            kind="test_security",
        )
    )
    candidate = np.asarray([1, 1], dtype=np.int8)
    fix_commitments(master, candidate == 0, candidate == 1)

    repairs = _network_feasible_commitment_repairs(
        case=case,
        master=master,
        commitment=candidate,
        masks=masks,
        violated_row_name="synthetic_constant_security_upper",
        demand_mw=60.0,
        on_values=np.asarray([0.0, 100.0]),
        maximum_repairs=4,
        pair_search_limit=8,
        tolerance_mw=1e-4,
    )

    assert len(repairs) == 1
    np.testing.assert_array_equal(repairs[0].commitment, np.asarray([1, 0]))
    assert repairs[0].audit["turned_off_generator_source_rows"] == [2]
    assert repairs[0].audit["repaired_row_shortfall_mw"] == 0.0
    assert repairs[0].audit["capacity_shortfall_mw"] == 0.0
    assert repairs[0].audit["exact_source_pmin_pmax_retained"] is True


def test_registered_activsg2000_v3_candidate_rejection_fix_is_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v3.json")
    registration = validate_lagrangian_experiment_config(config)

    assert config.benchmark_id == ACTIVSG2000_V3_EXPERIMENT_ID
    assert registration["benchmark"]["required_git_tag"] == ("experiment-2000-gpu-lagrangian-v3")
    fix = registration["benchmark"]["candidate_rejection_fix"]
    assert fix["constant_coupling_row_result"] == (
        "reject_only_the_fixed_commitment_candidate_before_pdlp"
    )
    assert fix["full_model_infeasibility_claimed"] is False
    assert fix["continue_candidate_queue"] is True
    assert config.runtime["deadline_seconds"] == 1800.0

    config.raw["benchmark"]["candidate_rejection_fix"]["continue_candidate_queue"] = False
    with pytest.raises(ScopfError, match="candidate-rejection identity changed"):
        validate_lagrangian_experiment_config(config)


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

    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
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


def test_v4_projected_phase_one_maps_dual_into_exact_cost_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "activsg2000-gpu-lagrangian-v4.json")
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)

    def fake_solve(model, **kwargs):
        values = np.asarray([62.0, 0.0])
        column_scale, _ = native_scaling_vectors(
            model,
            mode=str(kwargs["native_scaling_mode"]),
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

    monkeypatch.setattr(experiment_module, "solve_cuopt_continuous_pdlp", fake_solve)
    result = _solve_fixed_commitment_feasibility(
        region_id="v4_fixture",
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

    assert result.source_native_row_dual is not None
    assert result.source_native_row_dual.shape == (result.master.canonical.num_rows,)
    mapping = result.rounds[0]["cost_lp_dual_warm_start"]
    assert mapping["eligible"] is True
    assert mapping["projection_to_source"]["mapped_native_constraint_count"] > 0
