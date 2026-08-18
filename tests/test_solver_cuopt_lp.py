from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from activsg_scopf.errors import ScopfError
from activsg_scopf.network import NetworkData
from activsg_scopf.solvers.cuopt_lp import (
    audit_pdlp_warm_start_data,
    derive_rate_a_angle_bounds,
    prepare_pdlp_warm_start,
    solve_cuopt_continuous_pdlp,
    validate_numeric_lp_certificate,
)


def _complete_pdlp_state(columns: int, rows: int) -> SimpleNamespace:
    return SimpleNamespace(
        current_ATY=np.zeros(columns),
        current_dual_solution=np.zeros(rows),
        current_primal_solution=np.zeros(columns),
        initial_dual_average=np.zeros(rows),
        initial_primal_average=np.zeros(columns),
        initial_primal_weight=1.0,
        initial_step_size=1.0,
        iterations_since_last_restart=0,
        last_candidate_kkt_score=0.0,
        last_restart_duality_gap_dual_solution=np.zeros(rows),
        last_restart_duality_gap_primal_solution=np.zeros(columns),
        last_restart_kkt_score=0.0,
        sum_dual_solutions=np.zeros(rows),
        sum_primal_solutions=np.zeros(columns),
        sum_solution_weight=1.0,
        total_pdhg_iterations=1,
        total_pdlp_iterations=1,
    )


def test_concurrent_cuopt_context_requires_console_logging() -> None:
    with pytest.raises(ScopfError, match="require console logging"):
        solve_cuopt_continuous_pdlp(
            None,  # type: ignore[arg-type]
            time_limit_seconds=1.0,
            optimality_tolerance=1e-8,
            primal_feasibility_tolerance=1e-6,
            certificate_residual_tolerance=1e-7,
            native_scaling_mode="power_system_equilibrated_v2",
            native_base_mva=100.0,
            log_to_console=False,
            concurrent_solver_context=True,
        )


def _certificate(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "matrix": sparse.csr_matrix([[1.0]]),
        "objective": np.asarray([1.0]),
        "objective_offset": 0.0,
        "row_types": np.asarray(["G"]),
        "row_rhs": np.asarray([0.5]),
        "column_lower": np.asarray([0.0]),
        "column_upper": np.asarray([1.0]),
        "primal": np.asarray([0.5]),
        "row_dual": np.asarray([1.0]),
        "reduced_cost": np.asarray([0.0]),
        "reported_primal_objective": 0.5,
        "reported_dual_objective": 0.5,
        "native_primal_residual": 0.0,
        "native_dual_residual": 0.0,
        "native_gap": 0.0,
        "optimality_tolerance": 1e-8,
        "primal_feasibility_tolerance": 1e-6,
        "residual_tolerance": 1e-7,
    }
    arguments.update(overrides)
    return validate_numeric_lp_certificate(**arguments)  # type: ignore[arg-type]


def test_numeric_lp_certificate_reconstructs_valid_minimization_dual() -> None:
    certificate = _certificate()

    assert certificate["passed"] is True
    assert certificate["reconstructed_primal_objective"] == 0.5
    assert certificate["reconstructed_dual_objective"] == 0.5
    assert certificate["maximum_stationarity_residual"] == 0.0


def test_numeric_lp_certificate_requires_reported_reconstructed_dual_agreement() -> None:
    certificate = _certificate(reported_dual_objective=0.25)

    assert certificate["passed"] is False
    assert certificate["dual_feasible"] is True
    assert certificate["reported_dual_objective_consistent"] is False
    assert certificate["conservative_numerical_lower_bound"] > 0.49


def test_numeric_lp_certificate_rejects_wrong_lower_row_dual_sign() -> None:
    certificate = _certificate(
        row_dual=np.asarray([-1.0]),
        reduced_cost=np.asarray([2.0]),
        reported_dual_objective=-0.5,
    )

    assert certificate["passed"] is False
    assert certificate["maximum_row_dual_sign_violation"] == 1.0


def test_numeric_lp_certificate_uses_variable_lower_bound_term() -> None:
    certificate = _certificate(
        matrix=sparse.csr_matrix((0, 1)),
        row_types=np.asarray([], dtype="U1"),
        row_rhs=np.asarray([]),
        row_dual=np.asarray([]),
        column_lower=np.asarray([2.0]),
        column_upper=np.asarray([5.0]),
        primal=np.asarray([2.0]),
        reduced_cost=np.asarray([1.0]),
        reported_primal_objective=2.0,
        reported_dual_objective=2.0,
    )

    assert certificate["passed"] is True
    assert certificate["reconstructed_dual_objective"] == 2.0


def test_numeric_lp_certificate_rejects_reduced_cost_for_free_column() -> None:
    certificate = _certificate(
        matrix=sparse.csr_matrix((0, 1)),
        row_types=np.asarray([], dtype="U1"),
        row_rhs=np.asarray([]),
        row_dual=np.asarray([]),
        column_lower=np.asarray([-np.inf]),
        column_upper=np.asarray([np.inf]),
        primal=np.asarray([0.0]),
        reduced_cost=np.asarray([1.0]),
        reported_primal_objective=0.0,
        reported_dual_objective=0.0,
    )

    assert certificate["passed"] is False
    assert certificate["incompatible_bound_column_count"] == 1


def test_numeric_lp_certificate_derives_bound_reduced_cost_when_api_returns_zero() -> None:
    certificate = _certificate(
        matrix=sparse.csr_matrix([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]),
        objective=np.asarray([3.0, 2.0, 5.0]),
        row_types=np.asarray(["E", "E"]),
        row_rhs=np.asarray([4.0, 5.0]),
        column_lower=np.asarray([0.0, 0.0, 0.0]),
        column_upper=np.asarray([np.inf, np.inf, np.inf]),
        primal=np.asarray([1.0, 3.0, 0.0]),
        row_dual=np.asarray([1.0, 1.0]),
        reduced_cost=np.asarray([0.0, 0.0, 0.0]),
        reported_primal_objective=9.0,
        reported_dual_objective=9.0,
    )

    assert certificate["passed"] is True
    assert certificate["reconstructed_dual_objective"] == 9.0
    assert certificate["maximum_returned_reduced_cost_mismatch"] == 3.0
    assert certificate["returned_reduced_cost_used_for_certificate"] is False


def test_numeric_lp_certificate_accepts_negative_upper_row_multiplier() -> None:
    certificate = _certificate(
        objective=np.asarray([-1.0]),
        row_types=np.asarray(["L"]),
        row_rhs=np.asarray([1.0]),
        primal=np.asarray([1.0]),
        row_dual=np.asarray([-1.0]),
        reduced_cost=np.asarray([0.0]),
        reported_primal_objective=-1.0,
        reported_dual_objective=-1.0,
    )

    assert certificate["passed"] is True
    assert certificate["reconstructed_dual_objective"] == -1.0


def test_numeric_lp_certificate_tolerates_tiny_stationarity_error_on_free_column() -> None:
    certificate = _certificate(
        matrix=sparse.csr_matrix([[1.0]]),
        objective=np.asarray([0.0]),
        row_types=np.asarray(["E"]),
        row_rhs=np.asarray([0.0]),
        column_lower=np.asarray([-np.inf]),
        column_upper=np.asarray([np.inf]),
        primal=np.asarray([0.0]),
        row_dual=np.asarray([1e-9]),
        reduced_cost=np.asarray([0.0]),
        reported_primal_objective=0.0,
        reported_dual_objective=0.0,
        native_dual_residual=1e-9,
    )

    assert certificate["passed"] is True
    assert certificate["maximum_stationarity_residual"] == 1e-9


def test_numeric_lp_certificate_does_not_reconstruct_presolve_native_norm() -> None:
    certificate = _certificate(
        native_primal_residual=3e-7,
        native_dual_residual=3e-5,
    )

    assert certificate["passed"] is True
    assert certificate["native_metrics_used_as_telemetry_only"] is True


def test_pdlp_warm_start_zero_extends_new_constraint_rows() -> None:
    primal, dual, audit = prepare_pdlp_warm_start(
        initial_native_primal=np.asarray([0.25, 0.75]),
        initial_native_row_dual=np.asarray([2.0, -3.0]),
        num_columns=2,
        num_constraints=4,
        presolve=0,
    )

    np.testing.assert_array_equal(primal, np.asarray([0.25, 0.75]))
    np.testing.assert_array_equal(dual, np.asarray([2.0, -3.0, 0.0, 0.0]))
    assert audit["initial_dual_zero_extended_count"] == 2
    assert audit["presolve_disabled"] is True


def test_pdlp_warm_start_rejects_presolve_and_oversized_dual() -> None:
    with np.testing.assert_raises_regex(ScopfError, "presolve=0"):
        prepare_pdlp_warm_start(
            initial_native_primal=np.asarray([0.0]),
            initial_native_row_dual=None,
            num_columns=1,
            num_constraints=1,
            presolve=-1,
        )
    with np.testing.assert_raises_regex(ScopfError, "invalid shape"):
        prepare_pdlp_warm_start(
            initial_native_primal=None,
            initial_native_row_dual=np.asarray([1.0, 2.0]),
            num_columns=1,
            num_constraints=1,
            presolve=0,
        )


def test_pdlp_full_state_audit_accepts_only_complete_finite_payload() -> None:
    state = _complete_pdlp_state(columns=3, rows=2)

    audit = audit_pdlp_warm_start_data(
        state,
        num_columns=3,
        num_constraints=2,
    )

    assert audit["passed"] is True
    assert audit["reason"] == "complete"


def test_pdlp_full_state_audit_rejects_time_limit_none_field_for_raw_fallback() -> None:
    state = _complete_pdlp_state(columns=3, rows=2)
    state.last_restart_duality_gap_dual_solution = None

    audit = audit_pdlp_warm_start_data(
        state,
        num_columns=3,
        num_constraints=2,
    )

    assert audit["passed"] is False
    assert audit["reason"] == "incomplete_or_invalid_payload"
    assert audit["none_fields"] == ["last_restart_duality_gap_dual_solution"]


def test_pdlp_full_state_audit_rejects_wrong_shape_and_nonfinite_field() -> None:
    state = _complete_pdlp_state(columns=3, rows=2)
    state.current_primal_solution = np.zeros(2)
    state.initial_step_size = np.inf

    audit = audit_pdlp_warm_start_data(
        state,
        num_columns=3,
        num_constraints=2,
    )

    assert audit["passed"] is False
    assert audit["wrong_shape_fields"] == ["current_primal_solution"]
    assert audit["nonfinite_fields"] == ["initial_step_size"]


def test_rate_a_angle_bounds_follow_shortest_reference_paths() -> None:
    network = NetworkData(
        base_mva=100.0,
        bus_ids=np.asarray([1, 2, 3], dtype=np.int64),
        reference_bus_index=0,
        active_branch_source_rows=np.asarray([0, 1, 2], dtype=np.int64),
        from_bus_index=np.asarray([0, 1, 0], dtype=np.int64),
        to_bus_index=np.asarray([1, 2, 2], dtype=np.int64),
        incidence=sparse.csr_matrix((3, 3)),
        susceptance_pu=np.asarray([10.0, 5.0, 1.0]),
        phase_shift_rad=np.asarray([0.1, 0.0, 0.0]),
        rate_a_mw=np.asarray([100.0, 100.0, 1000.0]),
        angle_min_rad=np.zeros(3),
        angle_max_rad=np.zeros(3),
        bbus=sparse.csc_matrix((3, 3)),
    )

    bounds = derive_rate_a_angle_bounds(
        network,
        np.asarray([1, 2, 3], dtype=np.int64),
        total_columns=5,
    )

    np.testing.assert_allclose(bounds.lower[1:4], [0.0, -0.2, -0.4])
    np.testing.assert_allclose(bounds.upper[1:4], [0.0, 0.2, 0.4])
    assert np.isneginf(bounds.lower[0])
    assert np.isposinf(bounds.upper[4])
    assert bounds.audit["canonical_physical_feasible_set_changed"] is False
