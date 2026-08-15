from __future__ import annotations

import numpy as np
from scipy import sparse

from activsg_scopf.solvers.cuopt_lp import validate_numeric_lp_certificate


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
