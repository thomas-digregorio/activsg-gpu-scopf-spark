import json

import numpy as np
import pytest

from activsg_scopf.errors import ScopfError
from activsg_scopf.solvers.cuopt import (
    FULL_MIP_START,
    INTEGER_ONLY_MIP_START,
    evaluate_mip_gap_certificate,
    prepare_mip_start,
)


def _certificate(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "native_status": "FeasibleFound",
        "objective": 1_118_437.5525740345,
        "bound": 1_118_297.1059715485,
        "reported_gap": 0.0001255739331724089,
        "requested_gap": 1e-3,
        "native_residuals": {
            "max_constraint_violation": 1.892749423859641e-8,
            "max_int_violation": 1.6663004309691587e-11,
            "max_variable_bound_violation": 2.8268800633668434e-9,
        },
        "residual_tolerance": 1e-6,
    }
    values.update(overrides)
    return evaluate_mip_gap_certificate(**values)  # type: ignore[arg-type]


def test_feasible_found_with_finite_bound_certifies_requested_gap() -> None:
    certificate = _certificate()

    assert certificate["passed"] is True
    assert certificate["native_status_supports_certificate"] is True
    assert certificate["reported_gap_meets_request"] is True
    assert certificate["calculated_gap_meets_request"] is True
    assert certificate["native_residuals_meet_tolerance"] is True
    json.dumps(certificate)
    assert abs(
        float(certificate["calculated_mip_relative_gap"])
        - 0.0001255739331724089
    ) < 1e-14


def test_gap_certificate_fails_closed_without_requested_bound_gap() -> None:
    assert _certificate(bound=None)["passed"] is False
    assert _certificate(reported_gap=2e-3)["passed"] is False
    assert _certificate(bound=1_110_000.0, reported_gap=1e-4)["passed"] is False


def test_gap_certificate_fails_closed_on_native_residual() -> None:
    residuals = {
        "max_constraint_violation": 2e-6,
        "max_int_violation": 0.0,
        "max_variable_bound_violation": 0.0,
    }

    certificate = _certificate(native_residuals=residuals)

    assert certificate["native_residuals_available"] is True
    assert certificate["native_residuals_meet_tolerance"] is False
    assert certificate["passed"] is False


def test_gap_certificate_rejects_non_solution_status() -> None:
    certificate = _certificate(native_status="Infeasible")

    assert certificate["native_status_supports_certificate"] is False
    assert certificate["passed"] is False


def test_full_mip_start_selects_every_column_and_normalizes_integers() -> None:
    candidate = np.asarray([0.9999999999999, 12.5, -0.25])
    integrality = np.asarray([1, 0, 0], dtype=np.int32)

    columns, values = prepare_mip_start(
        candidate,
        expected_shape=(3,),
        integrality=integrality,
        mode=FULL_MIP_START,
    )

    np.testing.assert_array_equal(columns, np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(values, np.asarray([1.0, 12.5, -0.25]))


def test_integer_only_mip_start_preserves_existing_round_behavior() -> None:
    columns, values = prepare_mip_start(
        np.asarray([1.0, np.nan, 0.0]),
        expected_shape=(3,),
        integrality=np.asarray([1, 0, 1], dtype=np.int32),
        mode=INTEGER_ONLY_MIP_START,
    )

    np.testing.assert_array_equal(columns, np.asarray([0, 2]))
    np.testing.assert_array_equal(values, np.asarray([1.0, 0.0]))


def test_full_mip_start_rejects_nonfinite_continuous_value() -> None:
    with pytest.raises(ScopfError, match="nonfinite"):
        prepare_mip_start(
            np.asarray([1.0, np.nan]),
            expected_shape=(2,),
            integrality=np.asarray([1, 0], dtype=np.int32),
            mode=FULL_MIP_START,
        )


def test_full_mip_start_can_project_numerical_excess_to_exact_bounds() -> None:
    columns, values = prepare_mip_start(
        np.asarray([1.0, 10.00000006, -2.00000001]),
        expected_shape=(3,),
        integrality=np.asarray([1, 0, 0], dtype=np.int32),
        mode=FULL_MIP_START,
        lower_bounds=np.asarray([0.0, 0.0, -2.0]),
        upper_bounds=np.asarray([1.0, 10.0, 5.0]),
        clip_to_bounds=True,
    )

    np.testing.assert_array_equal(columns, np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(values, np.asarray([1.0, 10.0, -2.0]))
