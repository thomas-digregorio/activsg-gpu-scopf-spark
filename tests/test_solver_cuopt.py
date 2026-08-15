import json

import numpy as np
import pytest

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.errors import ScopfError
from activsg_scopf.solvers.cuopt import (
    FULL_MIP_START,
    INTEGER_ONLY_MIP_START,
    NO_NATIVE_SCALING,
    POWER_SYSTEM_PER_UNIT_SCALING,
    audit_cuopt_native_log,
    audit_mip_start_readback,
    evaluate_mip_gap_certificate,
    native_scaling_audit,
    native_scaling_vectors,
    normalize_cuopt_pdlp_profile,
    prepare_mip_start,
)

PDLP_PROFILE = {
    "method": "pdlp",
    "solver_mode": "stable3",
    "precision": "fp64",
    "batch_strong_branching": True,
    "batch_reliability_branching": True,
    "reliability_branching_factor": 1,
}


def test_empty_cuopt_pdlp_profile_preserves_native_defaults() -> None:
    assert normalize_cuopt_pdlp_profile({}) == {}
    assert normalize_cuopt_pdlp_profile(None) == {}


def test_registered_cuopt_pdlp_profile_maps_to_exact_native_parameters() -> None:
    assert normalize_cuopt_pdlp_profile(PDLP_PROFILE) == {
        "method": 1,
        "pdlp_solver_mode": 4,
        "pdlp_precision": 1,
        "mip_batch_pdlp_strong_branching": 1,
        "mip_batch_pdlp_reliability_branching": 1,
        "mip_reliability_branching": 1,
    }


@pytest.mark.parametrize(
    "mutation, match",
    [
        ({"unknown": 1}, "exact registered key set"),
        ({"method": "concurrent"}, "Unsupported cuOpt PDLP method"),
        ({"solver_mode": "fast1"}, "Unsupported cuOpt PDLP solver mode"),
        ({"precision": "mixed"}, "Unsupported cuOpt PDLP precision"),
        ({"batch_strong_branching": 1}, "must be boolean"),
        ({"reliability_branching_factor": 2}, "requires.*factor 1"),
    ],
)
def test_cuopt_pdlp_profile_rejects_unregistered_values(
    mutation: dict[str, object], match: str
) -> None:
    profile = dict(PDLP_PROFILE)
    profile.update(mutation)
    with pytest.raises(ScopfError, match=match):
        normalize_cuopt_pdlp_profile(profile)


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


def test_cuopt_mip_start_requires_original_space_readback_and_presolve_off() -> None:
    audit = audit_mip_start_readback(
        columns=np.asarray([0, 2]),
        expected_native_values=np.asarray([1.0, 0.0]),
        native_initial_primal=np.asarray([1.0, np.nan, 0.0]),
        total_columns=3,
        presolve_readback=0,
    )

    assert audit["contract_passed"] is True
    assert audit["original_space_vector_readback"] is True
    assert audit["presolve_parameter_readback"] == 0


def test_cuopt_mip_start_rejects_presolve_or_changed_values() -> None:
    arguments = {
        "columns": np.asarray([0, 2]),
        "expected_native_values": np.asarray([1.0, 0.0]),
        "native_initial_primal": np.asarray([1.0, np.nan, 0.0]),
        "total_columns": 3,
    }
    with pytest.raises(ScopfError, match="presolve=0"):
        audit_mip_start_readback(**arguments, presolve_readback=1)
    with pytest.raises(ScopfError, match="value readback"):
        audit_mip_start_readback(
            **{
                **arguments,
                "native_initial_primal": np.asarray([0.0, np.nan, 0.0]),
            },
            presolve_readback=0,
        )


def test_cuopt_native_log_rejects_failed_start_and_records_fallback_warning() -> None:
    with pytest.raises(ScopfError, match="rejected the submitted MIP start"):
        audit_cuopt_native_log(
            "cuOpt version: 26.6.0\n"
            "Error cannot add the provided initial solution! Assignment size 4\n"
        )

    audit = audit_cuopt_native_log(
        "cuOpt version: 26.6.0\n"
        "Free variable found! Make sure the correct bounds are given.\n"
        "Barrier Solve status A numerical error was encountered.\n"
        "Solution objective: 1.0\n"
    )
    assert audit["mip_start_rejection_count"] == 0
    assert audit["free_variable_warning_count"] == 1
    assert audit["barrier_numerical_warning_count"] == 1


def _scaling_model() -> CanonicalMILP:
    model = CanonicalMILP()
    u = model.add_variable("u_g0", lower=0.0, upper=1.0, integer=True)
    pg = model.add_variable("pg_g0", objective=20.0, lower=0.0, upper=100.0)
    segment = model.add_variable(
        "pseg_g0_s0", objective=5.0, lower=0.0, upper=25.0
    )
    theta = model.add_variable("theta_b0", lower=-np.pi, upper=np.pi)
    flow = model.add_variable("flow_l0", lower=-200.0, upper=200.0)
    model.add_row(
        "generation_link_g0",
        {pg: 1.0, segment: -1.0, u: -10.0},
        lower=0.0,
        upper=0.0,
    )
    model.add_row(
        "nodal_balance_b0", {pg: 1.0, flow: -1.0}, lower=50.0, upper=50.0
    )
    model.add_row(
        "dc_flow_l0", {flow: 1.0, theta: -500.0}, lower=0.0, upper=0.0
    )
    model.add_row("angle_upper_l0", {theta: 1.0}, upper=0.5)
    model.add_row("c0001_m0002_upper", {flow: 1.25}, upper=175.0)
    return model


def test_no_native_scaling_is_identity() -> None:
    model = _scaling_model()

    column_scale, row_scale = native_scaling_vectors(
        model, mode=NO_NATIVE_SCALING, base_mva=100.0
    )

    np.testing.assert_array_equal(column_scale, np.ones(model.num_columns))
    np.testing.assert_array_equal(row_scale, np.ones(model.num_rows))


def test_power_system_native_scaling_is_exact_diagonal_reformulation() -> None:
    model = _scaling_model()
    column_scale, row_scale = native_scaling_vectors(
        model, mode=POWER_SYSTEM_PER_UNIT_SCALING, base_mva=100.0
    )

    np.testing.assert_array_equal(
        column_scale, np.asarray([1.0, 100.0, 100.0, 1.0, 100.0])
    )
    np.testing.assert_allclose(
        row_scale, np.asarray([0.01, 0.01, 0.002, 1.0, 0.01])
    )

    canonical_values = np.asarray([1.0, 50.0, 40.0, 0.1, 50.0])
    native_values = canonical_values / column_scale
    matrix = model.matrix_csr()
    native_matrix = matrix.multiply(column_scale).multiply(row_scale[:, None])
    objective = model.column_arrays()[0]
    np.testing.assert_allclose(
        native_matrix @ native_values,
        row_scale * (matrix @ canonical_values),
        rtol=0.0,
        atol=1e-14,
    )
    assert float((objective * column_scale) @ native_values) == float(
        objective @ canonical_values
    )


def test_power_system_native_scaling_rejects_nonpositive_base_mva() -> None:
    with pytest.raises(ScopfError, match="positive finite base MVA"):
        native_scaling_vectors(
            _scaling_model(),
            mode=POWER_SYSTEM_PER_UNIT_SCALING,
            base_mva=0.0,
        )


def test_power_system_native_scaling_audit_reports_equivalence() -> None:
    model = _scaling_model()
    audit = native_scaling_audit(
        model,
        np.asarray([1.0, 50.0, 40.0, 0.1, 50.0]),
        mode=POWER_SYSTEM_PER_UNIT_SCALING,
        base_mva=100.0,
    )

    assert audit["scaled_columns"] == 3
    assert audit["scaled_rows"] == 4
    assert audit["maximum_native_activity_identity_error"] < 1e-12
    assert audit["maximum_canonicalized_row_violation_identity_error"] < 1e-12
    assert audit["maximum_value_round_trip_error"] == 0.0
    assert audit["objective_identity_error"] == 0.0
    assert (
        audit["native_matrix_coefficients"]["ratio"]
        < audit["raw_matrix_coefficients"]["ratio"]
    )
