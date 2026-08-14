from activsg_scopf.solvers.cuopt import evaluate_mip_gap_certificate


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
