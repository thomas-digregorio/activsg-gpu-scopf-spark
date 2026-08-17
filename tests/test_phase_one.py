import numpy as np
import pytest

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.phase_one import (
    build_phase_one_model,
    phase_one_certificate,
    replay_phase_one_certificate,
)


def test_phase_one_box_dual_certifies_infeasible_bounded_model() -> None:
    source = CanonicalMILP()
    x = source.add_variable("x", lower=0.0, upper=1.0)
    source.add_row("impossible", {x: 1.0}, lower=2.0)
    phase = build_phase_one_model(
        source, base_mva=100.0, maximum_violation_pu=1e6
    )
    # The lower row is -x - 100 rho <= -2.  y=-0.01 gives dual bound 0.01.
    certificate = phase_one_certificate(
        phase,
        np.asarray([-0.01]),
        safety_margin_pu=1e-8,
        infeasibility_threshold_pu=1e-6,
    )

    assert certificate["prune_certified"] is True
    assert certificate["conservative_lower_bound_pu"] == pytest.approx(0.00999999)
    replayed = replay_phase_one_certificate(phase, certificate)
    assert replayed["prune_certified"] is True
    assert replayed["conservative_lower_bound_pu"] == pytest.approx(
        certificate["conservative_lower_bound_pu"]
    )


def test_phase_one_invalid_positive_upper_row_dual_is_projected_to_zero() -> None:
    source = CanonicalMILP()
    x = source.add_variable("x", lower=0.0, upper=1.0)
    source.add_row("feasible", {x: 1.0}, lower=0.5)
    phase = build_phase_one_model(
        source, base_mva=100.0, maximum_violation_pu=1e6
    )
    certificate = phase_one_certificate(
        phase,
        np.asarray([3.0]),
        safety_margin_pu=1e-8,
        infeasibility_threshold_pu=1e-6,
    )

    assert certificate["maximum_projected_dual_sign_violation"] == 3.0
    assert certificate["prune_certified"] is False
