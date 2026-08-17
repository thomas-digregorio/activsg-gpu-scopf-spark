import numpy as np
import pytest

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.errors import ScopfError
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
    assert phase.column_upper[-1] == 1.0
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


def test_phase_one_registered_cap_fails_closed_if_box_bound_exceeds_it() -> None:
    source = CanonicalMILP()
    x = source.add_variable("x", lower=0.0, upper=1.0)
    source.add_row("far_away", {x: 1.0}, lower=100.0)

    with pytest.raises(ScopfError, match="exceeds the registered safety cap"):
        build_phase_one_model(
            source, base_mva=1.0, maximum_violation_pu=10.0
        )


def test_phase_one_certificate_replays_when_source_rows_are_reordered() -> None:
    first = CanonicalMILP()
    x_first = first.add_variable("x", lower=0.0, upper=1.0)
    first.add_row("force_high", {x_first: 1.0}, lower=2.0)
    first.add_row("named_upper", {x_first: 1.0}, upper=3.0)
    first_phase = build_phase_one_model(
        first, base_mva=100.0, maximum_violation_pu=1e6
    )
    certificate = phase_one_certificate(
        first_phase,
        np.asarray([-0.01, 0.0]),
        safety_margin_pu=1e-8,
        infeasibility_threshold_pu=1e-6,
    )

    reordered = CanonicalMILP()
    x_reordered = reordered.add_variable("x", lower=0.0, upper=1.0)
    reordered.add_row("named_upper", {x_reordered: 1.0}, upper=3.0)
    reordered.add_row("force_high", {x_reordered: 1.0}, lower=2.0)
    reordered_phase = build_phase_one_model(
        reordered, base_mva=100.0, maximum_violation_pu=1e6
    )
    replayed = replay_phase_one_certificate(reordered_phase, certificate)
    assert replayed["conservative_lower_bound_pu"] == pytest.approx(
        certificate["conservative_lower_bound_pu"]
    )

    legacy = dict(certificate)
    legacy.pop("semantic_row_dual_sha256")
    legacy.pop("row_identity_policy")
    legacy["canonical_row_duals"] = [
        {
            "row_name": record["row_name"],
            "canonical_row_dual": record["canonical_row_dual"],
        }
        for record in certificate["canonical_row_duals"]
    ]
    legacy_replayed = replay_phase_one_certificate(reordered_phase, legacy)
    assert legacy_replayed["conservative_lower_bound_pu"] == pytest.approx(
        certificate["conservative_lower_bound_pu"]
    )
