import pytest

from activsg_scopf.costs import build_pwl_costs

from .helpers import triangle_case


def test_equal_width_pwl_retains_exact_pmin_and_polynomial() -> None:
    case, _ = triangle_case()
    curve = build_pwl_costs(case)[0]
    assert curve.pmin_mw == 25
    assert curve.pmax_mw == 100
    assert len(curve.segment_widths_mw) == 10
    assert all(width == 7.5 for width in curve.segment_widths_mw)
    assert curve.committed_base_cost == pytest.approx(356.25)
    assert curve.pwl_value(25) == curve.polynomial_value(25)
    assert curve.pwl_value(100) == curve.polynomial_value(100)
    assert curve.maximum_absolute_error > 0


def test_source_offline_generator_has_no_commitment_curve() -> None:
    case, _ = triangle_case()
    assert set(build_pwl_costs(case)) == {0}

