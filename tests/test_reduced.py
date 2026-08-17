import numpy as np
import pytest

from activsg_scopf.lagrangian import (
    RegionMasks,
    bus_prices_from_coupling_duals,
    choose_split_generator,
    evaluate_lagrangian_bound,
    replay_lagrangian_certificate,
    verify_disjunctive_cover,
)
from activsg_scopf.network import build_contingency_catalog, build_network, solve_dc
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    commitment_vector,
    reduced_dispatch,
)
from activsg_scopf.screening import ContingencyScreener

from .helpers import triangle_case


def test_injection_elimination_matches_explicit_dc_solve() -> None:
    case, _ = triangle_case()
    network = build_network(case)
    master = build_reduced_master(case, network)
    dispatch = np.asarray([62.0])
    injection = np.asarray([62.0, -42.0, -20.0])
    expected_theta, expected_flow = solve_dc(network, injection)

    np.testing.assert_allclose(master.operator.angles(dispatch), expected_theta, atol=1e-13)
    np.testing.assert_allclose(master.operator.flows(dispatch), expected_flow, atol=1e-12)
    assert master.operator.total_demand_mw == 62.0


def test_reduced_security_rows_reproduce_screened_post_flow() -> None:
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    master = build_reduced_master(case, network)
    dispatch = np.asarray([62.0])
    flow = master.operator.flows(dispatch)
    screened = ContingencyScreener(network, catalog, backend="numpy").screen(
        flow, tolerance_pu=0.0
    )
    add_reduced_security_pairs(master, network, screened.violations)

    for pair in screened.violations:
        coupling = next(row for row in master.coupling_rows if row.row_name == pair.pair_id)
        activity = float(coupling.generator_coefficients @ dispatch)
        monitored = pair.monitored_active_index
        outaged = pair.outage_active_index
        post = flow[monitored] + pair.lodf_value * flow[outaged]
        expected_excess = (
            post - network.rate_a_mw[monitored]
            if pair.side == "upper"
            else -post - network.rate_a_mw[monitored]
        )
        assert activity - coupling.rhs == pytest.approx(expected_excess, abs=1e-12)


def test_lagrangian_bound_replays_exact_binary_generator_subproblem() -> None:
    case, _ = triangle_case()
    network = build_network(case)
    master = build_reduced_master(case, network)
    row_dual = np.zeros(master.canonical.num_rows)
    balance = next(row for row in master.coupling_rows if row.kind == "balance_equality")
    row_dual[balance.row_index] = 20.0
    region = RegionMasks.root(1)

    evaluated = evaluate_lagrangian_bound(
        master, row_dual, region, safety_margin_dollars=0.01
    )
    curve = master.costs[0]
    effective = -20.0
    on_value = curve.committed_base_cost + effective * curve.pmin_mw + sum(
        min(0.0, (slope + effective) * width)
        for slope, width in zip(
            curve.segment_slopes_per_mwh, curve.segment_widths_mw, strict=True
        )
    )
    expected = 20.0 * master.operator.total_demand_mw + min(0.0, on_value)
    assert evaluated.raw_lower_bound == pytest.approx(expected)
    assert evaluated.conservative_lower_bound == pytest.approx(expected - 0.01)

    replayed = replay_lagrangian_certificate(
        master, evaluated.as_dict(master.index.generator_source_rows), region
    )
    assert replayed.conservative_lower_bound == evaluated.conservative_lower_bound


def test_upper_dual_projection_preserves_valid_cone_and_prices() -> None:
    case, _ = triangle_case()
    master = build_reduced_master(case, build_network(case))
    row_dual = np.zeros(master.canonical.num_rows)
    balance = next(row for row in master.coupling_rows if row.kind == "balance_equality")
    upper = next(row for row in master.coupling_rows if row.kind == "base_upper")
    row_dual[balance.row_index] = 15.0
    row_dual[upper.row_index] = 3.0
    evaluated = evaluate_lagrangian_bound(
        master, row_dual, RegionMasks.root(1), safety_margin_dollars=0.0
    )
    assert evaluated.projected_dual_sign_violation == 3.0
    assert dict(evaluated.coupling_duals)[upper.row_name] == 0.0
    prices = bus_prices_from_coupling_duals(master, row_dual)
    np.testing.assert_allclose(prices, 15.0)


def test_reduced_vector_helpers_and_disjunctive_cover() -> None:
    case, _ = triangle_case()
    master = build_reduced_master(case, build_network(case))
    values = np.zeros(master.canonical.num_columns)
    values[master.index.commitment_by_generator[0]] = 0.4
    values[master.index.dispatch_by_generator[0]] = 30.0
    np.testing.assert_array_equal(commitment_vector(master, values), [0.4])
    np.testing.assert_array_equal(reduced_dispatch(master, values), [30.0])
    assert choose_split_generator([0.4], [5.0], RegionMasks.root(1)) == 0

    root = RegionMasks.root(1)
    off, on = root.split(0)
    records = [
        {
            "parent_region_id": "r",
            "off_child_region_id": "r0",
            "on_child_region_id": "r1",
            "generator_position": 0,
        }
    ]
    assert verify_disjunctive_cover(1, records, {"r0": off, "r1": on})
    assert not verify_disjunctive_cover(1, records, {"r0": off})
