import numpy as np
import pytest

from activsg_scopf.errors import ScopfError
from activsg_scopf.lagrangian import (
    RegionMasks,
    _generator_breakpoint_state_arrays,
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
    clean_sensitivity_coefficients,
    clean_upper_row_with_box_relaxation,
    commitment_vector,
    reduced_dispatch,
    security_pair_from_record,
    security_pair_record,
)
from activsg_scopf.screening import ContingencyScreener, SecurityPair

from .helpers import triangle_case


def test_generator_breakpoint_states_are_exact_pmin_pmax_pwl_extremes() -> None:
    case, _ = triangle_case()
    case.gen[1, 7] = 1.0
    network = build_network(case)
    master = build_reduced_master(case, network)

    dispatch, cost, commitment = _generator_breakpoint_state_arrays(master)

    assert dispatch.shape == cost.shape == commitment.shape == (2, 12)
    np.testing.assert_array_equal(dispatch[:, 0], np.zeros(2))
    np.testing.assert_array_equal(cost[:, 0], np.zeros(2))
    np.testing.assert_array_equal(commitment[:, 0], np.zeros(2, dtype=np.int8))
    np.testing.assert_array_equal(commitment[:, 1:], np.ones((2, 11), dtype=np.int8))
    for position, generator_index in enumerate(master.index.generator_source_rows):
        curve = master.costs[int(generator_index)]
        assert dispatch[position, 1] == pytest.approx(curve.pmin_mw)
        assert dispatch[position, -1] == pytest.approx(curve.pmax_mw)
        expected_cost = np.asarray(
            [curve.pwl_value(float(power)) for power in dispatch[position, 1:]]
        )
        np.testing.assert_allclose(cost[position, 1:], expected_cost, atol=1e-12)


def test_generator_breakpoint_minimum_matches_exact_lagrangian_subproblems() -> None:
    case, _ = triangle_case()
    case.gen[1, 7] = 1.0
    network = build_network(case)
    master = build_reduced_master(case, network)
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    for number, row in enumerate(sorted(master.coupling_rows, key=lambda item: item.row_name)):
        row_dual[row.row_index] = (
            7.25 if row.kind == "balance_equality" else -0.1 * (number + 1)
        )

    evaluation = evaluate_lagrangian_bound(
        master,
        row_dual,
        RegionMasks.root(master.index.generator_source_rows.size),
        safety_margin_dollars=0.0,
    )
    dispatch, cost, _commitment = _generator_breakpoint_state_arrays(master)
    coupling = sorted(master.coupling_rows, key=lambda row: row.row_name)
    dual = np.asarray([row_dual[row.row_index] for row in coupling])
    coefficients = np.stack([row.generator_coefficients for row in coupling])
    rhs = np.asarray([row.rhs for row in coupling])
    effective = -(dual @ coefficients)
    state_values = cost + effective[:, None] * dispatch
    enumerated = float(dual @ rhs + np.sum(np.min(state_values, axis=1)))

    assert enumerated == pytest.approx(evaluation.raw_lower_bound, abs=1e-10)


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


def test_small_coefficient_cleanup_is_audited_and_upper_row_is_relaxed() -> None:
    coefficients = np.asarray([1e-18, -1e-14, 2e-14, 0.2])
    lower = np.asarray([0.0, -3.0, 0.0, 1.0])
    upper = np.asarray([9.0, 4.0, 8.0, 2.0])
    cleaned, audit = clean_sensitivity_coefficients(
        coefficients, zero_tolerance=1e-14
    )
    np.testing.assert_array_equal(cleaned, [0.0, 0.0, 2e-14, 0.2])
    assert audit["dropped_coefficient_count"] == 2
    assert audit["maximum_absolute_dropped_coefficient"] == pytest.approx(1e-14)

    cleaned, relaxed_rhs, row_audit = clean_upper_row_with_box_relaxation(
        coefficients,
        lower,
        upper,
        5.0,
        zero_tolerance=1e-14,
    )
    vertices = np.asarray(
        [
            [lower[index] if bit & (1 << index) else upper[index] for index in range(4)]
            for bit in range(16)
        ]
    )
    original_activity = vertices @ coefficients
    cleaned_activity = vertices @ cleaned
    implied_slack = cleaned_activity - original_activity
    assert np.max(implied_slack) == pytest.approx(
        row_audit["rhs_outward_relaxation"]
    )
    assert np.all(cleaned_activity <= original_activity + row_audit["rhs_outward_relaxation"])
    assert relaxed_rhs == pytest.approx(5.0 + row_audit["rhs_outward_relaxation"])


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


def test_exact_duplicate_security_rows_keep_pair_provenance_once() -> None:
    case, _ = triangle_case()
    network = build_network(case)
    master = build_reduced_master(case, network)
    rows_before = master.canonical.num_rows
    first = SecurityPair(11, 1, "upper", 0, 0, 1, 0.25)
    alias = SecurityPair(12, 2, "upper", 1, 0, 1, 0.25)

    add_reduced_security_pairs(master, network, (first, alias))

    assert master.canonical.num_rows == rows_before + 1
    assert master.security_pair_ids == {first.pair_id, alias.pair_id}
    assert master.security_pair_representative_by_id == {
        first.pair_id: first.pair_id,
        alias.pair_id: first.pair_id,
    }
    assert master.security_pair_equivalence_classes[first.pair_id] == [
        first.pair_id,
        alias.pair_id,
    ]
    assert master.coefficient_cleanup_audit["exact_duplicate_security_row_count"] == 1


def test_serialized_exact_equivalence_replays_across_tiny_fp64_drift() -> None:
    case, _ = triangle_case()
    network = build_network(case)
    master = build_reduced_master(case, network)
    rows_before = master.canonical.num_rows
    first = SecurityPair(11, 1, "upper", 0, 0, 1, 0.25)
    drifted_alias = SecurityPair(12, 2, "upper", 1, 0, 1, 0.25 + 5e-15)
    representative_map = {
        first.pair_id: first.pair_id,
        drifted_alias.pair_id: first.pair_id,
    }

    add_reduced_security_pairs(
        master,
        network,
        (first, drifted_alias),
        expected_representative_by_pair_id=representative_map,
        equivalence_replay_tolerance=1e-12,
    )

    assert master.canonical.num_rows == rows_before + 1
    assert master.security_pair_representative_by_id == representative_map

    rejected = build_reduced_master(case, network)
    with pytest.raises(ScopfError, match="does not replay within tolerance"):
        add_reduced_security_pairs(
            rejected,
            network,
            (first, drifted_alias),
            expected_representative_by_pair_id=representative_map,
            equivalence_replay_tolerance=1e-16,
        )


def test_security_pair_lodf_replay_allows_only_registered_absolute_tolerance() -> None:
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    outage = catalog.valid[0]
    monitored = 1
    pair = SecurityPair(
        outage.contingency_label,
        int(network.active_branch_source_rows[monitored]) + 1,
        "upper",
        0,
        monitored,
        outage.active_branch_index,
        float(catalog.lodf[monitored, 0]),
    )
    record = security_pair_record(pair)
    record["lodf_value"] = float(record["lodf_value"]) + 5e-15

    with pytest.raises(ScopfError, match="LODF differs"):
        security_pair_from_record(record, catalog)
    replayed = security_pair_from_record(record, catalog, lodf_absolute_tolerance=1e-12)
    assert replayed.pair_id == pair.pair_id


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
