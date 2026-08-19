import hashlib

import numpy as np
import pytest

from activsg_scopf.commitment_cuts import (
    CommitmentFeasibilityCut,
    _relax_commitment_cut_coefficient_dust,
    build_commitment_cardinality_cut,
    commitment_capacity_cut_from_record,
    commitment_cover_cut_from_record,
    commitment_feasibility_cut_from_record,
    commitment_upper_cut_from_record,
    derive_binary_knapsack_cover_cut,
    derive_commitment_capacity_cut,
    derive_commitment_capacity_cut_for_side,
    derive_commitment_feasibility_cut,
    generate_commitment_cut_repairs,
    verify_binary_knapsack_cover_derivation,
)
from activsg_scopf.errors import ScopfError
from activsg_scopf.fixed_commitment import build_fixed_commitment_projection
from activsg_scopf.lagrangian import (
    RegionMasks,
    _coordinate_ascent_commitment_cut_arrays,
    evaluate_lagrangian_bound,
    evaluate_lagrangian_bound_cupy,
    optimize_commitment_cut_duals_coordinate_numpy,
    optimize_lagrangian_bound_cupy,
    optimize_lagrangian_bound_cupy_adam,
    replay_lagrangian_certificate,
)
from activsg_scopf.network import build_network
from activsg_scopf.phase_one import build_phase_one_model, phase_one_certificate
from activsg_scopf.reduced import CouplingRow, build_reduced_master, fix_commitments

from .helpers import triangle_case


def test_binary_knapsack_cover_strengthens_mixed_sign_parent_cut() -> None:
    source = np.asarray([1, 0, 1], dtype=np.int8)
    parent = CommitmentFeasibilityCut(
        cut_id="fc_test_mixed",
        coefficients=np.asarray([4.0, -3.0, 2.0]),
        rhs=1.0,
        source_commitment_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        conservative_source_violation_pu=5.0,
    )
    rows = np.asarray([11, 12, 13], dtype=np.int64)
    reference = np.asarray([0.8, 0.1, 0.2])

    cover, audit = derive_binary_knapsack_cover_cut(
        source_cut=parent,
        generator_source_rows=rows,
        source_commitment=source,
        separation_reference=reference,
    )

    np.testing.assert_array_equal(cover.coefficients, np.asarray([1.0, -1.0, 0.0]))
    assert cover.rhs == 0.0
    assert cover.violation(source) == 1.0
    assert cover.violation(reference) == pytest.approx(0.7)
    assert audit["verification"]["continuous_relaxation_strengthened"] is True
    assert audit["verification"]["original_binary_feasible_set_changed"] is False
    for encoded in range(8):
        binary = np.asarray([(encoded >> position) & 1 for position in range(3)])
        if parent.violation(binary) <= 0.0:
            assert cover.violation(binary) <= 0.0

    serialized = cover.as_dict(rows)
    rebuilt = commitment_cover_cut_from_record(serialized, rows)
    generic = commitment_upper_cut_from_record(serialized, rows)
    np.testing.assert_array_equal(rebuilt.coefficients, cover.coefficients)
    assert generic.cut_id == cover.cut_id
    replay = verify_binary_knapsack_cover_derivation(parent, rebuilt, rows)
    assert replay["passed"] is True

    reference_only, reference_audit = derive_binary_knapsack_cover_cut(
        source_cut=parent,
        generator_source_rows=rows,
        source_commitment=None,
        separation_reference=reference,
    )
    np.testing.assert_array_equal(reference_only.coefficients, cover.coefficients)
    assert reference_audit["source_parent_violation_pu"] is None
    assert reference_only.violation(reference) == pytest.approx(0.7)


def test_weight_dominance_extended_cover_is_valid_and_replayable() -> None:
    source = np.asarray([1, 0, 0], dtype=np.int8)
    parent = CommitmentFeasibilityCut(
        cut_id="fc_test_extended_cover",
        coefficients=np.asarray([4.0, -3.0, -3.0]),
        rhs=-0.5,
        source_commitment_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        conservative_source_violation_pu=4.5,
    )
    rows = np.asarray([11, 12, 13], dtype=np.int64)
    # The fractional reference selects rows 12 and 13 as the strict core.
    # Row 11 is then a valid extension because its weight (4) dominates the
    # maximum core weight (3).
    reference = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    cover, audit = derive_binary_knapsack_cover_cut(
        source_cut=parent,
        generator_source_rows=rows,
        source_commitment=source,
        separation_reference=reference,
        extend_cover=True,
    )

    np.testing.assert_array_equal(cover.coefficients, np.asarray([1.0, -1.0, -1.0]))
    assert cover.rhs == -1.0
    assert cover.core_cover_source_rows == (12, 13)
    assert cover.extended_source_rows == (11,)
    assert cover.extension_weight_threshold == 3.0
    assert audit["verification"]["core_cover_size"] == 2
    assert audit["verification"]["extended_item_count"] == 1
    for encoded in range(8):
        binary = np.asarray([(encoded >> position) & 1 for position in range(3)])
        if parent.violation(binary) <= 0.0:
            assert cover.violation(binary) <= 0.0

    serialized = cover.as_dict(rows)
    assert serialized["certificate_kind"] == (
        "binary_knapsack_extended_cover_from_commitment_cut_v2"
    )
    rebuilt = commitment_cover_cut_from_record(serialized, rows)
    generic = commitment_upper_cut_from_record(serialized, rows)
    np.testing.assert_array_equal(rebuilt.coefficients, cover.coefficients)
    assert generic.cut_id == cover.cut_id
    replay = verify_binary_knapsack_cover_derivation(parent, rebuilt, rows)
    assert replay["passed"] is True


def test_binary_knapsack_cover_rejects_nonmatching_source_commitment() -> None:
    source = np.asarray([1, 1], dtype=np.int8)
    parent = CommitmentFeasibilityCut(
        cut_id="fc_test_source",
        coefficients=np.asarray([2.0, 2.0]),
        rhs=3.0,
        source_commitment_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        conservative_source_violation_pu=1.0,
    )

    with pytest.raises(ScopfError, match="does not match"):
        derive_binary_knapsack_cover_cut(
            source_cut=parent,
            generator_source_rows=np.asarray([1, 2]),
            source_commitment=np.asarray([1, 0], dtype=np.int8),
        )


def test_cut_coordinate_uses_local_delta_below_absolute_objective_ulp() -> None:
    dual, raw, _on_values, commitment, cycle_raw, tie_tolerance = (
        _coordinate_ascent_commitment_cut_arrays(
            xp=np,
            base_on_values=np.asarray([1e-4]),
            coupling_constant=np.asarray(1e15),
            cut_coefficients=np.asarray([[-1.0]]),
            cut_rhs=np.asarray([-1.0]),
            initial_cut_dual=np.asarray([0.0]),
            fixed_off=np.asarray([False]),
            fixed_on=np.asarray([False]),
            cycles=1,
        )
    )

    # The true 1e-4 coordinate improvement is below one ULP of the 1e15
    # absolute objective.  Stable local deltas must still move through the
    # kink to the cut-satisfying tied minimizer.
    assert dual[0] < -1e-4
    assert commitment[0] == 1
    assert raw == 1e15
    assert tie_tolerance > 0.0
    np.testing.assert_array_equal(cycle_raw, np.asarray([1e15, 1e15]))


def test_direct_row_capacity_cut_replays_exact_pmin_pmax_envelope() -> None:
    case, _ = triangle_case()
    case.gen[1, 7] = 1.0
    network = build_network(case)
    master = build_reduced_master(case, network)
    source_rows = master.index.generator_source_rows
    dispatch_columns = [
        master.index.dispatch_by_generator[int(generator)] for generator in source_rows
    ]
    row = master.canonical.add_row(
        "test_minimum_activity_limit",
        {dispatch_columns[0]: 1.0, dispatch_columns[1]: 1.0},
        upper=20.0,
    )
    master.coupling_rows.append(
        CouplingRow(
            row_index=row,
            row_name="test_minimum_activity_limit",
            rhs=20.0,
            generator_coefficients=np.asarray([1.0, 1.0]),
            bus_coefficients=np.zeros(network.bus_ids.size),
            kind="test_upper",
        )
    )
    source = np.asarray([1, 0], dtype=np.int8)

    cut, audit = derive_commitment_capacity_cut(
        master=master,
        source_row_name="test_minimum_activity_limit",
        source_commitment=source,
        base_mva=100.0,
        safety_margin_pu=1e-8,
    )

    np.testing.assert_array_equal(cut.coefficients, np.asarray([0.25, 0.10]))
    assert cut.rhs == pytest.approx(0.20000001)
    assert cut.violation(source) == pytest.approx(0.04999999)
    assert cut.violation(np.asarray([0, 1], dtype=np.int8)) < 0.0
    assert audit["source_row_side"] == "upper"
    assert audit["outward_rhs_relaxation_pu"] >= 1e-8
    serialized = cut.as_dict(source_rows + 1)
    rebuilt = commitment_capacity_cut_from_record(serialized, source_rows + 1)
    generic = commitment_upper_cut_from_record(serialized, source_rows + 1)
    assert rebuilt.cut_id == cut.cut_id == generic.cut_id
    np.testing.assert_array_equal(rebuilt.coefficients, cut.coefficients)


def test_direct_lower_row_capacity_cut_uses_conditional_pmax() -> None:
    case, _ = triangle_case()
    case.gen[1, 7] = 1.0
    network = build_network(case)
    master = build_reduced_master(case, network)
    source_rows = master.index.generator_source_rows
    dispatch_columns = [
        master.index.dispatch_by_generator[int(generator)] for generator in source_rows
    ]
    row = master.canonical.add_row(
        "test_maximum_activity_requirement",
        {dispatch_columns[0]: 1.0, dispatch_columns[1]: 1.0},
        lower=60.0,
    )
    master.coupling_rows.append(
        CouplingRow(
            row_index=row,
            row_name="test_maximum_activity_requirement",
            rhs=60.0,
            generator_coefficients=np.asarray([1.0, 1.0]),
            bus_coefficients=np.zeros(network.bus_ids.size),
            kind="test_lower",
        )
    )

    cut, audit = derive_commitment_capacity_cut(
        master=master,
        source_row_name="test_maximum_activity_requirement",
        source_commitment=np.asarray([0, 1], dtype=np.int8),
        base_mva=100.0,
        safety_margin_pu=1e-8,
    )

    np.testing.assert_array_equal(cut.coefficients, np.asarray([-1.0, -0.5]))
    assert cut.rhs == pytest.approx(-0.59999999)
    assert cut.violation(np.asarray([0, 1], dtype=np.int8)) == pytest.approx(0.09999999)
    assert cut.violation(np.asarray([1, 0], dtype=np.int8)) < 0.0
    assert audit["source_row_side"] == "lower"


def test_analytic_row_capacity_cut_uses_maximally_violating_binary() -> None:
    case, _ = triangle_case()
    case.gen[1, 7] = 1.0
    network = build_network(case)
    master = build_reduced_master(case, network)
    source_rows = master.index.generator_source_rows
    dispatch_columns = [
        master.index.dispatch_by_generator[int(generator)] for generator in source_rows
    ]
    row = master.canonical.add_row(
        "test_analytic_capacity_limit",
        {dispatch_columns[0]: 1.0, dispatch_columns[1]: 1.0},
        upper=20.0,
    )
    master.coupling_rows.append(
        CouplingRow(
            row_index=row,
            row_name="test_analytic_capacity_limit",
            rhs=20.0,
            generator_coefficients=np.asarray([1.0, 1.0]),
            bus_coefficients=np.zeros(network.bus_ids.size),
            kind="test_upper",
        )
    )

    cut, audit = derive_commitment_capacity_cut_for_side(
        master=master,
        source_row_name="test_analytic_capacity_limit",
        source_row_side="upper",
        base_mva=100.0,
        safety_margin_pu=1e-8,
    )

    np.testing.assert_array_equal(cut.coefficients, np.asarray([0.25, 0.10]))
    analytic_source = np.asarray([1, 1], dtype=np.int8)
    assert cut.source_commitment_sha256 == hashlib.sha256(
        analytic_source.tobytes()
    ).hexdigest()
    assert cut.violation(analytic_source) == pytest.approx(
        cut.conservative_source_violation_pu
    )
    assert audit["analytic_source_commitment_generator_rows"] == [1, 2]
    assert audit["analytic_source_commitment_policy"].endswith("_v1")
    for encoded in range(4):
        binary = np.asarray([(encoded >> position) & 1 for position in range(2)])
        if cut.violation(binary) <= 0.0:
            minimum_dispatch_activity = float(
                np.asarray([25.0, 10.0]) @ binary
            )
            assert minimum_dispatch_activity <= 20.0 + 1e-5


def test_exact_coordinate_ascent_strengthens_at_most_commitment_cut() -> None:
    case, _table = triangle_case()
    master = build_reduced_master(case, build_network(case))
    rows = master.index.generator_source_rows
    cut = build_commitment_cardinality_cut(
        generator_source_rows=rows,
        subset_positions=np.asarray([0], dtype=np.int64),
        subset_id="only_generator",
        branch_side="at_most",
        integer_threshold=0,
    )
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    balance = next(row for row in master.coupling_rows if row.kind == "balance_equality")
    row_dual[balance.row_index] = 20.0
    region = RegionMasks.root(1)
    initial = evaluate_lagrangian_bound(
        master,
        row_dual,
        region,
        safety_margin_dollars=0.0,
        commitment_cuts=(cut,),
        commitment_cut_dual=np.asarray([0.0]),
    )

    cut_dual, audit = optimize_commitment_cut_duals_coordinate_numpy(
        master,
        row_dual,
        region,
        commitment_cuts=(cut,),
        initial_commitment_cut_dual=np.asarray([0.0]),
        cycles=2,
    )
    strengthened = evaluate_lagrangian_bound(
        master,
        row_dual,
        region,
        safety_margin_dollars=0.0,
        commitment_cuts=(cut,),
        commitment_cut_dual=cut_dual,
    )

    assert cut_dual[0] < 0.0
    assert strengthened.raw_lower_bound > initial.raw_lower_bound
    assert strengthened.raw_lower_bound == pytest.approx(audit["best_raw_lower_bound"])
    assert np.all(np.diff(audit["cycle_raw_lower_bounds"]) >= -1e-8)


def test_exact_coordinate_ascent_handles_negated_at_least_cut() -> None:
    case, _table = triangle_case()
    master = build_reduced_master(case, build_network(case))
    rows = master.index.generator_source_rows
    cut = build_commitment_cardinality_cut(
        generator_source_rows=rows,
        subset_positions=np.asarray([0], dtype=np.int64),
        subset_id="only_generator",
        branch_side="at_least",
        integer_threshold=1,
    )
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    region = RegionMasks.root(1)
    initial = evaluate_lagrangian_bound(
        master,
        row_dual,
        region,
        safety_margin_dollars=0.0,
        commitment_cuts=(cut,),
        commitment_cut_dual=np.asarray([0.0]),
    )

    cut_dual, audit = optimize_commitment_cut_duals_coordinate_numpy(
        master,
        row_dual,
        region,
        commitment_cuts=(cut,),
        initial_commitment_cut_dual=np.asarray([0.0]),
        cycles=2,
    )
    strengthened = evaluate_lagrangian_bound(
        master,
        row_dual,
        region,
        safety_margin_dollars=0.0,
        commitment_cuts=(cut,),
        commitment_cut_dual=cut_dual,
    )

    assert cut.coefficients[0] == -1.0
    assert cut.rhs == -1.0
    assert cut_dual[0] < 0.0
    assert strengthened.raw_lower_bound > initial.raw_lower_bound
    assert strengthened.raw_lower_bound == pytest.approx(audit["best_raw_lower_bound"])
    assert strengthened.minimizing_commitment[0] == 1
    assert (
        float(cut.coefficients @ strengthened.minimizing_commitment - cut.rhs)
        <= 0.0
    )


def test_exact_hard_cardinality_subproblem_is_replayable() -> None:
    case, _table = triangle_case()
    case.gen[1, 7] = 1.0
    master = build_reduced_master(case, build_network(case))
    rows = master.index.generator_source_rows
    cut = build_commitment_cardinality_cut(
        generator_source_rows=rows,
        subset_positions=np.asarray([0, 1], dtype=np.int64),
        subset_id="both_generators",
        branch_side="at_least",
        integer_threshold=1,
    )
    region = RegionMasks.root(2)
    evaluation = evaluate_lagrangian_bound(
        master,
        np.zeros(master.canonical.num_rows, dtype=np.float64),
        region,
        safety_margin_dollars=0.0,
        commitment_cuts=(cut,),
        commitment_cut_dual=np.asarray([0.0]),
        hard_cardinality_cuts=(cut,),
    )

    assert evaluation.hard_cardinality_cut_ids == (cut.cut_id,)
    assert int(np.sum(evaluation.minimizing_commitment)) == 1
    assert float(cut.coefficients @ evaluation.minimizing_commitment - cut.rhs) == 0.0
    assert evaluation.minimizing_commitment[1] == 1
    certificate = evaluation.as_dict(rows + 1, compact=True)
    replayed = replay_lagrangian_certificate(
        master,
        certificate,
        region,
        commitment_cuts_by_id={cut.cut_id: cut},
    )
    assert certificate["certificate_kind"].endswith("cardinality_v4")
    assert replayed.raw_lower_bound == pytest.approx(evaluation.raw_lower_bound)
    np.testing.assert_array_equal(
        replayed.minimizing_commitment,
        evaluation.minimizing_commitment,
    )


def test_exact_hard_cardinality_subproblem_rejects_overlapping_supports() -> None:
    case, _table = triangle_case()
    case.gen[1, 7] = 1.0
    master = build_reduced_master(case, build_network(case))
    rows = master.index.generator_source_rows
    at_most = build_commitment_cardinality_cut(
        generator_source_rows=rows,
        subset_positions=np.asarray([0, 1], dtype=np.int64),
        subset_id="overlap_at_most",
        branch_side="at_most",
        integer_threshold=1,
    )
    at_least = build_commitment_cardinality_cut(
        generator_source_rows=rows,
        subset_positions=np.asarray([1], dtype=np.int64),
        subset_id="overlap_at_least",
        branch_side="at_least",
        integer_threshold=1,
    )

    with pytest.raises(ScopfError, match="supports overlap"):
        evaluate_lagrangian_bound(
            master,
            np.zeros(master.canonical.num_rows, dtype=np.float64),
            RegionMasks.root(2),
            safety_margin_dollars=0.0,
            commitment_cuts=(at_most, at_least),
            commitment_cut_dual=np.zeros(2, dtype=np.float64),
            hard_cardinality_cuts=(at_most, at_least),
        )


def test_gpu_exact_hard_cardinality_subproblem_matches_host_fp64() -> None:
    pytest.importorskip("cupy")
    case, _table = triangle_case()
    case.gen[1, 7] = 1.0
    master = build_reduced_master(case, build_network(case))
    rows = master.index.generator_source_rows
    cut = build_commitment_cardinality_cut(
        generator_source_rows=rows,
        subset_positions=np.asarray([0, 1], dtype=np.int64),
        subset_id="gpu_both_generators",
        branch_side="at_least",
        integer_threshold=1,
    )
    dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    region = RegionMasks.root(2)
    host = evaluate_lagrangian_bound(
        master,
        dual,
        region,
        safety_margin_dollars=0.0,
        commitment_cuts=(cut,),
        commitment_cut_dual=np.asarray([0.0]),
        hard_cardinality_cuts=(cut,),
    )
    gpu = evaluate_lagrangian_bound_cupy(
        master,
        dual,
        region,
        commitment_cuts=(cut,),
        commitment_cut_dual=np.asarray([0.0]),
        hard_cardinality_cuts=(cut,),
    )

    assert gpu["raw_lower_bound"] == pytest.approx(host.raw_lower_bound, abs=1e-9)
    np.testing.assert_array_equal(
        gpu["minimizing_commitment"], host.minimizing_commitment
    )
    assert gpu["hard_cardinality_cut_ids"] == [cut.cut_id]


def test_gpu_hard_cardinality_multiplier_polish_matches_host_fp64() -> None:
    pytest.importorskip("cupy")
    case, _table = triangle_case()
    case.gen[1, 7] = 1.0
    master = build_reduced_master(case, build_network(case))
    cut = build_commitment_cardinality_cut(
        generator_source_rows=master.index.generator_source_rows,
        subset_positions=np.asarray([0, 1], dtype=np.int64),
        subset_id="gpu_polish_both_generators",
        branch_side="at_least",
        integer_threshold=1,
    )
    initial_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    region = RegionMasks.root(2)
    initial = evaluate_lagrangian_bound(
        master,
        initial_dual,
        region,
        safety_margin_dollars=0.0,
        commitment_cuts=(cut,),
        commitment_cut_dual=np.asarray([0.0]),
        hard_cardinality_cuts=(cut,),
    )

    polished_dual, audit = optimize_lagrangian_bound_cupy(
        master,
        initial_dual,
        region,
        relaxation_primal_objective=1_000.0,
        iterations=8,
        polyak_fraction=0.5,
        commitment_cuts=(cut,),
        initial_commitment_cut_dual=np.asarray([0.0]),
        hard_cardinality_cuts=(cut,),
    )
    replay = evaluate_lagrangian_bound(
        master,
        polished_dual,
        region,
        safety_margin_dollars=0.0,
        commitment_cuts=(cut,),
        commitment_cut_dual=np.asarray(audit["best_commitment_cut_dual"]),
        hard_cardinality_cuts=(cut,),
    )

    assert audit["hard_cardinality_subproblem_reoptimized"] is True
    assert audit["hard_cardinality_cut_ids"] == [cut.cut_id]
    assert audit["nonfinite_projected_update_iterations"] == 0
    assert replay.raw_lower_bound >= initial.raw_lower_bound - 1e-9
    assert replay.raw_lower_bound == pytest.approx(
        audit["best_raw_lower_bound"], abs=1e-9
    )
    assert float(cut.coefficients @ replay.minimizing_commitment - cut.rhs) <= 0.0


def test_gpu_multirate_adam_hard_cardinality_polish_matches_host_fp64() -> None:
    pytest.importorskip("cupy")
    case, _table = triangle_case()
    case.gen[1, 7] = 1.0
    master = build_reduced_master(case, build_network(case))
    cut = build_commitment_cardinality_cut(
        generator_source_rows=master.index.generator_source_rows,
        subset_positions=np.asarray([0, 1], dtype=np.int64),
        subset_id="gpu_adam_both_generators",
        branch_side="at_least",
        integer_threshold=1,
    )
    initial_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    region = RegionMasks.root(2)
    initial = evaluate_lagrangian_bound(
        master,
        initial_dual,
        region,
        safety_margin_dollars=0.0,
        commitment_cuts=(cut,),
        commitment_cut_dual=np.asarray([0.0]),
        hard_cardinality_cuts=(cut,),
    )

    polished_dual, audit = optimize_lagrangian_bound_cupy_adam(
        master,
        initial_dual,
        region,
        iterations=32,
        learning_rates=(0.01, 0.1),
        commitment_cuts=(cut,),
        initial_commitment_cut_dual=np.asarray([0.0]),
        hard_cardinality_cuts=(cut,),
    )
    replay = evaluate_lagrangian_bound(
        master,
        polished_dual,
        region,
        safety_margin_dollars=0.0,
        commitment_cuts=(cut,),
        commitment_cut_dual=np.asarray(audit["best_commitment_cut_dual"]),
        hard_cardinality_cuts=(cut,),
    )

    assert audit["selected_learning_rate"] in {0.01, 0.1}
    assert replay.raw_lower_bound >= initial.raw_lower_bound - 1e-9
    assert replay.raw_lower_bound == pytest.approx(
        audit["best_raw_lower_bound"], abs=1e-9
    )
    assert not np.any(audit["nonfinite_lagrangian_evaluation_iterations_by_lane"])
    assert not np.any(audit["nonfinite_projected_update_iterations_by_lane"])
    assert float(cut.coefficients @ replay.minimizing_commitment - cut.rhs) <= 0.0


def test_commitment_cut_dust_cleanup_is_an_outward_relaxation() -> None:
    coefficients = np.asarray([1e-12, -2e-12, 0.5, -0.25])
    source = np.asarray([1, 0, 1, 0], dtype=np.int8)
    rhs = 0.1
    source_violation = float(coefficients @ source - rhs)

    cleaned, cleaned_rhs, cleaned_source_violation, audit = _relax_commitment_cut_coefficient_dust(
        coefficients=coefficients,
        rhs=rhs,
        source_commitment=source,
        source_violation_pu=source_violation,
        requested_zero_tolerance=1e-8,
    )

    np.testing.assert_array_equal(cleaned, np.asarray([0.0, 0.0, 0.5, -0.25]))
    assert cleaned_rhs == pytest.approx(rhs + 2e-12)
    assert cleaned_source_violation > 0.0
    assert audit["dropped_coefficient_count"] == 2
    assert audit["outward_rhs_relaxation"] == pytest.approx(2e-12)
    assert audit["original_feasible_commitment_can_be_removed"] is False
    for encoded in range(16):
        binary = np.asarray([(encoded >> position) & 1 for position in range(4)])
        if float(coefficients @ binary) <= rhs:
            assert float(cleaned @ binary) <= cleaned_rhs


def test_phase_one_dual_lifts_to_global_exact_pmin_pmax_commitment_cut() -> None:
    case, _ = triangle_case()
    case.gen[1, 7] = 1.0
    network = build_network(case)
    master = build_reduced_master(case, network)
    source_rows = master.index.generator_source_rows
    assert source_rows.tolist() == [0, 1]

    dispatch_columns = [
        master.index.dispatch_by_generator[int(generator)] for generator in source_rows
    ]
    row = master.canonical.add_row(
        "test_transfer_limit",
        {dispatch_columns[0]: 1.0},
        upper=50.0,
    )
    master.coupling_rows.append(
        CouplingRow(
            row_index=row,
            row_name="test_transfer_limit",
            rhs=50.0,
            generator_coefficients=np.asarray([1.0, 0.0]),
            bus_coefficients=np.zeros(network.bus_ids.size),
            kind="test_upper",
        )
    )
    source_commitment = np.asarray([1, 0], dtype=np.int8)
    fix_commitments(master, source_commitment == 0, source_commitment == 1)
    projection = build_fixed_commitment_projection(master, source_commitment)
    phase = build_phase_one_model(
        projection.canonical,
        base_mva=100.0,
        maximum_violation_pu=1e6,
    )
    dual = np.zeros(phase.num_rows)
    for position, name in enumerate(phase.row_names):
        if name.endswith("__lower__lag_balance") or name.endswith("__upper__test_transfer_limit"):
            dual[position] = -0.005
    certificate = phase_one_certificate(
        phase,
        dual,
        safety_margin_pu=1e-8,
        infeasibility_threshold_pu=1e-6,
    )
    assert certificate["prune_certified"] is True
    assert certificate["raw_lower_bound_pu"] == pytest.approx(0.06)

    cut, audit = derive_commitment_feasibility_cut(
        master=master,
        phase_model=phase,
        phase_certificate=certificate,
        source_commitment=source_commitment,
        replay_tolerance_pu=1e-12,
    )

    assert cut.coefficients[0] == pytest.approx(0.0)
    assert cut.coefficients[1] == pytest.approx(-0.25)
    assert cut.rhs == pytest.approx(-0.05999999)
    assert cut.violation(source_commitment) == pytest.approx(0.05999999)
    assert cut.violation(np.asarray([1, 1], dtype=np.int8)) < 0.0
    assert audit["conservative_source_replay_difference_pu"] < 1e-14

    serialized_cut = cut.as_dict(source_rows)
    rebuilt_cut = commitment_feasibility_cut_from_record(serialized_cut, source_rows + 1)
    generic_cut = commitment_upper_cut_from_record(serialized_cut, source_rows + 1)
    assert rebuilt_cut.cut_id == cut.cut_id
    np.testing.assert_array_equal(rebuilt_cut.coefficients, cut.coefficients)
    assert generic_cut.cut_id == cut.cut_id

    repairs = generate_commitment_cut_repairs(
        cut=cut,
        commitment=source_commitment,
        fixed_off=np.zeros(2, dtype=bool),
        fixed_on=np.zeros(2, dtype=bool),
        pmin_mw=case.gen[source_rows, 9],
        pmax_mw=case.gen[source_rows, 8],
        demand_mw=master.operator.total_demand_mw,
        economic_on_values=np.asarray([0.0, 1.0]),
        maximum_repairs=3,
    )
    assert len(repairs) == 1
    np.testing.assert_array_equal(repairs[0][0], np.asarray([1, 1], dtype=np.int8))
    assert repairs[0][1]["repaired_cut_violation_pu"] < 0.0

    evaluation = evaluate_lagrangian_bound(
        master,
        np.zeros(master.canonical.num_rows),
        RegionMasks.root(2),
        safety_margin_dollars=0.01,
        commitment_cuts=(cut,),
        commitment_cut_dual=np.asarray([-10.0]),
    )
    serialized = evaluation.as_dict(source_rows + 1)
    replayed = replay_lagrangian_certificate(
        master,
        serialized,
        RegionMasks.root(2),
        commitment_cuts_by_id={cut.cut_id: cut},
    )
    assert serialized["certificate_kind"].endswith("commitment_upper_cuts_v3")
    assert replayed.raw_lower_bound == pytest.approx(evaluation.raw_lower_bound)
    np.testing.assert_array_equal(replayed.minimizing_commitment, evaluation.minimizing_commitment)
    compact = evaluation.as_dict(source_rows + 1, compact=True)
    compact_replayed = replay_lagrangian_certificate(
        master,
        compact,
        RegionMasks.root(2),
        commitment_cuts_by_id={cut.cut_id: cut},
    )
    assert compact["serialization"] == ("sparse_nonzero_dual_order_independent_identity_v3")
    assert "generator_subproblems" not in compact
    assert "effective_dispatch_coefficient_sha256" not in compact
    assert compact_replayed.conservative_lower_bound == pytest.approx(
        evaluation.conservative_lower_bound
    )

    original_order = [row.row_name for row in master.coupling_rows]
    master.coupling_rows.reverse()
    assert [row.row_name for row in master.coupling_rows] != original_order
    reordered_replay = replay_lagrangian_certificate(
        master,
        compact,
        RegionMasks.root(2),
        commitment_cuts_by_id={cut.cut_id: cut},
    )
    assert reordered_replay.conservative_lower_bound == pytest.approx(
        evaluation.conservative_lower_bound
    )
