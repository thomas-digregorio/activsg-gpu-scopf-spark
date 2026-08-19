import numpy as np
import pytest

from activsg_scopf.cardinality import (
    balanced_exact_type_group_rounding_candidates,
    choose_cardinality_split,
    commitment_branch_subsets,
    exact_cost_type_groups,
    exact_type_group_rounding,
)
from activsg_scopf.commitment_cuts import (
    add_commitment_upper_cuts,
    commitment_cardinality_cut_from_record,
)
from activsg_scopf.lagrangian import RegionMasks, verify_cardinality_disjunctive_cover
from activsg_scopf.network import build_network
from activsg_scopf.reduced import build_reduced_master

from .helpers import triangle_case


def _two_identical_type_master():
    case, _ = triangle_case()
    case.gen[1, 7] = 1.0
    case.gen[1, 8] = case.gen[0, 8]
    case.gen[1, 9] = case.gen[0, 9]
    case.gencost[1, :] = case.gencost[0, :]
    return build_reduced_master(case, build_network(case))


def test_exact_type_group_rounding_preserves_lp_preferred_placement() -> None:
    master = _two_identical_type_master()
    groups = exact_cost_type_groups(master)
    assert len(groups) == 1
    np.testing.assert_array_equal(groups[0], np.asarray([0, 1]))

    rounded, audit = exact_type_group_rounding(master, np.asarray([0.70, 0.60]))

    np.testing.assert_array_equal(rounded, np.asarray([1.0, 0.0]))
    assert audit["type_group_count"] == 1
    assert audit["fractional_type_sum_count"] == 1
    assert audit["groups"][0]["lp_sum"] == pytest.approx(1.3)
    assert audit["cpu_solution_data_used"] is False


def test_balanced_type_group_rounding_controls_global_count() -> None:
    master = _two_identical_type_master()

    candidates = balanced_exact_type_group_rounding_candidates(
        master,
        np.asarray([0.70, 0.60]),
        target_offsets=(-1, 0, 1),
    )

    assert [int(np.count_nonzero(candidate)) for candidate, _audit in candidates] == [1, 2]
    np.testing.assert_array_equal(candidates[0][0], np.asarray([1.0, 0.0]))
    for candidate, audit in candidates:
        assert int(np.count_nonzero(candidate)) == audit["global_commitment_count"]
        assert audit["exact_source_pmin_pmax_retained"] is True
        assert audit["candidate_only_not_feasibility_proof"] is True
        assert audit["cpu_solution_data_used"] is False


def test_cardinality_cut_roundtrip_rows_and_exhaustive_cover() -> None:
    master = _two_identical_type_master()
    exact_type_subsets = tuple(
        subset
        for subset in commitment_branch_subsets(master)
        if subset.family == "exact_type"
    )
    assert all(subset.positions.size >= 2 for subset in commitment_branch_subsets(master))
    split = choose_cardinality_split(
        master=master,
        commitments=np.asarray([0.70, 0.60]),
        subsets=exact_type_subsets,
        existing_cut_ids=set(),
    )
    assert split.floor_value == 1
    assert split.ceil_value == 2
    assert split.subset.family == "exact_type"
    assert split.subset.positions.size == 2
    assert split.at_most_cut.rhs == 1.0
    assert split.at_least_cut.rhs == -2.0

    source_rows = master.index.generator_source_rows + 1
    for cut in (split.at_most_cut, split.at_least_cut):
        record = cut.as_dict(source_rows)
        replayed = commitment_cardinality_cut_from_record(record, source_rows)
        assert replayed.cut_id == cut.cut_id
        np.testing.assert_array_equal(replayed.coefficients, cut.coefficients)

    row_by_id = add_commitment_upper_cuts(master, (split.at_most_cut,))
    row = row_by_id[split.at_most_cut.cut_id]
    indices, values = master.canonical.row_entries(row)
    assert len(indices) == 2
    np.testing.assert_array_equal(values, np.ones(2))
    assert master.canonical.row_upper[row] == 1.0

    record = {
        "parent_region_id": "r",
        "off_child_region_id": "r0",
        "on_child_region_id": "r1",
        **split.as_dict(),
    }
    root = RegionMasks.root(2)
    leaves = {
        "r0": (root, (split.at_most_cut,)),
        "r1": (root, (split.at_least_cut,)),
    }
    assert verify_cardinality_disjunctive_cover(source_rows, [record], leaves)
    tampered = dict(record)
    tampered["ceil_value"] = 3
    assert not verify_cardinality_disjunctive_cover(source_rows, [tampered], leaves)
