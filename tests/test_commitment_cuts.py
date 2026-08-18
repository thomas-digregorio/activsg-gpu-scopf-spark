import numpy as np
import pytest

from activsg_scopf.commitment_cuts import (
    derive_commitment_feasibility_cut,
    generate_commitment_cut_repairs,
)
from activsg_scopf.fixed_commitment import build_fixed_commitment_projection
from activsg_scopf.lagrangian import (
    RegionMasks,
    evaluate_lagrangian_bound,
    replay_lagrangian_certificate,
)
from activsg_scopf.network import build_network
from activsg_scopf.phase_one import build_phase_one_model, phase_one_certificate
from activsg_scopf.reduced import CouplingRow, build_reduced_master, fix_commitments

from .helpers import triangle_case


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
        if name.endswith("__lower__lag_balance") or name.endswith(
            "__upper__test_transfer_limit"
        ):
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
    assert serialized["certificate_kind"].endswith("feasibility_cuts_v2")
    assert replayed.raw_lower_bound == pytest.approx(evaluation.raw_lower_bound)
    np.testing.assert_array_equal(
        replayed.minimizing_commitment, evaluation.minimizing_commitment
    )
    compact = evaluation.as_dict(source_rows + 1, compact=True)
    compact_replayed = replay_lagrangian_certificate(
        master,
        compact,
        RegionMasks.root(2),
        commitment_cuts_by_id={cut.cut_id: cut},
    )
    assert compact["serialization"] == (
        "sparse_nonzero_dual_order_independent_identity_v3"
    )
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
