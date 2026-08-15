from dataclasses import replace

import numpy as np

from activsg_scopf.matpower import ContingencyChange, ContingencyTable
from activsg_scopf.network import build_contingency_catalog, build_network, solve_dc

from .helpers import triangle_case


def test_lodf_columns_match_explicit_tap_and_shift_outage_solves() -> None:
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(
        case, network, table, validation_columns=3, validation_tolerance_pu=1e-10
    )
    assert len(catalog.valid) == 3
    assert len(catalog.deferred_generator_outages) == 1
    assert catalog.validation_max_error_pu < 1e-10
    injections = np.asarray([62.0, -42.0, -20.0])
    _, base_flow = solve_dc(network, injections)
    for column, outage in enumerate(catalog.valid):
        _, explicit = solve_dc(network, injections, outage_active_index=outage.active_branch_index)
        predicted = base_flow + catalog.lodf[:, column] * base_flow[outage.active_branch_index]
        np.testing.assert_allclose(predicted, explicit, atol=1e-10)


def test_linear_time_bridge_catalog_excludes_islanding_source_outage() -> None:
    case, table = triangle_case()
    extra_bus = np.asarray([[4, 1, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9]])
    extra_branch = np.asarray([[3, 4, 0, 0.1, 0, 100, 0, 0, 0, 0, 1, 0, 0]])
    case = replace(
        case,
        bus=np.vstack((case.bus, extra_bus)),
        branch=np.vstack((case.branch, extra_branch)),
    )
    bridge_change = ContingencyChange(
        5, 5, 0, "CT_TBRCH", 4, "BR_STATUS", "CT_REP", 0
    )
    table = ContingencyTable(
        table.source_path, table.sha256, table.changes + (bridge_change,)
    )
    catalog = build_contingency_catalog(case, build_network(case), table, chunk_columns=1)
    assert len(catalog.valid) == 3
    assert any(
        item.branch_source_row == 4 and item.reason == "islanding_bridge"
        for item in catalog.excluded
    )
