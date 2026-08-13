import numpy as np

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

