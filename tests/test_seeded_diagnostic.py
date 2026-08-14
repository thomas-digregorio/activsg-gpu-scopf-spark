import numpy as np
import pytest

from activsg_scopf.errors import ProvenanceError
from activsg_scopf.model import build_master
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.screening import screen_contingencies
from activsg_scopf.seeded_diagnostic import (
    canonical_feasibility_audit,
    deserialize_solution_values,
    security_pairs_from_ids,
)
from activsg_scopf.solution import serialize_solution

from .helpers import triangle_case


def test_serialized_solution_reconstructs_every_canonical_column() -> None:
    case, _ = triangle_case()
    network = build_network(case)
    master = build_master(case, network)
    values = np.arange(master.canonical.num_columns, dtype=np.float64) / 10.0
    solution = serialize_solution(values, case, network, master)

    reconstructed = deserialize_solution_values(solution, case, network, master)

    np.testing.assert_array_equal(reconstructed, values)


def test_security_pair_ids_reconstruct_exact_screened_rows() -> None:
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    screened = screen_contingencies(
        np.asarray([90.0, -90.0, 0.0]),
        network,
        catalog,
        tolerance_pu=1e-5,
        backend="numpy",
    )
    pair_ids = [pair.pair_id for pair in screened.violations]

    reconstructed = security_pairs_from_ids(pair_ids, network, catalog)

    assert [pair.pair_id for pair in reconstructed] == pair_ids
    np.testing.assert_allclose(
        [pair.lodf_value for pair in reconstructed],
        [pair.lodf_value for pair in screened.violations],
    )


def test_security_pair_reconstruction_rejects_noncanonical_order() -> None:
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)

    with pytest.raises(ProvenanceError, match="canonically ordered"):
        security_pairs_from_ids(
            ["c0002_m0001_upper", "c0001_m0002_upper"], network, catalog
        )


def test_canonical_audit_names_the_largest_violation() -> None:
    case, _ = triangle_case()
    network = build_network(case)
    master = build_master(case, network)
    values = np.zeros(master.canonical.num_columns)
    objective = master.canonical.column_arrays()[0]

    audit = canonical_feasibility_audit(
        master.canonical, values, expected_objective=float(objective @ values)
    )

    assert audit["maximum_row_violation"] > 0
    assert str(audit["maximum_row_violation_name"]).startswith("nodal_balance")
