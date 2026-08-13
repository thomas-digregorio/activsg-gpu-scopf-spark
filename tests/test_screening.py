import numpy as np

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.model import ModelIndex
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.screening import add_security_pairs, screen_contingencies

from .helpers import triangle_case


def test_numpy_screening_adds_all_violated_pairs_in_stable_order() -> None:
    case, table = triangle_case()
    network = build_network(case)
    catalog = build_contingency_catalog(case, network, table)
    flow = np.asarray([90.0, -90.0, 0.0])
    screened = screen_contingencies(
        flow, network, catalog, tolerance_pu=1e-5, backend="numpy"
    )
    assert list(screened.violations) == sorted(screened.violations)
    assert screened.maximum_violation_pu > 0
    canonical = CanonicalMILP()
    columns = np.asarray(
        [canonical.add_variable(f"f{i}") for i in range(3)], dtype=np.int64
    )
    index = ModelIndex(np.asarray([], dtype=np.int64), {}, {}, {}, np.asarray([]), columns)
    add_security_pairs(canonical, index, network, catalog, screened.violations)
    assert canonical.num_rows == len(screened.violations)

