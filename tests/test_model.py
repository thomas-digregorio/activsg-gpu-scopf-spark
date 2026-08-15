import numpy as np

from activsg_scopf.model import build_master
from activsg_scopf.network import build_network

from .helpers import triangle_case


def test_master_uses_exact_conditional_pmin_and_source_offline_unavailable() -> None:
    case, _ = triangle_case()
    master = build_master(case, build_network(case))
    assert set(master.index.commitment_by_generator) == {0}
    u = master.index.commitment_by_generator[0]
    pg = master.index.dispatch_by_generator[0]
    row = master.canonical.row_names.index("exact_pmin_dispatch_g0001")
    matrix = master.canonical.matrix_csr()
    assert matrix[row, pg] == 1
    assert matrix[row, u] == -25
    assert len(master.index.segments_by_generator[0]) == 10
    assert np.sum(master.canonical.integrality) == 1


def test_master_includes_pd_plus_gs_balance_and_dc_flow_rows() -> None:
    case, _ = triangle_case()
    master = build_master(case, build_network(case))
    row = master.canonical.row_names.index("nodal_balance_b0002")
    assert master.canonical.row_lower[row] == 42
    assert master.canonical.row_upper[row] == 42
    assert sum(name.startswith("dc_flow_l") for name in master.canonical.row_names) == 3


def test_fixed_output_generator_keeps_ten_zero_segments_without_zero_columns() -> None:
    case, _ = triangle_case()
    case.gen[0, 8] = case.gen[0, 9]
    master = build_master(case, build_network(case))
    segments = master.index.segments_by_generator[0]
    assert len(segments) == 10
    assert all(column is None for column in segments)
    assert not any(name.startswith("pseg_g0001") for name in master.canonical.variable_names)
