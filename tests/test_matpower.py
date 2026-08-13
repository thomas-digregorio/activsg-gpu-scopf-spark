from pathlib import Path

from activsg_scopf.matpower import (
    GEN_STATUS,
    PMIN,
    read_contingency_table,
    read_matpower_case,
    sha256_file,
)
from activsg_scopf.provenance import build_source_manifest

from .helpers import write_triangle_matpower


def test_tiny_matpower_sources_preserve_rows_and_exact_pmin(tmp_path: Path) -> None:
    case_path, contingency_path = write_triangle_matpower(tmp_path)
    case = read_matpower_case(case_path, expected_sha256=sha256_file(case_path))
    table = read_contingency_table(
        contingency_path, expected_sha256=sha256_file(contingency_path)
    )
    assert case.bus.shape[0] == 3
    assert case.gen.shape[0] == 2
    assert case.branch.shape[0] == 3
    assert sum(case.gen[:, GEN_STATUS] > 0) == 1
    assert case.gen[0, PMIN] == 25
    assert len(table.changes) == 4
    assert sum(change.table == "CT_TBRCH" for change in table.changes) == 3
    assert sum(change.table == "CT_TGEN" for change in table.changes) == 1


def test_manifest_preserves_every_tiny_generator_source_row(tmp_path: Path) -> None:
    case_path, contingency_path = write_triangle_matpower(tmp_path)
    case = read_matpower_case(case_path, expected_sha256=sha256_file(case_path))
    table = read_contingency_table(
        contingency_path, expected_sha256=sha256_file(contingency_path)
    )
    manifest = build_source_manifest(case, table)
    assert len(manifest["generators"]) == 2
    assert manifest["generators"][0]["source_id"] == "gen-row-0001"
    assert manifest["generators"][0]["pmin_mw"] == 25
    assert manifest["source_online_capacity_mw"]["pmin_sum"] == 25

