from dataclasses import replace
from pathlib import Path

import numpy as np

from activsg_scopf.contingency_mapping import (
    BRANCH_MAPPING_METHOD,
    map_reference_branch_contingencies,
)
from activsg_scopf.matpower import (
    GEN_STATUS,
    PMIN,
    enumerate_in_service_branch_contingencies,
    read_contingency_table,
    read_matpower_case,
    sha256_file,
)
from activsg_scopf.provenance import build_source_manifest

from .helpers import triangle_case, write_triangle_matpower


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


def test_series24_cp1252_case_is_parsed_without_rewriting_source(tmp_path: Path) -> None:
    original_path, _ = write_triangle_matpower(tmp_path)
    series24_path = original_path.with_name(
        "Texas2k_series24_case1_2016summerPeak.m"
    )
    source = "% “Series24 source comment”\n" + original_path.read_text(
        encoding="utf-8"
    )
    series24_path.write_bytes(source.encode("cp1252"))
    case = read_matpower_case(
        series24_path,
        expected_sha256=sha256_file(series24_path),
    )
    assert case.case_name == "Texas2kSeries24Case1"
    assert case.gen[0, PMIN] == 25


def test_topology_derived_contingencies_preserve_every_branch_row() -> None:
    case, _ = triangle_case()
    table = enumerate_in_service_branch_contingencies(case)
    assert table.mode == "enumerate_in_service_branches"
    assert table.source_path is None
    assert table.sha256 is None
    assert [change.element_row for change in table.changes] == [1, 2, 3]
    assert [change.source_row for change in table.changes] == [1, 2, 3]


def test_reference_contingencies_map_by_identity_not_source_row() -> None:
    reference_case, reference_table = triangle_case()
    extra_parallel = reference_case.branch[0].copy()
    extra_parallel[3] = 0.3
    target_case = replace(
        reference_case,
        case_name="Texas2kSeries24Case1",
        source_path=Path("Texas2k_series24_case1_2016summerPeak.m"),
        sha256="target-fixture",
        branch=np.stack(
            (
                reference_case.branch[2],
                extra_parallel,
                reference_case.branch[1],
                reference_case.branch[0],
            )
        ),
    )
    mapped = map_reference_branch_contingencies(
        reference_case,
        target_case,
        reference_table,
        method=BRANCH_MAPPING_METHOD,
    )
    assert mapped.mode == "mapped_reference_branch_table"
    assert [change.element_row for change in mapped.changes[:3]] == [4, 3, 1]
    assert mapped.changes[3].table == "CT_TGEN"
    assert mapped.changes[3].element_row == 1
    assert mapped.derivation is not None
    assert mapped.derivation["mapped_unique_reference_branches"] == 3
    assert mapped.derivation["exact_parameter_matches"] == 3
    assert mapped.derivation["target_only_branches_not_outage_candidates"] == 1
    assert mapped.derivation["target_only_branch_records"] == [
        {
            "target_branch_source_row": 2,
            "from_bus": 1,
            "to_bus": 2,
        }
    ]
    assert len(mapped.derivation["branch_mapping_records_sha256"]) == 64
