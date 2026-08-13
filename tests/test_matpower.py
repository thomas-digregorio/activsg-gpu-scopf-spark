from pathlib import Path

import numpy as np

from activsg_scopf.matpower import (
    EXPECTED_CASE_SHA256,
    EXPECTED_CONTINGENCY_SHA256,
    GEN_STATUS,
    PMIN,
    read_contingency_table,
    read_matpower_case,
)
from activsg_scopf.provenance import build_source_manifest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "matpower-8.1"


def test_registered_activsg500_sources_and_exact_pmin_are_read() -> None:
    case = read_matpower_case(RAW / "case_ACTIVSg500.m")
    table = read_contingency_table(RAW / "contab_ACTIVSg500.m")
    assert case.sha256 == EXPECTED_CASE_SHA256
    assert table.sha256 == EXPECTED_CONTINGENCY_SHA256
    assert case.bus.shape[0] == 500
    assert case.gen.shape[0] == 90
    assert case.branch.shape[0] == 597
    assert np.count_nonzero(case.gen[:, GEN_STATUS] > 0) == 56
    assert case.gen[0, PMIN] == 231.54
    assert len(table.changes) == 681
    assert sum(change.table == "CT_TBRCH" for change in table.changes) == 591
    assert sum(change.table == "CT_TGEN" for change in table.changes) == 90


def test_manifest_preserves_every_generator_source_row() -> None:
    case = read_matpower_case(RAW / "case_ACTIVSg500.m")
    table = read_contingency_table(RAW / "contab_ACTIVSg500.m")
    manifest = build_source_manifest(case, table)
    assert len(manifest["generators"]) == 90
    assert manifest["generators"][0]["source_id"] == "gen-row-0001"
    assert manifest["generators"][0]["pmin_mw"] == 231.54
    assert manifest["source_online_capacity_mw"]["pmin_sum"] == 2659.06

