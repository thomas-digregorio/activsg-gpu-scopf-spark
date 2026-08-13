from pathlib import Path

import pytest

from activsg_scopf.errors import ScopeViolation
from activsg_scopf.paths import (
    assert_approved_activsg_name,
    assert_not_onedrive,
    guard_runtime_environment,
)


def test_onedrive_is_rejected_before_any_write() -> None:
    with pytest.raises(ScopeViolation, match="OneDrive"):
        assert_not_onedrive(Path("C:/Users/example/OneDrive/output.json"), purpose="test")


@pytest.mark.parametrize(
    "name", ["case_ACTIVSg501.m", "ACTIVSg2k", "ACTIVSg2001", "activsg-42.m"]
)
def test_larger_or_other_cases_are_rejected(name: str) -> None:
    with pytest.raises(
        ScopeViolation,
        match="Only ACTIVSg500, ACTIVSg2000, and ACTIVSg10k",
    ):
        assert_approved_activsg_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "case_ACTIVSg500.m",
        "case_ACTIVSg2000.m",
        "contab_ACTIVSg2000.m",
        "contab_ACTIVSg10k.m",
    ],
)
def test_explicitly_approved_scope_names_are_accepted(name: str) -> None:
    assert_approved_activsg_name(name)


def test_synced_cache_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUPY_CACHE_DIR", "C:/Users/example/OneDrive/cupy-cache")
    with pytest.raises(ScopeViolation, match="CUPY_CACHE_DIR"):
        guard_runtime_environment(tmp_path)
