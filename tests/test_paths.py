from pathlib import Path

import pytest

from activsg_scopf.errors import ScopeViolation
from activsg_scopf.paths import assert_activsg500_name, assert_not_onedrive


def test_onedrive_is_rejected_before_any_write() -> None:
    with pytest.raises(ScopeViolation, match="OneDrive"):
        assert_not_onedrive(Path("C:/Users/example/OneDrive/output.json"), purpose="test")


@pytest.mark.parametrize("name", ["case_ACTIVSg2000.m", "ACTIVSg10k", "activsg-200.m"])
def test_larger_or_other_cases_are_rejected(name: str) -> None:
    with pytest.raises(ScopeViolation, match="Only ACTIVSg500"):
        assert_activsg500_name(name)


def test_activsg500_scope_name_is_accepted() -> None:
    assert_activsg500_name("case_ACTIVSg500.m")

