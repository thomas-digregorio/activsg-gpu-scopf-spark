"""Fail-closed scope and path guards."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

from .errors import ScopeViolation

_ACTIVSG_TOKEN = re.compile(r"activsg[_-]?(\d+)", re.IGNORECASE)


def resolved(path: str | os.PathLike[str]) -> Path:
    """Return an absolute path without requiring the target to exist."""

    return Path(path).expanduser().resolve(strict=False)


def assert_not_onedrive(path: str | os.PathLike[str], *, purpose: str = "path") -> Path:
    """Reject every path whose absolute spelling contains ``OneDrive``."""

    target = resolved(path)
    if "onedrive" in str(target).casefold():
        raise ScopeViolation(f"{purpose} is inside or names OneDrive: {target}")
    return target


def assert_activsg500_name(path_or_name: str | os.PathLike[str]) -> None:
    """Reject any explicit ACTIVSg case token other than ACTIVSg500."""

    text = str(path_or_name)
    for match in _ACTIVSG_TOKEN.finditer(text):
        if match.group(1) != "500":
            raise ScopeViolation(
                f"Only ACTIVSg500 is approved; rejected scope token {match.group(0)!r}"
            )


def guard_input_path(path: str | os.PathLike[str]) -> Path:
    """Validate an immutable input path under the local-only policy."""

    target = assert_not_onedrive(path, purpose="input path")
    assert_activsg500_name(target)
    if not target.is_file():
        raise ScopeViolation(f"Input file does not exist: {target}")
    return target


def guard_output_path(path: str | os.PathLike[str]) -> Path:
    """Validate a writable output path without creating it."""

    target = assert_not_onedrive(path, purpose="output path")
    assert_activsg500_name(target)
    assert_not_onedrive(target.parent, purpose="output parent")
    return target


def guard_runtime_environment(repo_root: str | os.PathLike[str]) -> Path:
    """Validate the repository and common Python write/cache locations."""

    root = assert_not_onedrive(repo_root, purpose="repository root")
    for key in (
        "CONDA_PREFIX",
        "TMP",
        "TEMP",
        "TMPDIR",
        "VIRTUAL_ENV",
        "PIP_CACHE_DIR",
        "PYTHONUSERBASE",
        "PYTHONPYCACHEPREFIX",
        "XDG_CACHE_HOME",
        "CUDA_CACHE_PATH",
        "CUPY_CACHE_DIR",
    ):
        value = os.environ.get(key)
        if value:
            assert_not_onedrive(value, purpose=f"environment variable {key}")
    assert_not_onedrive(sys.executable, purpose="Python executable")
    assert_not_onedrive(sys.prefix, purpose="Python environment prefix")
    assert_not_onedrive(tempfile.gettempdir(), purpose="system temporary directory")
    return root
