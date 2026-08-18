import tomllib
from pathlib import Path

from activsg_scopf import __version__


def test_runtime_and_package_versions_match() -> None:
    root = Path(__file__).resolve().parents[1]
    package = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == package["project"]["version"]
