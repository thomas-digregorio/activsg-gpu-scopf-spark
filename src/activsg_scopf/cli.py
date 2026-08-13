"""Versioned command-line surface for ingest, solve, verify, and benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import load_config
from .errors import ScopfError
from .matpower import read_contingency_table, read_matpower_case
from .paths import guard_output_path, guard_runtime_environment
from .provenance import build_source_manifest, write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="activsg-scopf")
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "solve", "verify", "benchmark"):
        command = subcommands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name in {"solve", "benchmark"}:
            command.add_argument(
                "--platform", choices=("laptop_cpu", "dgx_spark"), required=True
            )
        if name == "verify":
            command.add_argument("--solution", type=Path, required=True)
    return parser


def _ingest(config_path: Path, output_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    guard_runtime_environment(config.root)
    case = read_matpower_case(
        config.case_path, expected_sha256=config.raw["raw_inputs"]["case_sha256"]
    )
    contingencies = read_contingency_table(
        config.contingency_path,
        expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
    )
    output = guard_output_path(output_path)
    payload = build_source_manifest(case, contingencies)
    write_json_atomic(payload, output)
    return {"status": "ok", "command": "ingest", "output": str(output)}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "ingest":
            response = _ingest(args.config, args.output)
        else:
            raise ScopfError(f"{args.command} is reserved for the next implementation milestone")
    except (ScopfError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(response, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
