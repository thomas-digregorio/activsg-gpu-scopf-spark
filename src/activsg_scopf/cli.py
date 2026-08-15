"""Versioned command-line surface for ingest, solve, verify, and benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import load_config
from .errors import DeadlineExceeded, ScopfError
from .experiments import run_one_shot_gap_experiment
from .lp_certificate import (
    run_lp_certificate_worker_serialized,
    run_one_shot_lp_certificate,
)
from .matpower import read_contingency_table, read_matpower_case
from .official import run_controlled
from .paths import guard_input_path, guard_output_path, guard_runtime_environment
from .provenance import build_source_manifest, write_json_atomic
from .runner import run_end_to_end
from .seeded_diagnostic import (
    run_one_shot_seeded_round2_diagnostic,
    run_seeded_round2_worker_serialized,
)
from .verify import verify_serialized_solution


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="activsg-scopf")
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "ingest",
        "solve",
        "verify",
        "benchmark",
        "gap-experiment",
        "lp-certificate",
        "seeded-round2-diagnostic",
    ):
        command = subcommands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name in {"solve", "benchmark"}:
            command.add_argument(
                "--platform", choices=("laptop_cpu", "dgx_spark"), required=True
            )
        if name == "verify":
            command.add_argument("--solution", type=Path, required=True)
        if name == "benchmark":
            command.add_argument("--laptop-result", type=Path)
    return parser


def _worker_parser() -> argparse.ArgumentParser:
    worker = argparse.ArgumentParser(prog="activsg-scopf _worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--platform", choices=("laptop_cpu", "dgx_spark"), required=True)
    worker.add_argument("--deadline-seconds", type=float)
    worker.add_argument("--official", action="store_true")
    return worker


def _seeded_worker_parser() -> argparse.ArgumentParser:
    worker = argparse.ArgumentParser(prog="activsg-scopf _seeded_round2_worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    return worker


def _lp_certificate_worker_parser() -> argparse.ArgumentParser:
    worker = argparse.ArgumentParser(prog="activsg-scopf _lp_certificate_worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    return worker


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


def _read_json(path: Path) -> dict[str, object]:
    source = guard_input_path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def _worker(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
    output = guard_output_path(args.output)
    checkpoint_path = guard_output_path(args.checkpoint)

    def checkpoint(payload: dict[str, object]) -> None:
        write_json_atomic(payload, checkpoint_path)

    try:
        result = run_end_to_end(
            config,
            platform_name=args.platform,
            deadline_seconds=args.deadline_seconds,
            official=args.official,
            checkpoint=checkpoint,
        )
    except DeadlineExceeded as exc:
        result = _read_json(checkpoint_path) if checkpoint_path.exists() else {}
        result.update(
            {
                "status": "deadline_budget_exhausted",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    except Exception as exc:  # worker must serialize any partial failure once
        result = _read_json(checkpoint_path) if checkpoint_path.exists() else {}
        result.update(
            {
                "status": "failed_exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    serialization_started = time.perf_counter()
    write_json_atomic(result, output)
    result.setdefault("timings_seconds", {})["result_serialization"] = (
        time.perf_counter() - serialization_started
    )
    write_json_atomic(result, output)
    return {"status": str(result.get("status")), "output": str(output)}


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(argv) if argv is not None else sys.argv[1:]
    if raw_arguments and raw_arguments[0] == "_worker":
        args = _worker_parser().parse_args(raw_arguments[1:])
        args.command = "_worker"
    elif raw_arguments and raw_arguments[0] == "_seeded_round2_worker":
        args = _seeded_worker_parser().parse_args(raw_arguments[1:])
        args.command = "_seeded_round2_worker"
    elif raw_arguments and raw_arguments[0] == "_lp_certificate_worker":
        args = _lp_certificate_worker_parser().parse_args(raw_arguments[1:])
        args.command = "_lp_certificate_worker"
    else:
        args = _parser().parse_args(raw_arguments)
    try:
        if args.command == "ingest":
            response = _ingest(args.config, args.output)
        elif args.command == "_worker":
            response = _worker(args)
        elif args.command == "_seeded_round2_worker":
            config = load_config(args.config)
            result = run_seeded_round2_worker_serialized(
                config,
                output_path=args.output,
                checkpoint_path=args.checkpoint,
            )
            response = {"status": result["status"], "output": str(args.output)}
        elif args.command == "_lp_certificate_worker":
            config = load_config(args.config)
            result = run_lp_certificate_worker_serialized(
                config,
                output_path=args.output,
                checkpoint_path=args.checkpoint,
            )
            response = {"status": result["status"], "output": str(args.output)}
        elif args.command == "solve":
            config = load_config(args.config)
            result = run_controlled(
                config,
                platform_name=args.platform,
                output_path=args.output,
                official=False,
            )
            response = {"status": result["status"], "output": str(args.output)}
        elif args.command == "verify":
            config = load_config(args.config)
            result_payload = _read_json(args.solution)
            verification = verify_serialized_solution(config, result_payload)
            output = guard_output_path(args.output)
            write_json_atomic(verification.as_dict(), output)
            response = {
                "status": "verified" if verification.passed else "failed_verification",
                "output": str(output),
            }
        elif args.command == "benchmark":
            config = load_config(args.config)
            result = run_controlled(
                config,
                platform_name=args.platform,
                output_path=args.output,
                official=True,
                laptop_result=args.laptop_result,
            )
            response = {"status": result["status"], "output": str(args.output)}
        elif args.command == "gap-experiment":
            config = load_config(args.config)
            result = run_one_shot_gap_experiment(config, output_path=args.output)
            response = {"status": result["status"], "output": str(args.output)}
        elif args.command == "lp-certificate":
            config = load_config(args.config)
            result = run_one_shot_lp_certificate(config, output_path=args.output)
            response = {"status": result["status"], "output": str(args.output)}
        elif args.command == "seeded-round2-diagnostic":
            config = load_config(args.config)
            result = run_one_shot_seeded_round2_diagnostic(
                config, output_path=args.output
            )
            response = {"status": result["status"], "output": str(args.output)}
        else:
            raise ScopfError(f"Unknown command {args.command}")
    except (ScopfError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(response, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
