"""Replay serialized GPU primal and Lagrangian certificates from raw inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activsg_scopf.config import load_config
from activsg_scopf.lagrangian_experiment import verify_lagrangian_certificate_payload
from activsg_scopf.paths import guard_input_path
from activsg_scopf.verify import verify_serialized_solution


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result_path = guard_input_path(args.result)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if "solution" in payload:
        primal = {
            "available": True,
            **verify_serialized_solution(config, payload).as_dict(),
        }
    else:
        primal = {
            "available": False,
            "passed": False,
            "reason": "serialized_result_has_no_secure_primal",
        }
    lower_bound = verify_lagrangian_certificate_payload(config, payload)
    passed = bool(primal["passed"] and lower_bound["passed"])
    print(
        json.dumps(
            {
                "passed": passed,
                "primal": primal,
                "lower_bound": lower_bound,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
