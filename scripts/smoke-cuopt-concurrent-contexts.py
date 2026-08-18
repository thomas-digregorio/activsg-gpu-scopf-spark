#!/usr/bin/env python3
"""One-shot safety smoke for two independent concurrent cuOpt PDLP contexts."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.solvers.cuopt_lp import solve_cuopt_continuous_pdlp


def build_problem(label: str, lower_bound: float) -> CanonicalMILP:
    model = CanonicalMILP()
    columns = [
        model.add_variable(f"pg_{label}_{index:05d}", objective=1.0, lower=0.0, upper=100.0)
        for index in range(4096)
    ]
    for index, column in enumerate(columns):
        model.add_row(f"lower_{label}_{index:05d}", {column: 1.0}, lower=lower_bound)
        model.add_row(f"upper_{label}_{index:05d}", {column: 1.0}, upper=75.0)
    return model


def solve_one(label: str, lower_bound: float) -> dict[str, Any]:
    started = time.perf_counter()
    solved = solve_cuopt_continuous_pdlp(
        build_problem(label, lower_bound),
        time_limit_seconds=10.0,
        optimality_tolerance=1e-8,
        primal_feasibility_tolerance=1e-6,
        certificate_residual_tolerance=1e-7,
        native_scaling_mode="power_system_equilibrated_v2",
        native_base_mva=100.0,
        log_to_console=True,
        per_constraint_residual=True,
        presolve=0,
    )
    expected = 4096.0 * lower_bound
    objective_error = (
        None if solved.primal_objective is None else abs(float(solved.primal_objective) - expected)
    )
    passed = bool(
        solved.statistics.get("error_status") == "Success"
        and solved.statistics.get("solved_by_pdlp") is True
        and solved.values is not None
        and solved.native_row_dual is not None
        and solved.statistics.get("native_log_branch_and_bound_markers_absent") is True
        and solved.statistics.get("dual_certificate", {}).get("primal_feasible") is True
        and objective_error is not None
        and objective_error <= 1e-2
    )
    return {
        "label": label,
        "passed": passed,
        "status": solved.status,
        "error_status": solved.statistics.get("error_status"),
        "solved_by": solved.statistics.get("solved_by"),
        "solved_by_pdlp": solved.statistics.get("solved_by_pdlp"),
        "objective": solved.primal_objective,
        "expected_objective": expected,
        "objective_error": objective_error,
        "native_solve_seconds": solved.solve_time_seconds,
        "adapter_wall_seconds": time.perf_counter() - started,
        "dual_certificate_passed": solved.statistics.get("dual_certificate", {}).get("passed"),
        "primal_feasible": solved.statistics.get("dual_certificate", {}).get("primal_feasible"),
    }


def main() -> int:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cuopt-smoke") as pool:
        futures = [
            pool.submit(solve_one, "a", 25.0),
            pool.submit(solve_one, "b", 35.0),
        ]
        results = [future.result() for future in futures]
    wall = time.perf_counter() - started
    payload = {
        "schema_version": "1.0.0",
        "policy": "two_independent_threaded_cuopt_pdlp_contexts_one_shot_v1",
        "context_count": 2,
        "repetitions_per_context": 1,
        "parallel_wall_seconds": wall,
        "sum_adapter_wall_seconds": sum(
            float(record["adapter_wall_seconds"]) for record in results
        ),
        "contexts": results,
        "passed": all(bool(record["passed"]) for record in results),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
