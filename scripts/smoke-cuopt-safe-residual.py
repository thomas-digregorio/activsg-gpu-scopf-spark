#!/usr/bin/env python3
"""One-shot cuOpt smoke for safe scaling and primal/dual warm refinement."""

from __future__ import annotations

import json
import time

import numpy as np

from activsg_scopf.canonical import CanonicalMILP
from activsg_scopf.solvers.cuopt import native_scaling_vectors
from activsg_scopf.solvers.cuopt_lp import solve_cuopt_continuous_pdlp


def build_problem() -> CanonicalMILP:
    model = CanonicalMILP()
    columns = [
        model.add_variable(
            f"pg_g{index:04d}",
            objective=1.0 + index / 1000.0,
            lower=0.0,
            upper=100.0,
        )
        for index in range(128)
    ]
    model.add_row(
        "lag_balance",
        {column: 1.0 for column in columns},
        lower=6200.0,
        upper=6200.0,
    )
    return model


def main() -> int:
    model = build_problem()
    _columns, baseline_rows = native_scaling_vectors(
        model,
        mode="power_system_per_unit_v1",
        base_mva=100.0,
    )
    _columns, v2_rows = native_scaling_vectors(
        model,
        mode="power_system_equilibrated_v2",
        base_mva=100.0,
    )
    _columns, safe_rows = native_scaling_vectors(
        model,
        mode="power_system_equilibrated_safe_v3",
        base_mva=100.0,
    )
    solve_started = time.perf_counter()
    first = solve_cuopt_continuous_pdlp(
        model,
        time_limit_seconds=10.0,
        optimality_tolerance=1e-8,
        primal_feasibility_tolerance=1e-6,
        certificate_residual_tolerance=1e-7,
        native_scaling_mode="power_system_equilibrated_safe_v3",
        native_base_mva=100.0,
        log_to_console=True,
        per_constraint_residual=True,
        presolve=0,
    )
    refined = solve_cuopt_continuous_pdlp(
        model,
        time_limit_seconds=10.0,
        optimality_tolerance=1e-10,
        primal_feasibility_tolerance=1e-6,
        certificate_residual_tolerance=1e-7,
        native_scaling_mode="power_system_equilibrated_safe_v3",
        native_base_mva=100.0,
        log_to_console=True,
        per_constraint_residual=True,
        presolve=0,
        initial_native_primal=first.native_primal,
        initial_native_row_dual=first.native_row_dual,
    )
    first_residual_pu = (
        None
        if first.values is None
        else model.max_row_violation(first.values) / 100.0
    )
    refined_residual_pu = (
        None
        if refined.values is None
        else model.max_row_violation(refined.values) / 100.0
    )
    warm_start = refined.statistics.get("warm_start", {})
    passed = bool(
        first.status == "Optimal"
        and refined.status == "Optimal"
        and first.statistics.get("dual_certificate", {}).get("passed") is True
        and refined.statistics.get("dual_certificate", {}).get("passed") is True
        and first_residual_pu is not None
        and first_residual_pu <= 1e-6
        and refined_residual_pu is not None
        and refined_residual_pu <= 1e-6
        and warm_start.get("initial_primal_submitted") is True
        and warm_start.get("initial_dual_submitted") is True
        and np.all(safe_rows >= baseline_rows)
        and np.any(v2_rows < baseline_rows)
    )
    payload = {
        "schema_version": "1.0.0",
        "policy": "safe_scaling_plus_same_master_primal_dual_refinement_smoke_v1",
        "solve_count": 2,
        "first_optimality_tolerance": 1e-8,
        "refinement_optimality_tolerance": 1e-10,
        "baseline_row_scale": float(baseline_rows[0]),
        "v2_row_scale": float(v2_rows[0]),
        "safe_v3_row_scale": float(safe_rows[0]),
        "safe_v3_never_weakens_baseline": bool(np.all(safe_rows >= baseline_rows)),
        "first": {
            "status": first.status,
            "canonical_residual_pu": first_residual_pu,
            "native_solve_seconds": first.solve_time_seconds,
            "dual_certificate_passed": first.statistics.get("dual_certificate", {}).get(
                "passed"
            ),
        },
        "refinement": {
            "status": refined.status,
            "canonical_residual_pu": refined_residual_pu,
            "native_solve_seconds": refined.solve_time_seconds,
            "dual_certificate_passed": refined.statistics.get("dual_certificate", {}).get(
                "passed"
            ),
            "initial_primal_submitted": warm_start.get("initial_primal_submitted"),
            "initial_dual_submitted": warm_start.get("initial_dual_submitted"),
        },
        "adapter_wall_seconds": time.perf_counter() - solve_started,
        "passed": passed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
