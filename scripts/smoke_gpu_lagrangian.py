"""DGX-only tiny-fixture smoke for PDLP and the resident Lagrangian loop."""

from __future__ import annotations

from activsg_scopf.lagrangian import (
    RegionMasks,
    canonical_row_duals,
    evaluate_lagrangian_bound,
    optimize_lagrangian_bound_cupy,
)
from activsg_scopf.network import build_network
from activsg_scopf.reduced import build_reduced_master
from activsg_scopf.solvers.cuopt_lp import solve_cuopt_continuous_pdlp
from tests.helpers import triangle_case


def main() -> None:
    case, _ = triangle_case()
    network = build_network(case)
    master = build_reduced_master(case, network)
    solved = solve_cuopt_continuous_pdlp(
        master.canonical,
        time_limit_seconds=10.0,
        optimality_tolerance=1e-8,
        primal_feasibility_tolerance=1e-6,
        certificate_residual_tolerance=1e-7,
        native_scaling_mode="power_system_per_unit_v1",
        native_base_mva=case.base_mva,
        log_to_console=True,
        per_constraint_residual=True,
        presolve=0,
    )
    if (
        solved.native_row_dual is None
        or solved.native_primal is None
        or solved.primal_objective is None
    ):
        raise RuntimeError("Tiny PDLP solve did not return primal/dual vectors")
    dispatch_column = master.index.dispatch_by_generator[0]
    master.canonical.add_row(
        "smoke_appended_security_row", {dispatch_column: 1.0}, upper=100.0
    )
    resolved = solve_cuopt_continuous_pdlp(
        master.canonical,
        time_limit_seconds=10.0,
        optimality_tolerance=1e-8,
        primal_feasibility_tolerance=1e-6,
        certificate_residual_tolerance=1e-7,
        native_scaling_mode="power_system_per_unit_v1",
        native_base_mva=case.base_mva,
        log_to_console=True,
        per_constraint_residual=True,
        presolve=0,
        initial_native_primal=solved.native_primal,
        initial_native_row_dual=solved.native_row_dual,
    )
    if resolved.native_row_dual is None or resolved.primal_objective is None:
        raise RuntimeError("Warm-started tiny PDLP re-solve did not return vectors")
    warm_start = resolved.statistics["warm_start"]
    if warm_start.get("initial_dual_zero_extended_count") != 1:
        raise RuntimeError(f"PDLP row-dual warm start was not extended once: {warm_start}")
    row_dual = canonical_row_duals(
        master,
        resolved.native_row_dual,
        native_scaling_mode="power_system_per_unit_v1",
        base_mva=case.base_mva,
    )
    polished, gpu = optimize_lagrangian_bound_cupy(
        master,
        row_dual,
        RegionMasks.root(1),
        relaxation_primal_objective=resolved.primal_objective,
        iterations=32,
        polyak_fraction=0.5,
    )
    replay = evaluate_lagrangian_bound(
        master,
        polished,
        RegionMasks.root(1),
        safety_margin_dollars=0.01,
    )
    difference = abs(float(gpu["best_raw_lower_bound"]) - replay.raw_lower_bound)
    if difference > 1e-6:
        raise RuntimeError(f"GPU/CPU tiny-certificate replay difference is {difference}")
    print(
        {
            "status": resolved.status,
            "objective": resolved.primal_objective,
            "solved_by_pdlp": resolved.statistics["solved_by_pdlp"],
            "native_integer_columns": resolved.statistics["native_integer_columns"],
            "dual_certificate_passed": resolved.statistics["dual_certificate"]["passed"],
            "warm_start_dual_zero_extended_count": warm_start[
                "initial_dual_zero_extended_count"
            ],
            "gpu_raw_lower_bound": gpu["best_raw_lower_bound"],
            "cpu_raw_lower_bound": replay.raw_lower_bound,
            "gpu_cpu_difference": difference,
            "device_state_persistent": gpu["device_state_persistent_across_iterations"],
        }
    )


if __name__ == "__main__":
    main()
