"""One-shot ACTIVSg500 GPU Lagrangian/disjunctive experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig
from .costs import pwl_approximation_report
from .deadline import Deadline, PeakMemorySampler
from .environment import environment_manifest, validate_platform
from .errors import (
    DeadlineExceeded,
    PrimalCandidateRejected,
    ProvenanceError,
    ScopfError,
)
from .lagrangian import (
    LagrangianEvaluation,
    RegionMasks,
    bus_prices_from_coupling_duals,
    canonical_row_duals,
    choose_split_generator,
    evaluate_lagrangian_bound,
    optimize_lagrangian_bound_cupy,
    replay_lagrangian_certificate,
    verify_disjunctive_cover,
)
from .matpower import (
    GEN_STATUS,
    PMAX,
    PMIN,
    read_contingency_table,
    read_matpower_case,
)
from .network import (
    ContingencyCatalog,
    NetworkData,
    build_contingency_catalog,
    build_network,
    contingency_catalog_report,
)
from .official import frozen_identity
from .paths import guard_output_path, guard_runtime_environment
from .phase_one import (
    build_phase_one_model,
    phase_one_certificate,
    replay_phase_one_certificate,
)
from .provenance import build_source_manifest, write_json_atomic
from .reduced import (
    ReducedMaster,
    add_reduced_security_pairs,
    build_reduced_master,
    commitment_vector,
    fix_commitments,
    reconstruct_full_values,
    reduced_dispatch,
    security_pair_from_record,
    security_pair_record,
)
from .screening import ContingencyScreener, SecurityPair
from .solution import serialize_solution
from .solvers.cuopt import native_scaling_vectors
from .solvers.cuopt_lp import ContinuousSolveResult, solve_cuopt_continuous_pdlp
from .verify import verify_serialized_solution

EXPERIMENT_ID = "activsg500-gpu-lagrangian-v1"
EXPERIMENT_TAG = "experiment-500-gpu-lagrangian-v1"
REGISTERED_EXPERIMENTS = {
    EXPERIMENT_ID: {
        "tag": EXPERIMENT_TAG,
        "policy": "gpu_pdlp_primal_plus_replayable_lagrangian_cover_v1",
    },
    "activsg500-gpu-lagrangian-v2": {
        "tag": "experiment-500-gpu-lagrangian-v2",
        "policy": "gpu_pdlp_primal_plus_replayable_lagrangian_cover_v2",
    },
    "activsg500-gpu-lagrangian-v3": {
        "tag": "experiment-500-gpu-lagrangian-v3",
        "policy": "gpu_pdlp_bounded_candidate_queue_plus_lagrangian_cover_v3",
    },
    "activsg500-gpu-lagrangian-v4": {
        "tag": "experiment-500-gpu-lagrangian-v4",
        "policy": "gpu_pdlp_phase_one_pruning_plus_lagrangian_cover_v4",
    },
    "activsg500-gpu-lagrangian-v5": {
        "tag": "experiment-500-gpu-lagrangian-v5",
        "policy": "gpu_pdlp_phase_one_pruning_plus_lagrangian_cover_v5",
    },
    "activsg500-gpu-lagrangian-v6": {
        "tag": "experiment-500-gpu-lagrangian-v6",
        "policy": "gpu_pdlp_phase_one_first_plus_lagrangian_cover_v6",
    },
}
V2_BUGFIX_CHANGE = {
    "comparison_baseline": "activsg500-gpu-lagrangian-v1",
    "coefficient_cleanup": (
        "drop_affine_dispatch_coefficients_abs_le_1e-14_with_box_rhs_relaxation"
    ),
    "pdlp_primal_gate": "never_screen_or_add_rows_from_primal_infeasible_vector",
    "continuation": ("warm_start_same_master_after_time_limit_until_feasible_or_global_deadline"),
}
V3_CONTROLLER_CHANGE = {
    "comparison_baseline": "activsg500-gpu-lagrangian-v2",
    "root_certificate_persistence": "checkpoint_before_any_primal_repair",
    "candidate_budget": "15_seconds_total_with_5_second_pdlp_slices",
    "candidate_gate": "reject_dual_divergence_or_two_round_residual_stagnation",
    "candidate_queue": (
        "root_pdlp_rounding_then_gpu_lagrangian_then_low_threshold_then_all_online"
    ),
    "global_deadline": "never_convert_deadline_exceeded_into_candidate_rejection",
}
V4_CONTROLLER_CHANGE = {
    "comparison_baseline": "activsg500-gpu-lagrangian-v3",
    "security_row_deduplication": (
        "exact_post_cleanup_fp64_solver_rows_with_all_source_pair_ids_retained"
    ),
    "region_attempts": "bounded_warm_attempt_then_one_cold_restart",
    "failed_split_policy": "rollback_transaction_and_try_next_deterministic_generator",
    "infeasible_leaf_gate": "replayable_gpu_phase_one_box_dual_certificate",
    "secure_incumbent": "serialize_and_independently_verify_before_bound_refinement",
    "portable_lodf_replay": "absolute_tolerance_1e-12",
}
V5_BUGFIX_CHANGE = {
    "comparison_baseline": "activsg500-gpu-lagrangian-v4",
    "phase_one_row_identity": "semantic_source_row_and_side_order_independent_v1",
    "legacy_phase_one_replay": "ordered_hash_then_semantic_alignment",
    "cleanup_audit_replay": "exact_proof_invariants_with_portable_fp64_dust_telemetry",
    "gap_bookkeeping": "refresh_at_every_frontier_checkpoint",
}
V6_CONTROLLER_CHANGE = {
    "comparison_baseline": "activsg500-gpu-lagrangian-v5",
    "child_order": "exact_pmin_pmax_capacity_gate_then_short_gpu_phase_one_then_cost_lp",
    "capacity_gate": "aggregate_exact_source_pmin_pmax_interval_telemetry_only",
    "phase_one_precheck_budget": "2_seconds_per_child_within_600_second_global_deadline",
    "prune_authority": "positive_independently_replayable_phase_one_dual_only",
    "feasible_child_handoff": "phase_one_source_native_primal_to_cost_lp_without_row_dual",
    "uncertain_child_fallback": "ordinary_cost_lp_then_full_phase_one_if_cost_attempt_rejects",
}


@dataclass
class SolvedRegion:
    region_id: str
    masks: RegionMasks
    master: ReducedMaster
    solve: ContinuousSolveResult
    canonical_row_dual: np.ndarray
    lagrangian: LagrangianEvaluation
    commitment: np.ndarray
    security_pairs: tuple[SecurityPair, ...]
    rounds: list[dict[str, Any]]
    final_screen: dict[str, Any]
    gpu_lagrangian: dict[str, Any]


@dataclass(frozen=True)
class PrimalCandidatePolicy:
    total_seconds: float
    maximum_round_seconds: float
    minimum_round_seconds: float
    stagnation_window_rounds: int
    minimum_relative_residual_improvement: float
    dual_divergence_multiple: float
    cold_restart_attempts: int

    @classmethod
    def from_config(
        cls, config: RunConfig, *, scope: str = "primal_candidate"
    ) -> PrimalCandidatePolicy:
        runtime = config.runtime
        if scope not in {"primal_candidate", "disjunctive_region"}:
            raise ScopfError(f"Unknown bounded PDLP policy scope: {scope}")
        prefix = "primal_candidate" if scope == "primal_candidate" else "region_attempt"
        return cls(
            total_seconds=float(runtime[f"maximum_{prefix}_seconds"]),
            maximum_round_seconds=float(runtime[f"maximum_{prefix}_round_seconds"]),
            minimum_round_seconds=float(runtime[f"minimum_{prefix}_round_seconds"]),
            stagnation_window_rounds=int(runtime[f"{prefix}_stagnation_window_rounds"]),
            minimum_relative_residual_improvement=float(
                runtime[f"{prefix}_minimum_relative_residual_improvement"]
            ),
            dual_divergence_multiple=float(runtime[f"{prefix}_dual_divergence_multiple"]),
            cold_restart_attempts=int(runtime.get(f"{prefix}_cold_restart_attempts", 0)),
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_seconds": self.total_seconds,
            "maximum_round_seconds": self.maximum_round_seconds,
            "minimum_round_seconds": self.minimum_round_seconds,
            "stagnation_window_rounds": self.stagnation_window_rounds,
            "minimum_relative_residual_improvement": (self.minimum_relative_residual_improvement),
            "dual_divergence_multiple": self.dual_divergence_multiple,
            "cold_restart_attempts": self.cold_restart_attempts,
        }


class RegionAttemptRejected(PrimalCandidateRejected):
    """Bounded PDLP attempt rejected with its exact append-only model retained."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        master: ReducedMaster,
        security_pairs: tuple[SecurityPair, ...],
        rounds: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.master = master
        self.security_pairs = security_pairs
        self.rounds = rounds


@dataclass
class PhaseOneAttemptResult:
    record: dict[str, Any]
    source_native_primal: np.ndarray | None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read experiment evidence {path}: {exc}") from exc


def _write_console(path: Path, stdout: str | bytes | None, stderr: str | bytes | None) -> None:
    def decode(value: str | bytes | None) -> str:
        return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(decode(stdout) + decode(stderr), encoding="utf-8")


def validate_lagrangian_experiment_config(config: RunConfig) -> dict[str, Any]:
    if config.case_name != "ACTIVSg500":
        raise ScopfError("The first GPU Lagrangian experiment is ACTIVSg500-only")
    benchmark = config.raw["benchmark"]
    experiment = REGISTERED_EXPERIMENTS.get(config.benchmark_id)
    if experiment is None:
        raise ScopfError(f"Unregistered GPU Lagrangian experiment: {config.benchmark_id!r}")
    if benchmark.get("kind") != "gpu_lagrangian_disjunctive_experiment":
        raise ScopfError("GPU Lagrangian experiment kind changed")
    if benchmark.get("required_git_tag") != experiment["tag"]:
        raise ScopfError("GPU Lagrangian frozen tag changed")
    if (
        config.benchmark_id.endswith(("-v2", "-v3", "-v4", "-v5", "-v6"))
        and float(config.model.get("reduced_coefficient_zero_tolerance", -1.0)) != 1e-14
    ):
        raise ScopfError("GPU Lagrangian coefficient threshold changed")
    if config.benchmark_id.endswith("-v2"):
        observed_change = benchmark.get("bugfix_change")
        if observed_change != V2_BUGFIX_CHANGE:
            raise ScopfError(
                "GPU Lagrangian v2 bugfix identity changed: "
                f"expected={V2_BUGFIX_CHANGE}, observed={observed_change}"
            )
    if config.benchmark_id.endswith("-v3"):
        observed_change = benchmark.get("controller_change")
        if observed_change != V3_CONTROLLER_CHANGE:
            raise ScopfError(
                "GPU Lagrangian v3 controller identity changed: "
                f"expected={V3_CONTROLLER_CHANGE}, observed={observed_change}"
            )
    if config.benchmark_id.endswith("-v4"):
        observed_change = benchmark.get("controller_change")
        if observed_change != V4_CONTROLLER_CHANGE:
            raise ScopfError(
                "GPU Lagrangian v4 controller identity changed: "
                f"expected={V4_CONTROLLER_CHANGE}, observed={observed_change}"
            )
    if config.benchmark_id.endswith("-v5"):
        observed_change = benchmark.get("bugfix_change")
        if observed_change != V5_BUGFIX_CHANGE:
            raise ScopfError(
                "GPU Lagrangian v5 bugfix identity changed: "
                f"expected={V5_BUGFIX_CHANGE}, observed={observed_change}"
            )
    if config.benchmark_id.endswith("-v6"):
        observed_change = benchmark.get("controller_change")
        if observed_change != V6_CONTROLLER_CHANGE:
            raise ScopfError(
                "GPU Lagrangian v6 controller identity changed: "
                f"expected={V6_CONTROLLER_CHANGE}, observed={observed_change}"
            )
    profile = config.raw["platforms"].get("dgx_spark", {})
    required_profile = {
        "solver": "cuopt",
        "screening": "cupy",
        "lp_method": "pdlp",
        "pdlp_precision": "fp64",
        "native_scaling_mode": "power_system_per_unit_v1",
        "presolve": 0,
        "lagrangian_gpu_iterations": 512,
        "lagrangian_polyak_fraction": 0.5,
        "console_logging": True,
        "integer_solver": "none",
        "branch_and_bound": False,
        "custom_cuda_kernels": "allowed_but_not_required",
    }
    observed = {key: profile.get(key) for key in required_profile}
    if observed != required_profile:
        raise ScopfError(
            f"GPU Lagrangian profile changed: expected={required_profile}, observed={observed}"
        )
    runtime = config.runtime
    if float(runtime.get("deadline_seconds", 0.0)) != 600.0:
        raise ScopfError("The registered GPU Lagrangian experiment has one 600-second run")
    if int(runtime.get("maximum_frontier_regions", 0)) < 1:
        raise ScopfError("maximum_frontier_regions must be positive")
    if float(config.model["mip_relative_gap_tolerance"]) != 1e-3:
        raise ScopfError("The registered Lagrangian target is exactly 1e-3")
    if config.benchmark_id.endswith("-v3"):
        required_runtime = {
            "maximum_primal_candidate_seconds": 15.0,
            "maximum_primal_candidate_round_seconds": 5.0,
            "minimum_primal_candidate_round_seconds": 0.25,
            "primal_candidate_stagnation_window_rounds": 2,
            "primal_candidate_minimum_relative_residual_improvement": 0.01,
            "primal_candidate_dual_divergence_multiple": 1e6,
        }
        observed_runtime = {key: runtime.get(key) for key in required_runtime}
        if observed_runtime != required_runtime:
            raise ScopfError(
                "GPU Lagrangian v3 candidate policy changed: "
                f"expected={required_runtime}, observed={observed_runtime}"
            )
    if config.benchmark_id.endswith(("-v4", "-v5", "-v6")):
        required_runtime = {
            "maximum_primal_candidate_seconds": 15.0,
            "maximum_primal_candidate_round_seconds": 5.0,
            "minimum_primal_candidate_round_seconds": 0.25,
            "primal_candidate_stagnation_window_rounds": 2,
            "primal_candidate_minimum_relative_residual_improvement": 0.01,
            "primal_candidate_dual_divergence_multiple": 1e6,
            "primal_candidate_cold_restart_attempts": 1,
            "maximum_region_attempt_seconds": 15.0,
            "maximum_region_attempt_round_seconds": 5.0,
            "minimum_region_attempt_round_seconds": 0.25,
            "region_attempt_stagnation_window_rounds": 2,
            "region_attempt_minimum_relative_residual_improvement": 0.01,
            "region_attempt_dual_divergence_multiple": 1e6,
            "region_attempt_cold_restart_attempts": 1,
            "maximum_failed_split_attempts": 8,
            "phase_one_time_limit_seconds": 15.0,
            "phase_one_maximum_violation_pu": 1e6,
            "phase_one_safety_margin_pu": 1e-8,
            "phase_one_infeasibility_threshold_pu": 1e-6,
        }
        observed_runtime = {key: runtime.get(key) for key in required_runtime}
        if observed_runtime != required_runtime:
            raise ScopfError(
                "GPU Lagrangian v4/v5/v6 bounded-region policy changed: "
                f"expected={required_runtime}, observed={observed_runtime}"
            )
        if config.benchmark_id.endswith("-v6") and float(
            runtime.get("precheck_phase_one_time_limit_seconds", -1.0)
        ) != 2.0:
            raise ScopfError("GPU Lagrangian v6 short Phase-I budget changed")
        if float(config.model.get("serialized_lodf_replay_tolerance", -1.0)) != 1e-12:
            raise ScopfError("GPU Lagrangian v4/v5/v6 LODF replay tolerance changed")
        if (
            float(config.model.get("security_equivalence_replay_tolerance", -1.0))
            != 1e-12
        ):
            raise ScopfError(
                "GPU Lagrangian v4/v5/v6 security-row replay tolerance changed"
            )
        if float(config.model.get("phase_one_replay_tolerance_pu", -1.0)) != 1e-10:
            raise ScopfError("GPU Lagrangian v4/v5/v6 Phase-I replay tolerance changed")
    return {
        "benchmark": benchmark,
        "profile": profile,
        "cpu_comparison": benchmark["cpu_comparison"],
        "experiment_policy": experiment["policy"],
    }


def _solve_summary(result: ContinuousSolveResult) -> dict[str, Any]:
    certificate = result.statistics.get("dual_certificate", {})
    return {
        "status": result.status,
        "optimal": result.optimal,
        "primal_objective": result.primal_objective,
        "dual_objective": result.dual_objective,
        "native_solve_time_seconds": result.solve_time_seconds,
        "solved_by": result.statistics.get("solved_by"),
        "solved_by_pdlp": result.statistics.get("solved_by_pdlp"),
        "native_integer_columns": result.statistics.get("native_integer_columns"),
        "dual_certificate_passed": certificate.get("passed"),
        "primal_feasible": certificate.get("primal_feasible"),
        "native_log_final_metrics": result.statistics.get("native_log_final_metrics"),
        "native_log_sha256": result.statistics.get("native_log_sha256"),
        "native_lp_stats": result.statistics.get("lp_stats"),
        "termination_reason": result.statistics.get("termination_reason"),
        "warm_start": result.statistics.get("warm_start"),
    }


def _prepare_region_master(
    *,
    case: Any,
    network: NetworkData,
    config: RunConfig,
    masks: RegionMasks,
    initial_pairs: tuple[SecurityPair, ...],
) -> ReducedMaster:
    master = build_reduced_master(
        case,
        network,
        segments=int(config.model["pwl_segments"]),
        coefficient_zero_tolerance=float(
            config.model.get("reduced_coefficient_zero_tolerance", 1e-14)
        ),
    )
    add_reduced_security_pairs(master, network, initial_pairs)
    fix_commitments(master, masks.fixed_off, masks.fixed_on)
    return master


def _validate_prepared_region_master(
    master: ReducedMaster,
    masks: RegionMasks,
    initial_pairs: tuple[SecurityPair, ...],
) -> None:
    expected_pair_ids = {pair.pair_id for pair in initial_pairs}
    if set(master.security_pair_ids) != expected_pair_ids:
        raise ScopfError("Prepared region master security-pair identity changed")
    masks.validate(master.index.generator_source_rows.size)
    for position, column in enumerate(master.index.commitment_by_generator):
        expected_lower = 1.0 if masks.fixed_on[position] else 0.0
        expected_upper = 0.0 if masks.fixed_off[position] else 1.0
        if (
            float(master.canonical.column_lower[column]) != expected_lower
            or float(master.canonical.column_upper[column]) != expected_upper
        ):
            raise ScopfError("Prepared region master commitment bounds changed")


def _region_pmin_pmax_capacity_gate(
    *,
    case: Any,
    master: ReducedMaster,
    masks: RegionMasks,
    tolerance_pu: float,
) -> dict[str, Any]:
    """Check the exact aggregate dispatch interval implied by region masks."""

    source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64)
    masks.validate(source_rows.size)
    pmin = np.asarray(case.gen[source_rows, PMIN], dtype=np.float64)
    pmax = np.asarray(case.gen[source_rows, PMAX], dtype=np.float64)
    if not np.all(np.isfinite(pmin)) or not np.all(np.isfinite(pmax)):
        raise ScopfError("Region PMIN/PMAX capacity gate found nonfinite source data")
    if np.any(pmin > pmax):
        raise ScopfError("Region PMIN/PMAX capacity gate found PMIN above PMAX")
    free = ~(masks.fixed_off | masks.fixed_on)
    minimum_by_generator = np.zeros(source_rows.size, dtype=np.float64)
    maximum_by_generator = np.zeros(source_rows.size, dtype=np.float64)
    minimum_by_generator[masks.fixed_on] = pmin[masks.fixed_on]
    maximum_by_generator[masks.fixed_on] = pmax[masks.fixed_on]
    minimum_by_generator[free] = np.minimum(0.0, pmin[free])
    maximum_by_generator[free] = np.maximum(0.0, pmax[free])
    minimum_dispatch = float(np.sum(minimum_by_generator))
    maximum_dispatch = float(np.sum(maximum_by_generator))
    demand = float(master.operator.total_demand_mw)
    tolerance_mw = float(tolerance_pu) * float(case.base_mva)
    shortfall = max(0.0, minimum_dispatch - demand, demand - maximum_dispatch)
    return {
        "kind": "exact_aggregate_region_pmin_pmax_interval_v1",
        "passes": bool(shortfall <= tolerance_mw),
        "demand_mw": demand,
        "minimum_dispatch_mw": minimum_dispatch,
        "maximum_dispatch_mw": maximum_dispatch,
        "capacity_shortfall_mw": shortfall,
        "registered_tolerance_pu": float(tolerance_pu),
        "registered_tolerance_mw": tolerance_mw,
        "fixed_off_count": int(np.count_nonzero(masks.fixed_off)),
        "fixed_on_count": int(np.count_nonzero(masks.fixed_on)),
        "free_count": int(np.count_nonzero(free)),
        "exact_source_pmin_changed": False,
    }


def _solve_region(
    *,
    region_id: str,
    masks: RegionMasks,
    case: Any,
    network: NetworkData,
    catalog: ContingencyCatalog,
    config: RunConfig,
    deadline: Deadline,
    initial_pairs: tuple[SecurityPair, ...],
    screener: ContingencyScreener,
    checkpoint: Callable[[], None],
    progress: Callable[[dict[str, Any]], None] | None = None,
    candidate_policy: PrimalCandidatePolicy | None = None,
    prepared_master: ReducedMaster | None = None,
    initial_native_primal: np.ndarray | None = None,
    initial_native_row_dual: np.ndarray | None = None,
    initial_warm_start_origin: str | None = None,
) -> SolvedRegion:
    region_started = time.perf_counter()
    profile = config.raw["platforms"]["dgx_spark"]
    master = prepared_master or _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=initial_pairs,
    )
    _validate_prepared_region_master(master, masks, initial_pairs)
    pairs_by_id = {pair.pair_id: pair for pair in initial_pairs}
    native_primal = (
        None
        if initial_native_primal is None
        else np.asarray(initial_native_primal, dtype=np.float64).copy()
    )
    native_dual = (
        None
        if initial_native_row_dual is None
        else np.asarray(initial_native_row_dual, dtype=np.float64).copy()
    )
    if native_primal is not None and native_primal.shape != (master.canonical.num_columns,):
        raise ScopfError("Prepared region native-primal warm start has the wrong shape")
    if native_dual is not None and (
        native_dual.ndim != 1 or native_dual.size > master.canonical.num_rows
    ):
        raise ScopfError("Prepared region native-dual warm start has the wrong shape")
    if (native_primal is not None or native_dual is not None) and not initial_warm_start_origin:
        raise ScopfError("Prepared region warm start lacks an origin")
    rounds: list[dict[str, Any]] = []
    final_screen: dict[str, Any] | None = None
    last_solve: ContinuousSolveResult | None = None
    usable_primal_seen = False
    infeasible_residuals: list[float] = []
    cold_restarts_remaining = (
        candidate_policy.cold_restart_attempts if candidate_policy is not None else 0
    )
    cold_restart_active = False

    def reject_attempt(reason: str, detail: str) -> None:
        raise RegionAttemptRejected(
            detail,
            reason=reason,
            master=master,
            security_pairs=tuple(sorted(pairs_by_id.values())),
            rounds=rounds,
        )

    def emit_progress() -> None:
        region_elapsed = time.perf_counter() - region_started
        if progress is not None:
            progress(
                {
                    "region_id": region_id,
                    "rows": master.canonical.num_rows,
                    "security_pair_count": len(pairs_by_id),
                    "constraint_generation_rounds": rounds,
                    "usable_primal_seen": usable_primal_seen,
                    "region_elapsed_seconds": region_elapsed,
                    "candidate_policy": (
                        candidate_policy.as_dict() if candidate_policy is not None else None
                    ),
                    "initial_warm_start_origin": initial_warm_start_origin,
                }
            )
        else:
            checkpoint()

    maximum_rounds = int(config.runtime["maximum_constraint_generation_rounds"])
    for round_number in range(1, maximum_rounds + 1):
        deadline.require(f"PDLP region {region_id} round {round_number}")
        solver_budget = min(
            deadline.solver_budget(), float(config.runtime["maximum_pdlp_round_seconds"])
        )
        if candidate_policy is not None:
            candidate_remaining = candidate_policy.total_seconds - (
                time.perf_counter() - region_started
            )
            if candidate_remaining < candidate_policy.minimum_round_seconds:
                reject_attempt(
                    "attempt_budget_exhausted",
                    f"Region {region_id} exhausted its {candidate_policy.total_seconds:g}s "
                    "bounded PDLP-attempt budget",
                )
            solver_budget = min(
                solver_budget,
                candidate_policy.maximum_round_seconds,
                candidate_remaining,
            )
        emit_progress()
        started = time.perf_counter()
        last_solve = solve_cuopt_continuous_pdlp(
            master.canonical,
            time_limit_seconds=solver_budget,
            optimality_tolerance=float(profile["pdlp_optimality_tolerance"]),
            primal_feasibility_tolerance=float(config.model["model_residual_tolerance_pu"]),
            certificate_residual_tolerance=float(profile["dual_certificate_residual_tolerance"]),
            native_scaling_mode=str(profile["native_scaling_mode"]),
            native_base_mva=float(case.base_mva),
            log_to_console=True,
            per_constraint_residual=bool(profile["per_constraint_residual"]),
            presolve=int(profile["presolve"]),
            initial_native_primal=native_primal,
            initial_native_row_dual=native_dual,
        )
        adapter_wall = time.perf_counter() - started
        round_record: dict[str, Any] = {
            "round": round_number,
            "rows_before_solve": master.canonical.num_rows,
            "security_pairs_before_solve": len(pairs_by_id),
            "solver_budget_seconds": solver_budget,
            "adapter_wall_time_seconds": adapter_wall,
            "solve": _solve_summary(last_solve),
            "cold_restart": cold_restart_active,
        }
        if round_number == 1 and initial_warm_start_origin is not None:
            round_record["initial_warm_start_origin"] = initial_warm_start_origin
        rounds.append(round_record)
        emit_progress()
        error_status = str(last_solve.statistics.get("error_status", ""))
        if (
            error_status != "Success"
            or last_solve.values is None
            or last_solve.native_row_dual is None
        ):
            if (
                candidate_policy is not None
                and error_status == "Success"
                and last_solve.status in {"Infeasible", "TimeLimit"}
            ):
                if cold_restarts_remaining > 0 and not cold_restart_active:
                    cold_restarts_remaining -= 1
                    native_primal = None
                    native_dual = None
                    infeasible_residuals.clear()
                    cold_restart_active = True
                    round_record["candidate_gate"] = {
                        "rejected": False,
                        "reason": "cold_restart_scheduled_after_missing_vectors",
                        "remaining_cold_restarts": cold_restarts_remaining,
                    }
                    emit_progress()
                    continue
                round_record["candidate_gate"] = {
                    "rejected": True,
                    "reason": (
                        "cold_restart_returned_no_usable_primal_vectors"
                        if cold_restart_active
                        else "solver_returned_no_usable_primal_vectors"
                    ),
                    "solve_status": last_solve.status,
                }
                emit_progress()
                reject_attempt(
                    str(round_record["candidate_gate"]["reason"]),
                    f"Region {region_id} rejected candidate after round "
                    f"{round_number}: solver returned no usable primal vectors "
                    f"with status={last_solve.status}",
                )
            raise ScopfError(
                f"Region {region_id} PDLP did not return usable vectors: "
                f"status={last_solve.status}, error={error_status}"
            )
        native_primal = last_solve.native_primal
        native_dual = last_solve.native_row_dual
        dual_certificate = last_solve.statistics.get("dual_certificate", {})
        certificate_primal_feasible = bool(dual_certificate.get("primal_feasible", False))
        canonical_residual_pu = (
            master.canonical.max_row_violation(last_solve.values) / case.base_mva
        )
        canonical_primal_feasible = canonical_residual_pu <= float(
            config.model["model_residual_tolerance_pu"]
        )
        round_record["primal_acceptance"] = {
            "numeric_certificate_primal_feasible": certificate_primal_feasible,
            "canonical_model_residual_pu": canonical_residual_pu,
            "canonical_model_residual_passed": canonical_primal_feasible,
        }
        if not certificate_primal_feasible or not canonical_primal_feasible:
            infeasible_residuals.append(canonical_residual_pu)
            round_record["screen"] = {
                "skipped": True,
                "reason": "pdlp_primal_infeasible",
            }
            if candidate_policy is not None:
                primal_scale = max(
                    1.0,
                    abs(float(last_solve.primal_objective or 0.0)),
                )
                dual_magnitude = abs(float(last_solve.dual_objective or 0.0))
                dual_multiple = dual_magnitude / primal_scale
                gate: dict[str, Any] = {
                    "dual_objective_multiple_of_primal_scale": dual_multiple,
                    "dual_divergence_threshold": (candidate_policy.dual_divergence_multiple),
                    "residual_history_pu": list(infeasible_residuals),
                    "rejected": False,
                }
                rejection_reason: str | None = None
                if dual_multiple >= candidate_policy.dual_divergence_multiple:
                    rejection_reason = "dual_objective_divergence"
                window = candidate_policy.stagnation_window_rounds
                if rejection_reason is None and len(infeasible_residuals) >= window:
                    selected = infeasible_residuals[-window:]
                    baseline = max(
                        selected[0],
                        float(config.model["model_residual_tolerance_pu"]),
                    )
                    best_followup = min(selected[1:])
                    improvement = (selected[0] - best_followup) / baseline
                    gate["window_relative_residual_improvement"] = improvement
                    gate["minimum_required_relative_residual_improvement"] = (
                        candidate_policy.minimum_relative_residual_improvement
                    )
                    if improvement < candidate_policy.minimum_relative_residual_improvement:
                        rejection_reason = "primal_residual_stagnation"
                if rejection_reason is not None:
                    if cold_restarts_remaining > 0 and not cold_restart_active:
                        cold_restarts_remaining -= 1
                        native_primal = None
                        native_dual = None
                        infeasible_residuals.clear()
                        cold_restart_active = True
                        gate["cold_restart_scheduled"] = True
                        gate["trigger_reason"] = rejection_reason
                        gate["reason"] = "cold_restart_scheduled"
                        round_record["candidate_gate"] = gate
                        emit_progress()
                        continue
                    if cold_restart_active:
                        rejection_reason = f"cold_restart_failed_{rejection_reason}"
                    gate["rejected"] = True
                    gate["reason"] = rejection_reason
                    round_record["candidate_gate"] = gate
                    emit_progress()
                    reject_attempt(
                        rejection_reason,
                        f"Region {region_id} rejected candidate after round "
                        f"{round_number}: {rejection_reason}",
                    )
                round_record["candidate_gate"] = gate
            if last_solve.status == "TimeLimit" and native_primal is not None:
                if cold_restart_active:
                    round_record["candidate_gate"] = {
                        "rejected": True,
                        "reason": "cold_restart_primal_infeasible",
                        "canonical_model_residual_pu": canonical_residual_pu,
                    }
                    emit_progress()
                    reject_attempt(
                        "cold_restart_primal_infeasible",
                        f"Region {region_id} cold restart remained primal infeasible "
                        f"after round {round_number}",
                    )
                emit_progress()
                continue
            if candidate_policy is not None:
                round_record["candidate_gate"] = {
                    "rejected": True,
                    "reason": "solver_terminated_with_primal_infeasibility",
                    "solve_status": last_solve.status,
                    "canonical_model_residual_pu": canonical_residual_pu,
                }
                emit_progress()
                reject_attempt(
                    "solver_terminated_with_primal_infeasibility",
                    f"Region {region_id} rejected candidate after round "
                    f"{round_number}: status={last_solve.status}, "
                    f"residual_pu={canonical_residual_pu:.6e}",
                )
            raise ScopfError(
                f"Region {region_id} PDLP primal is unusable: "
                f"status={last_solve.status}, residual_pu={canonical_residual_pu:.6e}"
            )
        usable_primal_seen = True
        cold_restart_active = False
        dispatch = reduced_dispatch(master, last_solve.values)
        flow = master.operator.flows(dispatch)
        screen_started = time.perf_counter()
        screened = screener.screen(
            flow,
            tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
            already_added=set(pairs_by_id),
        )
        final_screen = {
            "wall_time_seconds": time.perf_counter() - screen_started,
            "evaluated_sides": screened.evaluated_pairs,
            "new_violated_pairs": len(screened.violations),
            "maximum_violation_pu": screened.maximum_violation_pu,
            "maximum_pair_id": screened.maximum_pair_id,
        }
        round_record["screen"] = final_screen
        emit_progress()
        if not screened.violations:
            if screened.maximum_violation_pu > float(
                config.model["security_violation_tolerance_pu"]
            ):
                raise ScopfError(f"Region {region_id} final screen residual exceeds tolerance")
            break
        add_reduced_security_pairs(master, network, screened.violations)
        pairs_by_id.update((pair.pair_id, pair) for pair in screened.violations)
        round_record["added_pair_ids"] = [pair.pair_id for pair in screened.violations]
        emit_progress()
    else:
        reason = (
            "without a primal-feasible PDLP vector"
            if not usable_primal_seen
            else "before a zero-violation exhaustive screen"
        )
        if candidate_policy is not None:
            reject_attempt(
                "constraint_generation_round_limit",
                f"Region {region_id} reached its constraint-generation limit {reason}",
            )
        raise ScopfError(f"Region {region_id} reached its constraint-generation limit {reason}")

    assert last_solve is not None and final_screen is not None
    if last_solve.native_row_dual is None or last_solve.values is None:
        raise ScopfError(f"Region {region_id} lost its final PDLP vectors")
    row_dual = canonical_row_duals(
        master,
        last_solve.native_row_dual,
        native_scaling_mode=str(profile["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    if last_solve.primal_objective is None:
        raise ScopfError(f"Region {region_id} lacks a finite LP primal target")
    gpu_started = time.perf_counter()
    polished_row_dual, gpu_evaluation = optimize_lagrangian_bound_cupy(
        master,
        row_dual,
        masks,
        relaxation_primal_objective=float(last_solve.primal_objective),
        iterations=int(profile["lagrangian_gpu_iterations"]),
        polyak_fraction=float(profile["lagrangian_polyak_fraction"]),
    )
    gpu_evaluation["wall_time_seconds"] = time.perf_counter() - gpu_started
    replay_started = time.perf_counter()
    evaluation = evaluate_lagrangian_bound(
        master,
        polished_row_dual,
        masks,
        safety_margin_dollars=float(config.raw["benchmark"]["certificate_safety_margin_dollars"]),
    )
    gpu_evaluation["independent_cpu_replay_wall_time_seconds"] = (
        time.perf_counter() - replay_started
    )
    replay_difference = abs(
        float(gpu_evaluation["best_raw_lower_bound"]) - evaluation.raw_lower_bound
    )
    if replay_difference > float(config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]):
        raise ScopfError(
            f"Region {region_id} GPU/CPU Lagrangian replay differs by {replay_difference}"
        )
    gpu_evaluation["cpu_replay_difference_dollars"] = replay_difference
    return SolvedRegion(
        region_id=region_id,
        masks=masks,
        master=master,
        solve=last_solve,
        canonical_row_dual=row_dual,
        lagrangian=evaluation,
        commitment=commitment_vector(master, last_solve.values),
        security_pairs=tuple(sorted(pairs_by_id.values())),
        rounds=rounds,
        final_screen=final_screen,
        gpu_lagrangian=gpu_evaluation,
    )


def _capacity_repaired_commitment(
    case: Any,
    source_rows: np.ndarray,
    proposed: np.ndarray,
    on_values: np.ndarray,
    masks: RegionMasks,
    demand_mw: float,
) -> np.ndarray:
    commitment = np.asarray(proposed >= 0.5, dtype=np.int8)
    commitment[masks.fixed_off] = 0
    commitment[masks.fixed_on] = 1
    pmax = case.gen[source_rows, PMAX]
    pmin = case.gen[source_rows, PMIN]
    free_off = np.flatnonzero((commitment == 0) & ~masks.fixed_off)
    for position in free_off[np.argsort(on_values[free_off])]:
        if float(pmax @ commitment) >= demand_mw:
            break
        commitment[position] = 1
    if float(pmax @ commitment) < demand_mw:
        raise ScopfError("Candidate commitment cannot cover demand at PMAX")
    removable = np.flatnonzero((commitment == 1) & ~masks.fixed_on)
    for position in removable[np.argsort(on_values[removable])[::-1]]:
        if float(pmin @ commitment) <= demand_mw:
            break
        trial = commitment.copy()
        trial[position] = 0
        if float(pmax @ trial) >= demand_mw:
            commitment = trial
    if float(pmin @ commitment) > demand_mw:
        raise ScopfError("Candidate commitment exceeds demand at PMIN")
    return commitment


def _region_record(region: SolvedRegion) -> dict[str, Any]:
    record = {
        "region_id": region.region_id,
        **region.masks.as_dict(region.master.index.generator_source_rows + 1),
        "lp_objective": region.solve.primal_objective,
        "lp_commitment_fractional_count": int(
            np.count_nonzero(np.abs(region.commitment - np.rint(region.commitment)) > 1e-6)
        ),
        "constraint_generation_rounds": region.rounds,
        "final_screen": region.final_screen,
        "coefficient_cleanup_audit": dict(region.master.coefficient_cleanup_audit),
        "security_row_equivalence": _security_row_equivalence_record(region.master),
        "security_pairs": [security_pair_record(pair) for pair in region.security_pairs],
        "gpu_lagrangian_evaluation": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in region.gpu_lagrangian.items()
        },
        "lagrangian_certificate": region.lagrangian.as_dict(
            region.master.index.generator_source_rows + 1
        ),
    }
    return record


def _security_row_equivalence_record(master: ReducedMaster) -> dict[str, Any]:
    classes = {
        representative: list(members)
        for representative, members in sorted(
            master.security_pair_equivalence_classes.items()
        )
    }
    return {
        "policy": "exact_post_cleanup_fp64_coefficient_and_rhs_identity",
        "logical_security_pair_count": len(master.security_pair_ids),
        "distinct_solver_security_row_count": len(classes),
        "duplicate_solver_row_count": len(master.security_pair_ids) - len(classes),
        "representative_by_pair_id": dict(
            sorted(master.security_pair_representative_by_id.items())
        ),
        "equivalence_classes": classes,
    }


def _relative_gap(*, objective: float, lower_bound: float) -> float:
    """Return the registered incumbent-relative minimization gap."""

    gap = (float(objective) - float(lower_bound)) / max(1.0, abs(float(objective)))
    if gap < -1e-9:
        raise ScopfError("Lagrangian lower bound exceeds the secure incumbent")
    return gap


def _replay_cleanup_audit_comparison(
    recorded: dict[str, Any],
    rebuilt: dict[str, Any],
) -> dict[str, Any]:
    """Validate cleanup invariants while allowing architecture-dependent dust counts."""

    maximum_difference = 0.0
    maximum_integer_difference = 0
    differing_integer_fields: list[dict[str, Any]] = []
    exact_paths = {
        "coefficient_cleanup_audit.policy",
        "coefficient_cleanup_audit.zero_tolerance",
        "coefficient_cleanup_audit.potential_flow_operator_dust.zero_tolerance",
        "coefficient_cleanup_audit.physical_injection_operator_changed",
        "coefficient_cleanup_audit.solver_rows_are_relaxations_of_original_rows",
        "coefficient_cleanup_audit.exact_duplicate_security_row_count",
    }

    def compare(left: Any, right: Any, path: str) -> None:
        nonlocal maximum_difference, maximum_integer_difference
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                raise ScopfError(f"Cleanup-audit key mismatch at {path}")
            for key in sorted(left):
                compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                raise ScopfError(f"Cleanup-audit boolean mismatch at {path}")
            return
        if isinstance(left, int) or isinstance(right, int):
            if type(left) is not type(right):
                raise ScopfError(f"Cleanup-audit numeric type mismatch at {path}")
            difference = abs(int(left) - int(right))
            if path in exact_paths and difference:
                raise ScopfError(f"Cleanup-audit exact integer mismatch at {path}")
            maximum_integer_difference = max(maximum_integer_difference, difference)
            if difference:
                differing_integer_fields.append(
                    {
                        "path": path,
                        "recorded": int(left),
                        "rebuilt": int(right),
                        "absolute_difference": difference,
                    }
                )
            return
        if isinstance(left, (float, np.floating)) and isinstance(
            right, (float, np.floating)
        ):
            difference = abs(float(left) - float(right))
            if not np.isfinite(difference):
                raise ScopfError(f"Cleanup-audit nonfinite FP64 value at {path}")
            if path in exact_paths and difference:
                raise ScopfError(f"Cleanup-audit exact FP64 mismatch at {path}")
            maximum_difference = max(maximum_difference, difference)
            return
        if left != right:
            raise ScopfError(f"Cleanup-audit identity mismatch at {path}")

    compare(recorded, rebuilt, "coefficient_cleanup_audit")
    for label, audit in (("recorded", recorded), ("rebuilt", rebuilt)):
        zero_tolerance = float(audit["zero_tolerance"])
        for key in (
            "maximum_absolute_dropped_generator_coefficient",
            "maximum_row_rhs_outward_relaxation",
            "total_rhs_outward_relaxation",
        ):
            value = float(audit.get(key, 0.0))
            if not np.isfinite(value) or value < 0.0:
                raise ScopfError(f"Cleanup-audit {label} {key} is invalid")
        potential = audit["potential_flow_operator_dust"]
        maximum_dropped = float(potential["maximum_absolute_dropped_coefficient"])
        if maximum_dropped > zero_tolerance * (1.0 + 1e-12):
            raise ScopfError(
                f"Cleanup-audit {label} flow dust exceeds its zero tolerance"
            )
        generator_maximum = float(
            audit.get("maximum_absolute_dropped_generator_coefficient", 0.0)
        )
        if generator_maximum > zero_tolerance * (1.0 + 1e-12):
            raise ScopfError(
                f"Cleanup-audit {label} generator dust exceeds its zero tolerance"
            )
        if audit.get("physical_injection_operator_changed") is not False:
            raise ScopfError(f"Cleanup-audit {label} changed the physical operator")
        if audit.get("solver_rows_are_relaxations_of_original_rows") is not True:
            raise ScopfError(f"Cleanup-audit {label} lost outward-relaxation proof")
    return {
        "maximum_fp64_difference": maximum_difference,
        "maximum_integer_difference": maximum_integer_difference,
        "differing_integer_fields": differing_integer_fields,
    }


def _run_phase_one_attempt(
    *,
    region_id: str,
    masks: RegionMasks,
    rejected: RegionAttemptRejected,
    case: Any,
    config: RunConfig,
    deadline: Deadline,
    attempt_kind: str = "post_cost_lp_rejection",
    time_limit_seconds: float | None = None,
    capacity_gate: dict[str, Any] | None = None,
) -> PhaseOneAttemptResult:
    """Run one cold GPU Phase-I solve and serialize its projected box dual."""

    runtime = config.runtime
    profile = config.raw["platforms"]["dgx_spark"]
    attempt_started = time.perf_counter()
    deadline.require(f"Phase-I certificate for {region_id}")
    phase_model = build_phase_one_model(
        rejected.master.canonical,
        base_mva=float(case.base_mva),
        maximum_violation_pu=float(runtime["phase_one_maximum_violation_pu"]),
    )
    source_model = rejected.master.canonical
    if phase_model.variable_names[:-1] != source_model.variable_names:
        raise ScopfError("Phase-I and cost-LP source-column identities changed")
    if not np.array_equal(
        np.asarray(phase_model.column_lower[:-1], dtype=np.float64),
        np.asarray(source_model.column_lower, dtype=np.float64),
    ) or not np.array_equal(
        np.asarray(phase_model.column_upper[:-1], dtype=np.float64),
        np.asarray(source_model.column_upper, dtype=np.float64),
    ):
        raise ScopfError("Phase-I and cost-LP source-column bounds changed")
    source_column_scale, _ = native_scaling_vectors(
        source_model,
        mode=str(profile["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    phase_column_scale, phase_row_scale = native_scaling_vectors(
        phase_model,
        mode=str(profile["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    if not np.array_equal(source_column_scale, phase_column_scale[:-1]):
        raise ScopfError("Phase-I and cost-LP native source-column scaling changed")
    preparation_wall = time.perf_counter() - attempt_started
    budget = min(
        deadline.solver_budget(),
        (
            float(runtime["phase_one_time_limit_seconds"])
            if time_limit_seconds is None
            else float(time_limit_seconds)
        ),
    )
    solve_started = time.perf_counter()
    solve = solve_cuopt_continuous_pdlp(
        phase_model,
        time_limit_seconds=budget,
        optimality_tolerance=float(profile["pdlp_optimality_tolerance"]),
        primal_feasibility_tolerance=float(config.model["model_residual_tolerance_pu"]),
        certificate_residual_tolerance=float(profile["dual_certificate_residual_tolerance"]),
        native_scaling_mode=str(profile["native_scaling_mode"]),
        native_base_mva=float(case.base_mva),
        log_to_console=True,
        per_constraint_residual=bool(profile["per_constraint_residual"]),
        presolve=int(profile["presolve"]),
    )
    record: dict[str, Any] = {
        "region_id": region_id,
        "phase_one_attempt_kind": attempt_kind,
        **masks.as_dict(rejected.master.index.generator_source_rows + 1),
        "source_attempt_rejection_reason": rejected.reason,
        "source_constraint_generation_rounds": rejected.rounds,
        "security_pairs": [
            security_pair_record(pair) for pair in rejected.security_pairs
        ],
        "coefficient_cleanup_audit": dict(rejected.master.coefficient_cleanup_audit),
        "security_row_equivalence": _security_row_equivalence_record(rejected.master),
        "phase_one_model": {
            "columns": phase_model.num_columns,
            "rows": phase_model.num_rows,
            "registered_maximum_violation_cap_pu": float(
                runtime["phase_one_maximum_violation_pu"]
            ),
            "box_derived_violation_upper_bound_pu": float(
                phase_model.column_upper[-1]
            ),
        },
        "solver_budget_seconds": budget,
        "preparation_wall_time_seconds": preparation_wall,
        "adapter_wall_time_seconds": time.perf_counter() - solve_started,
        "total_attempt_wall_time_seconds": time.perf_counter() - attempt_started,
        "solve": _solve_summary(solve),
        "prune_certified": False,
    }
    if capacity_gate is not None:
        record["pmin_pmax_capacity_gate"] = capacity_gate
    source_native_primal: np.ndarray | None = None
    source_residual_pu: float | None = None
    violation_value_pu: float | None = None
    numeric_primal_feasible = bool(
        solve.statistics.get("dual_certificate", {}).get("primal_feasible", False)
    )
    if solve.values is not None and solve.native_primal is not None:
        canonical_values = np.asarray(solve.values, dtype=np.float64)
        native_values = np.asarray(solve.native_primal, dtype=np.float64)
        if (
            canonical_values.shape == (phase_model.num_columns,)
            and native_values.shape == (phase_model.num_columns,)
            and np.all(np.isfinite(canonical_values))
            and np.all(np.isfinite(native_values))
        ):
            violation_value_pu = float(canonical_values[-1])
            source_values = canonical_values[:-1]
            source_residual_pu = (
                rejected.master.canonical.max_row_violation(source_values)
                / float(case.base_mva)
            )
            source_tolerance = float(config.model["model_residual_tolerance_pu"])
            if (
                numeric_primal_feasible
                and violation_value_pu <= source_tolerance
                and source_residual_pu <= source_tolerance
            ):
                source_native_primal = native_values[:-1].copy()
    record["source_feasible_warm_start"] = {
        "eligible": source_native_primal is not None,
        "phase_one_violation_pu": violation_value_pu,
        "source_model_residual_pu": source_residual_pu,
        "numeric_certificate_primal_feasible": numeric_primal_feasible,
        "native_primal_count": (
            None if source_native_primal is None else int(source_native_primal.size)
        ),
        "native_primal_sha256": (
            None
            if source_native_primal is None
            else hashlib.sha256(source_native_primal.tobytes()).hexdigest()
        ),
        "dual_transferred": False,
        "dual_not_transferred_reason": (
            "phase_one_uses_split_upper_rows_while_cost_lp_uses_native_ranged_rows"
        ),
        "source_column_scaling_exactly_equal": True,
    }
    if (
        str(solve.statistics.get("error_status")) != "Success"
        or solve.native_row_dual is None
    ):
        record["certificate_status"] = "unavailable_solver_dual"
        record["total_attempt_wall_time_seconds"] = time.perf_counter() - attempt_started
        return PhaseOneAttemptResult(record, source_native_primal)
    if solve.native_row_dual.shape != (phase_model.num_rows,):
        record["certificate_status"] = "invalid_native_dual_shape"
        record["total_attempt_wall_time_seconds"] = time.perf_counter() - attempt_started
        return PhaseOneAttemptResult(record, source_native_primal)
    canonical_dual = np.asarray(solve.native_row_dual, dtype=np.float64) * phase_row_scale
    certificate = phase_one_certificate(
        phase_model,
        canonical_dual,
        safety_margin_pu=float(runtime["phase_one_safety_margin_pu"]),
        infeasibility_threshold_pu=float(
            runtime["phase_one_infeasibility_threshold_pu"]
        ),
    )
    record["phase_one_certificate"] = certificate
    record["certificate_status"] = (
        "independently_replayable_prune"
        if certificate["prune_certified"]
        else "dual_bound_not_strong_enough"
    )
    record["prune_certified"] = bool(certificate["prune_certified"])
    if record["prune_certified"] and source_native_primal is not None:
        raise ScopfError(
            "Phase-I produced contradictory feasible-primal and positive-dual certificates"
        )
    record["total_attempt_wall_time_seconds"] = time.perf_counter() - attempt_started
    return PhaseOneAttemptResult(record, source_native_primal)


def _phase_one_attempt_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Retain Phase-I timing/gate evidence without duplicating full dual vectors."""

    certificate = record.get("phase_one_certificate")
    certificate_summary = None
    if isinstance(certificate, dict):
        certificate_summary = {
            key: value
            for key, value in certificate.items()
            if key != "canonical_row_duals"
        }
    return {
        "region_id": record["region_id"],
        "phase_one_attempt_kind": record["phase_one_attempt_kind"],
        "pmin_pmax_capacity_gate": record.get("pmin_pmax_capacity_gate"),
        "phase_one_model": record["phase_one_model"],
        "solver_budget_seconds": record["solver_budget_seconds"],
        "adapter_wall_time_seconds": record["adapter_wall_time_seconds"],
        "preparation_wall_time_seconds": record["preparation_wall_time_seconds"],
        "total_attempt_wall_time_seconds": record["total_attempt_wall_time_seconds"],
        "solve": record["solve"],
        "certificate_status": record.get("certificate_status"),
        "prune_certified": record["prune_certified"],
        "phase_one_certificate": certificate_summary,
        "source_feasible_warm_start": record["source_feasible_warm_start"],
        "security_pair_count": len(record["security_pairs"]),
    }


def _masks_from_record(record: dict[str, Any], source_rows: np.ndarray) -> RegionMasks:
    lookup = {int(row): position for position, row in enumerate(source_rows)}
    off = np.zeros(source_rows.size, dtype=bool)
    on = np.zeros(source_rows.size, dtype=bool)
    for row in record["fixed_off_generator_source_rows"]:
        if int(row) not in lookup:
            raise ScopfError("Certificate fixes an unknown generator source row off")
        off[lookup[int(row)]] = True
    for row in record["fixed_on_generator_source_rows"]:
        if int(row) not in lookup:
            raise ScopfError("Certificate fixes an unknown generator source row on")
        on[lookup[int(row)]] = True
    masks = RegionMasks(off, on)
    masks.validate(source_rows.size)
    return masks


def verify_lagrangian_certificate_payload(
    config: RunConfig, payload: dict[str, Any]
) -> dict[str, Any]:
    """Reread raw inputs and replay every leaf bound and the region cover on CPU."""

    started = time.perf_counter()
    case = read_matpower_case(
        config.case_path, expected_sha256=config.raw["raw_inputs"]["case_sha256"]
    )
    table = read_contingency_table(
        config.contingency_path,
        expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
    )
    network = build_network(case)
    catalog = build_contingency_catalog(
        case,
        network,
        table,
        validation_columns=int(config.model["lodf_validation_columns"]),
        validation_tolerance_pu=float(config.model["lodf_validation_tolerance_pu"]),
        chunk_columns=int(config.model["lodf_build_chunk_columns"]),
    )
    source_rows = np.flatnonzero(case.gen[:, GEN_STATUS] > 0).astype(np.int64) + 1
    leaves: dict[str, RegionMasks] = {}
    replayed_bounds: dict[str, float] = {}
    maximum_difference = 0.0
    maximum_lodf_replay_difference = 0.0
    maximum_cleanup_audit_difference = 0.0
    maximum_cleanup_audit_integer_difference = 0
    cleanup_audit_integer_differences: list[dict[str, Any]] = []
    lodf_tolerance = float(config.model.get("serialized_lodf_replay_tolerance", 0.0))

    def update_lodf_replay_difference(
        records: list[dict[str, Any]], pairs: tuple[SecurityPair, ...]
    ) -> None:
        nonlocal maximum_lodf_replay_difference
        for pair_record, pair in zip(records, pairs, strict=True):
            observed = (
                catalog.lodf[pair.monitored_active_index, pair.outage_column]
                if catalog.lodf is not None
                else catalog.lodf_operator.lodf_columns(
                    np.asarray([pair.outage_active_index], dtype=np.int64)
                )[pair.monitored_active_index, 0]
            )
            maximum_lodf_replay_difference = max(
                maximum_lodf_replay_difference,
                abs(float(observed) - float(pair_record["lodf_value"])),
            )

    for record in payload["frontier_regions"]:
        region_id = str(record["region_id"])
        if region_id in leaves:
            raise ScopfError("Duplicate frontier region id")
        masks = _masks_from_record(record, source_rows)
        master = build_reduced_master(
            case,
            network,
            segments=10,
            coefficient_zero_tolerance=float(
                config.model.get("reduced_coefficient_zero_tolerance", 1e-14)
            ),
        )
        pairs = tuple(
            security_pair_from_record(
                pair_record,
                catalog,
                lodf_absolute_tolerance=lodf_tolerance,
            )
            for pair_record in record["security_pairs"]
        )
        update_lodf_replay_difference(record["security_pairs"], pairs)
        equivalence = record.get("security_row_equivalence", {})
        add_reduced_security_pairs(
            master,
            network,
            pairs,
            expected_representative_by_pair_id=equivalence.get(
                "representative_by_pair_id"
            ),
            equivalence_replay_tolerance=float(
                config.model.get("security_equivalence_replay_tolerance", 0.0)
            ),
        )
        cleanup_comparison = _replay_cleanup_audit_comparison(
            record["coefficient_cleanup_audit"],
            master.coefficient_cleanup_audit,
        )
        maximum_cleanup_audit_difference = max(
            maximum_cleanup_audit_difference,
            float(cleanup_comparison["maximum_fp64_difference"]),
        )
        maximum_cleanup_audit_integer_difference = max(
            maximum_cleanup_audit_integer_difference,
            int(cleanup_comparison["maximum_integer_difference"]),
        )
        cleanup_audit_integer_differences.extend(
            {"region_id": region_id, **difference}
            for difference in cleanup_comparison["differing_integer_fields"]
        )
        if record.get("security_row_equivalence") != _security_row_equivalence_record(master):
            raise ScopfError(f"Independent region {region_id} security-row audit mismatch")
        replayed = replay_lagrangian_certificate(master, record["lagrangian_certificate"], masks)
        recorded = float(record["lagrangian_certificate"]["conservative_lower_bound"])
        difference = abs(replayed.conservative_lower_bound - recorded)
        maximum_difference = max(maximum_difference, difference)
        if difference > float(config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]):
            raise ScopfError(f"Independent region {region_id} replay mismatch")
        leaves[region_id] = masks
        replayed_bounds[region_id] = replayed.conservative_lower_bound
    maximum_phase_one_difference = 0.0
    pruned_records = payload.get("pruned_regions", [])
    for record in pruned_records:
        region_id = str(record["region_id"])
        if region_id in leaves:
            raise ScopfError("Duplicate active/pruned region id")
        masks = _masks_from_record(record, source_rows)
        master = build_reduced_master(
            case,
            network,
            segments=10,
            coefficient_zero_tolerance=float(
                config.model.get("reduced_coefficient_zero_tolerance", 1e-14)
            ),
        )
        pairs = tuple(
            security_pair_from_record(
                pair_record,
                catalog,
                lodf_absolute_tolerance=lodf_tolerance,
            )
            for pair_record in record["security_pairs"]
        )
        update_lodf_replay_difference(record["security_pairs"], pairs)
        equivalence = record.get("security_row_equivalence", {})
        add_reduced_security_pairs(
            master,
            network,
            pairs,
            expected_representative_by_pair_id=equivalence.get(
                "representative_by_pair_id"
            ),
            equivalence_replay_tolerance=float(
                config.model.get("security_equivalence_replay_tolerance", 0.0)
            ),
        )
        fix_commitments(master, masks.fixed_off, masks.fixed_on)
        cleanup_comparison = _replay_cleanup_audit_comparison(
            record["coefficient_cleanup_audit"],
            master.coefficient_cleanup_audit,
        )
        maximum_cleanup_audit_difference = max(
            maximum_cleanup_audit_difference,
            float(cleanup_comparison["maximum_fp64_difference"]),
        )
        maximum_cleanup_audit_integer_difference = max(
            maximum_cleanup_audit_integer_difference,
            int(cleanup_comparison["maximum_integer_difference"]),
        )
        cleanup_audit_integer_differences.extend(
            {"region_id": region_id, **difference}
            for difference in cleanup_comparison["differing_integer_fields"]
        )
        if record.get("security_row_equivalence") != _security_row_equivalence_record(master):
            raise ScopfError(f"Independent pruned region {region_id} row audit mismatch")
        phase_model = build_phase_one_model(
            master.canonical,
            base_mva=float(case.base_mva),
            maximum_violation_pu=float(
                config.runtime["phase_one_maximum_violation_pu"]
            ),
        )
        recorded_phase_model = record["phase_one_model"]
        if (
            phase_model.num_columns != int(recorded_phase_model["columns"])
            or phase_model.num_rows != int(recorded_phase_model["rows"])
            or abs(
                float(phase_model.column_upper[-1])
                - float(
                    recorded_phase_model[
                        "box_derived_violation_upper_bound_pu"
                    ]
                )
            )
            > float(config.model["phase_one_replay_tolerance_pu"])
        ):
            raise ScopfError(f"Independent Phase-I model mismatch for region {region_id}")
        replayed_phase = replay_phase_one_certificate(
            phase_model, record["phase_one_certificate"]
        )
        recorded_phase_bound = float(
            record["phase_one_certificate"]["conservative_lower_bound_pu"]
        )
        phase_difference = abs(
            float(replayed_phase["conservative_lower_bound_pu"])
            - recorded_phase_bound
        )
        maximum_phase_one_difference = max(
            maximum_phase_one_difference, phase_difference
        )
        if (
            not replayed_phase["prune_certified"]
            or phase_difference
            > float(config.model["phase_one_replay_tolerance_pu"])
        ):
            raise ScopfError(f"Independent Phase-I replay failed for region {region_id}")
        leaves[region_id] = masks
    cover_passed = verify_disjunctive_cover(source_rows.size, payload["disjunctive_splits"], leaves)
    if not replayed_bounds:
        raise ScopfError("Lagrangian cover has no active feasible frontier region")
    global_bound = min(replayed_bounds.values())
    recorded_global = float(payload["bound"])
    global_difference = abs(global_bound - recorded_global)
    passed = bool(
        cover_passed
        and global_difference <= float(config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"])
    )
    return {
        "passed": passed,
        "certificate_kind": "independent_raw_input_lagrangian_cover_replay_v1",
        "frontier_region_count": len(leaves),
        "active_frontier_region_count": len(replayed_bounds),
        "phase_one_pruned_region_count": len(pruned_records),
        "disjunctive_cover_passed": cover_passed,
        "replayed_global_lower_bound": global_bound,
        "recorded_global_lower_bound": recorded_global,
        "global_bound_difference_dollars": global_difference,
        "maximum_region_replay_difference_dollars": maximum_difference,
        "maximum_phase_one_replay_difference_pu": maximum_phase_one_difference,
        "maximum_lodf_replay_difference": maximum_lodf_replay_difference,
        "registered_lodf_replay_tolerance": lodf_tolerance,
        "maximum_cleanup_audit_replay_difference": (
            maximum_cleanup_audit_difference
        ),
        "maximum_cleanup_audit_integer_difference": (
            maximum_cleanup_audit_integer_difference
        ),
        "cleanup_audit_integer_differences": cleanup_audit_integer_differences,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _load_cpu_comparison(config: RunConfig, registration: dict[str, Any]) -> dict[str, Any]:
    reference = registration["cpu_comparison"]
    path = (config.root / str(reference["result_file"])).resolve()
    result = _read_json(path)
    canonical_bytes = json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    if canonical_sha256 != str(reference["canonical_json_sha256"]):
        raise ScopfError("Registered laptop comparison canonical JSON hash mismatch")
    if result.get("experiment_suite_id") != "activsg500-gap-sensitivity-v1":
        raise ScopfError("Registered laptop comparison suite identity changed")
    if result.get("tag") != reference.get("source_tag"):
        raise ScopfError("Registered laptop comparison tag changed")
    if result.get("commit") != reference.get("source_commit"):
        raise ScopfError("Registered laptop comparison commit changed")
    hashes = result.get("source_hashes", {})
    if (
        hashes.get("case_sha256") != config.raw["raw_inputs"]["case_sha256"]
        or hashes.get("contingency_sha256") != config.raw["raw_inputs"]["contingency_sha256"]
    ):
        raise ScopfError("Registered laptop comparison raw-input hashes changed")
    matching = [record for record in result.get("summary", []) if record.get("gap_label") == "1e-3"]
    if len(matching) != 1:
        raise ScopfError("Registered laptop comparison lacks one 1e-3 record")
    summary = matching[0]
    if summary.get("raw_result_sha256") != reference.get("raw_result_sha256"):
        raise ScopfError("Registered laptop raw-result identity changed")
    if (
        float(summary.get("final_security_violation_pu", float("inf"))) > 1e-5
        or float(summary.get("final_model_residual_pu", float("inf"))) > 1e-6
    ):
        raise ScopfError("Registered laptop comparison did not pass its original gates")
    return {
        "source": str(path.relative_to(config.root)),
        "canonical_json_sha256": canonical_sha256,
        "raw_result_sha256": summary["raw_result_sha256"],
        "frozen_identity": {
            "commit": result.get("commit"),
            "tag": result.get("tag"),
        },
        "status": "optimal_verified",
        "objective": summary.get("objective"),
        "bound": summary.get("bound"),
        "mip_gap": summary.get("achieved_mip_gap"),
        "commitment_count": summary.get("commitment_count"),
        "total_wall_time_seconds": summary.get("total_wall_seconds"),
    }


def run_gpu_lagrangian_experiment(
    config: RunConfig,
    *,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    registration = validate_lagrangian_experiment_config(config)
    deadline = Deadline(
        total_seconds=float(config.runtime["deadline_seconds"]),
        verification_reserve_seconds=float(config.runtime["verification_reserve_seconds"]),
        serialization_reserve_seconds=float(config.runtime["serialization_reserve_seconds"]),
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "case_name": config.case_name,
        "benchmark_id": config.benchmark_id,
        "experiment_policy": registration["experiment_policy"],
        "platform": "dgx_spark",
        "status": "running",
        "deadline_seconds": float(config.runtime["deadline_seconds"]),
        "requested_relative_gap": float(config.model["mip_relative_gap_tolerance"]),
        "integer_solver_used": False,
        "branch_and_bound_performed": False,
        "custom_cuda_kernel_used": False,
        "disjunctive_splits": [],
        "failed_disjunctive_split_attempts": [],
        "pruned_regions": [],
        "phase_one_prechecks": [],
        "phase_one_fallback_attempts": [],
        "frontier_regions": [],
        "primal_repairs": [],
        "secure_incumbent_checkpoint_history": [],
        "timings_seconds": {},
    }
    memory = PeakMemorySampler(sample_gpu=True)

    def save() -> None:
        payload["elapsed_seconds"] = deadline.elapsed
        payload["peak_memory"] = {
            "process_rss_bytes": memory.peak_process_rss_bytes,
            "cupy_pool_used_bytes": memory.peak_gpu_pool_used_bytes,
            "cuda_device_memory_delta_bytes": memory.peak_cuda_device_memory_delta_bytes,
        }
        if checkpoint is not None:
            checkpoint(payload)

    def save_region_progress(record: dict[str, Any]) -> None:
        payload["active_region_progress"] = record
        save()

    with memory:
        guard_runtime_environment(config.root)
        validate_platform(config, "dgx_spark")
        payload["environment"] = environment_manifest("dgx_spark")
        payload["cpu_comparison"] = _load_cpu_comparison(config, registration)
        save()

        started = time.perf_counter()
        case = read_matpower_case(
            config.case_path,
            expected_sha256=config.raw["raw_inputs"]["case_sha256"],
        )
        table = read_contingency_table(
            config.contingency_path,
            expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
        )
        payload["source_manifest"] = build_source_manifest(case, table)
        payload["timings_seconds"]["raw_input_loading"] = time.perf_counter() - started

        started = time.perf_counter()
        network = build_network(case)
        catalog = build_contingency_catalog(
            case,
            network,
            table,
            validation_columns=int(config.model["lodf_validation_columns"]),
            validation_tolerance_pu=float(config.model["lodf_validation_tolerance_pu"]),
            chunk_columns=int(config.model["lodf_build_chunk_columns"]),
        )
        base_master = build_reduced_master(
            case,
            network,
            segments=10,
            coefficient_zero_tolerance=float(
                config.model.get("reduced_coefficient_zero_tolerance", 1e-14)
            ),
        )
        payload["timings_seconds"]["network_and_reduced_model_build"] = (
            time.perf_counter() - started
        )
        payload["contingencies"] = contingency_catalog_report(catalog)
        payload["pwl_costs"] = pwl_approximation_report(base_master.costs)
        payload["reduced_model_identity"] = {
            "generator_count": int(base_master.index.generator_source_rows.size),
            "base_rows": base_master.canonical.num_rows,
            "base_columns": base_master.canonical.num_columns,
            "network_elimination": "FP64 reference-bus affine angle/flow map",
            "exact_pmin_changed": False,
            "integer_generator_subproblem": "off_or_exact_pmin_plus_ten_segments",
            "coefficient_cleanup": dict(base_master.coefficient_cleanup_audit),
        }
        screener = ContingencyScreener(
            network,
            catalog,
            backend="cupy",
            chunk_columns=int(config.model["screen_chunk_columns"]),
        )
        save()

        source_rows = base_master.index.generator_source_rows
        global_pairs: dict[str, SecurityPair] = {}
        frontier: dict[str, SolvedRegion] = {}
        all_region_records: list[dict[str, Any]] = []

        def persist_region_evidence() -> None:
            payload["all_solved_region_count"] = len(all_region_records)
            payload["solved_region_history"] = list(all_region_records)
            payload["frontier_regions"] = [
                _region_record(frontier[region_id]) for region_id in sorted(frontier)
            ]
            if frontier:
                payload["bound"] = min(
                    region.lagrangian.conservative_lower_bound for region in frontier.values()
                )
                payload["bound_status"] = "gpu_generated_pending_independent_replay"
                if payload.get("objective") is not None:
                    payload["relative_gap"] = _relative_gap(
                        objective=float(payload["objective"]),
                        lower_bound=float(payload["bound"]),
                    )
            save()

        def replay_and_checkpoint_frontier(stage: str) -> None:
            payload["active_stage"] = stage
            replay = verify_lagrangian_certificate_payload(config, payload)
            if not replay["passed"]:
                raise ScopfError(f"Independent Lagrangian replay failed during {stage}")
            payload["bound"] = replay["replayed_global_lower_bound"]
            payload["bound_status"] = "independently_replayed_current_frontier"
            if payload.get("objective") is not None:
                payload["relative_gap"] = _relative_gap(
                    objective=float(payload["objective"]),
                    lower_bound=float(payload["bound"]),
                )
            payload.setdefault("lagrangian_replay_history", []).append(
                {
                    "stage": stage,
                    **replay,
                }
            )
            save()

        payload["active_stage"] = "root_lagrangian_relaxation"
        root = _solve_region(
            region_id="r",
            masks=RegionMasks.root(source_rows.size),
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=deadline,
            initial_pairs=(),
            screener=screener,
            checkpoint=save,
            progress=save_region_progress,
        )
        payload.pop("active_region_progress", None)
        frontier[root.region_id] = root
        global_pairs.update((pair.pair_id, pair) for pair in root.security_pairs)
        all_region_records.append(_region_record(root))
        persist_region_evidence()
        replay_and_checkpoint_frontier("independent_root_lagrangian_replay")
        payload["root_lagrangian_verification"] = payload["lagrangian_replay_history"][-1]
        save()

        best_primal: dict[str, Any] | None = None
        tried_commitments: set[str] = set()
        last_tried_commitment: np.ndarray | None = None
        candidate_policy = (
            PrimalCandidatePolicy.from_config(config)
            if config.benchmark_id.endswith(("-v3", "-v4", "-v5", "-v6"))
            else None
        )
        region_attempt_policy = (
            PrimalCandidatePolicy.from_config(config, scope="disjunctive_region")
            if config.benchmark_id.endswith(("-v4", "-v5", "-v6"))
            else None
        )
        phase_one_first = config.benchmark_id.endswith("-v6")
        payload["primal_candidate_policy"] = (
            candidate_policy.as_dict() if candidate_policy is not None else None
        )
        payload["primal_candidate_queue"] = []
        payload["disjunctive_region_attempt_policy"] = (
            region_attempt_policy.as_dict()
            if region_attempt_policy is not None
            else None
        )
        payload["phase_one_first_policy"] = (
            {
                "enabled": True,
                "precheck_time_limit_seconds": float(
                    config.runtime["precheck_phase_one_time_limit_seconds"]
                ),
                "capacity_gate_is_pruning_authority": False,
                "positive_replayable_phase_one_dual_is_pruning_authority": True,
                "zero_or_uncertain_phase_one_proceeds_to_cost_lp": True,
                "phase_one_source_primal_warm_starts_cost_lp": True,
                "phase_one_row_dual_transferred": False,
            }
            if phase_one_first
            else {"enabled": False}
        )

        def try_primal(parent: SolvedRegion, proposed: np.ndarray, origin: str) -> bool:
            nonlocal best_primal, last_tried_commitment
            if len(tried_commitments) >= int(config.runtime["maximum_primal_repairs"]):
                payload["primal_candidate_queue"].append(
                    {"origin": origin, "status": "skipped_repair_limit"}
                )
                save()
                return False
            proposed_binary = np.asarray(proposed >= 0.5, dtype=np.int8)
            try:
                candidate = _capacity_repaired_commitment(
                    case,
                    source_rows,
                    proposed_binary,
                    parent.lagrangian.on_subproblem_values,
                    parent.masks,
                    parent.master.operator.total_demand_mw,
                )
            except ScopfError as exc:
                payload["primal_candidate_queue"].append(
                    {
                        "origin": origin,
                        "status": "rejected_capacity_precheck",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                save()
                return False
            digest = hashlib.sha256(candidate.tobytes()).hexdigest()
            if digest in tried_commitments:
                payload["primal_candidate_queue"].append(
                    {
                        "origin": origin,
                        "commitment_sha256": digest,
                        "status": "skipped_duplicate",
                    }
                )
                save()
                return False
            tried_commitments.add(digest)
            payload["active_stage"] = f"fixed_commitment_pdlp_primal_{origin}"
            fixed = RegionMasks(candidate == 0, candidate == 1)
            repair_started = time.perf_counter()
            capacity_changed = candidate != proposed_binary
            if last_tried_commitment is None:
                transition: dict[str, Any] = {
                    "prior_candidate_available": False,
                    "hamming_distance": None,
                    "turned_on_generator_source_rows": [],
                    "turned_off_generator_source_rows": [],
                }
            else:
                turned_on = (candidate == 1) & (last_tried_commitment == 0)
                turned_off = (candidate == 0) & (last_tried_commitment == 1)
                transition = {
                    "prior_candidate_available": True,
                    "hamming_distance": int(np.count_nonzero(candidate != last_tried_commitment)),
                    "turned_on_generator_source_rows": (source_rows[turned_on] + 1).tolist(),
                    "turned_off_generator_source_rows": (source_rows[turned_off] + 1).tolist(),
                }
            attempt: dict[str, Any] = {
                "origin": origin,
                "commitment_sha256": digest,
                "commitment_count": int(np.count_nonzero(candidate)),
                "committed_generator_source_rows": (source_rows[candidate == 1] + 1).tolist(),
                "candidate_policy": (
                    candidate_policy.as_dict() if candidate_policy is not None else None
                ),
                "capacity_repair": {
                    "proposed_commitment_count": int(np.count_nonzero(proposed_binary)),
                    "hamming_distance": int(np.count_nonzero(capacity_changed)),
                    "turned_on_generator_source_rows": (
                        source_rows[capacity_changed & (candidate == 1)] + 1
                    ).tolist(),
                    "turned_off_generator_source_rows": (
                        source_rows[capacity_changed & (candidate == 0)] + 1
                    ).tolist(),
                },
                "transition_from_prior_candidate": transition,
                "status": "running",
            }
            last_tried_commitment = candidate.copy()
            payload["primal_candidate_queue"].append(attempt)
            payload["primal_repairs"].append(attempt)
            save()

            def save_candidate_progress(record: dict[str, Any]) -> None:
                attempt["solver_progress"] = record
                save_region_progress(record)

            try:
                solved = _solve_region(
                    region_id=f"p{len(tried_commitments)}",
                    masks=fixed,
                    case=case,
                    network=network,
                    catalog=catalog,
                    config=config,
                    deadline=deadline,
                    initial_pairs=tuple(sorted(global_pairs.values())),
                    screener=screener,
                    checkpoint=save,
                    progress=save_candidate_progress,
                    candidate_policy=candidate_policy,
                )
            except DeadlineExceeded:
                attempt.update(
                    {
                        "status": "global_deadline_exceeded",
                        "wall_time_seconds": time.perf_counter() - repair_started,
                    }
                )
                save()
                raise
            except PrimalCandidateRejected as exc:
                failed_progress = payload.pop("active_region_progress", None)
                attempt.update(
                    {
                        "status": "rejected",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "wall_time_seconds": time.perf_counter() - repair_started,
                        "solver_progress": failed_progress,
                    }
                )
                save()
                return False
            payload.pop("active_region_progress", None)
            global_pairs.update((pair.pair_id, pair) for pair in solved.security_pairs)
            if solved.solve.values is None:
                raise ScopfError("Fixed-commitment PDLP lost its primal vector")
            full, full_values = reconstruct_full_values(
                case,
                network,
                solved.master,
                solved.solve.values,
                exact_commitment=candidate,
            )
            objective = float(np.asarray(full.canonical.objective) @ full_values)
            candidate_payload = {
                "case_name": config.case_name,
                "objective": objective,
                "solution": serialize_solution(full_values, case, network, full),
            }
            canonical_residual = full.canonical.max_row_violation(full_values) / case.base_mva
            attempt.update(
                {
                    "status": "secure_fixed_commitment_pdlp",
                    "objective": objective,
                    "canonical_model_residual_pu": canonical_residual,
                    "constraint_generation_rounds": solved.rounds,
                    "final_screen": solved.final_screen,
                    "wall_time_seconds": time.perf_counter() - repair_started,
                }
            )
            if canonical_residual > float(config.model["model_residual_tolerance_pu"]):
                attempt["status"] = "rejected_model_residual"
                save()
                return False
            if best_primal is None or objective < float(best_primal["objective"]):
                prices = bus_prices_from_coupling_duals(solved.master, solved.canonical_row_dual)
                pricing = {
                    "status": "gpu_pdlp_fixed_commitment_dual",
                    "definition": (
                        "Demand-derivative dual prices from the final secure "
                        "fixed-commitment cuOpt PDLP; not MILP duals"
                    ),
                    "bus_prices": [
                        {
                            "bus": int(bus),
                            "price_per_mwh": float(price),
                            "price_per_pu_hour": float(price * case.base_mva),
                        }
                        for bus, price in zip(
                            network.bus_ids, prices, strict=True
                        )
                    ],
                }
                best_primal = {
                    **candidate_payload,
                    "commitment": candidate,
                    "fixed_region": solved,
                    "bus_prices": prices,
                    "pricing": pricing,
                }
                payload["solution"] = candidate_payload["solution"]
                payload["objective"] = objective
                payload["commitment_count"] = int(np.count_nonzero(candidate))
                payload["pricing"] = pricing
                payload["secure_incumbent_checkpoint"] = {
                    "origin": origin,
                    "commitment_sha256": digest,
                    "objective": objective,
                    "commitment_count": int(np.count_nonzero(candidate)),
                    "solution": candidate_payload["solution"],
                    "pricing": pricing,
                    "status": "serialized_pending_independent_verification",
                }
                save()
                incumbent_verification = verify_serialized_solution(config, payload)
                verification_record = incumbent_verification.as_dict()
                payload["secure_incumbent_checkpoint"]["verification"] = (
                    verification_record
                )
                payload["secure_incumbent_checkpoint"]["status"] = (
                    "independently_verified"
                    if incumbent_verification.passed
                    else "failed_independent_verification"
                )
                payload["secure_incumbent_checkpoint_history"].append(
                    payload["secure_incumbent_checkpoint"]
                )
                attempt["immediate_independent_verification"] = verification_record
                save()
                if not incumbent_verification.passed:
                    raise ScopfError(
                        "Secure incumbent failed immediate independent verification"
                    )
            save()
            return True

        initial_candidates = (
            (
                np.rint(root.commitment),
                "root_pdlp_rounding",
            ),
            (
                np.asarray(
                    root.gpu_lagrangian["best_minimizing_commitment"],
                    dtype=np.float64,
                ),
                "root_gpu_lagrangian_minimizer",
            ),
            (
                np.asarray(root.commitment >= 0.25, dtype=np.float64),
                "root_pdlp_threshold_0p25",
            ),
            (
                np.ones(source_rows.size, dtype=np.float64),
                "all_online_fallback",
            ),
        )
        for proposed, origin in initial_candidates:
            if best_primal is not None:
                break
            try_primal(root, proposed, origin)

        target_gap = float(config.model["mip_relative_gap_tolerance"])
        maximum_regions = int(config.runtime["maximum_frontier_regions"])
        failed_split_positions: dict[str, set[int]] = {}
        split_attempt_number = 0
        while True:
            if best_primal is None:
                gap = None
            else:
                lower_bound = min(
                    region.lagrangian.conservative_lower_bound for region in frontier.values()
                )
                upper_bound = float(best_primal["objective"])
                gap = _relative_gap(objective=upper_bound, lower_bound=lower_bound)
                payload["objective"] = upper_bound
                payload["bound"] = lower_bound
                payload["relative_gap"] = gap
                if gap <= target_gap:
                    break
            if len(frontier) >= maximum_regions:
                payload["status"] = "incomplete_frontier_region_limit"
                break
            deadline.require("disjunctive refinement")
            parent = None
            split_position = None
            for candidate_parent in sorted(
                frontier.values(),
                key=lambda region: (
                    region.lagrangian.conservative_lower_bound,
                    region.region_id,
                ),
            ):
                try:
                    candidate_position = choose_split_generator(
                        candidate_parent.commitment,
                        candidate_parent.lagrangian.on_subproblem_values,
                        candidate_parent.masks,
                        excluded_positions=failed_split_positions.get(
                            candidate_parent.region_id, set()
                        ),
                    )
                except ScopfError:
                    continue
                parent = candidate_parent
                split_position = candidate_position
                break
            if parent is None or split_position is None:
                payload["status"] = "incomplete_no_remaining_split_generator"
                break
            split_attempt_number += 1
            off_masks, on_masks = parent.masks.split(split_position)
            if region_attempt_policy is None:
                off_id = f"{parent.region_id}0"
                on_id = f"{parent.region_id}1"
            else:
                off_id = f"{parent.region_id}_s{split_attempt_number:03d}_0"
                on_id = f"{parent.region_id}_s{split_attempt_number:03d}_1"
            split_record = {
                "parent_region_id": parent.region_id,
                "off_child_region_id": off_id,
                "on_child_region_id": on_id,
                "generator_position": split_position,
                "generator_source_row": int(source_rows[split_position]) + 1,
                "transaction_attempt": split_attempt_number,
            }
            payload["pending_disjunctive_split"] = {
                **split_record,
                "status": "solving_children_parent_certificate_retained",
                "completed_child_region_ids": [],
            }
            save()
            solved_children: dict[str, SolvedRegion] = {}
            pruned_children: dict[str, dict[str, Any]] = {}
            tentative_outcomes: list[dict[str, Any]] = []
            split_failed = False
            for child_id, child_masks in ((off_id, off_masks), (on_id, on_masks)):
                child_initial_pairs = tuple(sorted(global_pairs.values()))
                prepared_master: ReducedMaster | None = None
                phase_warm_start: np.ndarray | None = None
                warm_start_origin: str | None = None
                if phase_one_first:
                    payload["active_stage"] = f"phase_one_precheck_{child_id}"
                    prepared_master = _prepare_region_master(
                        case=case,
                        network=network,
                        config=config,
                        masks=child_masks,
                        initial_pairs=child_initial_pairs,
                    )
                    _validate_prepared_region_master(
                        prepared_master, child_masks, child_initial_pairs
                    )
                    capacity_gate = _region_pmin_pmax_capacity_gate(
                        case=case,
                        master=prepared_master,
                        masks=child_masks,
                        tolerance_pu=float(
                            config.model["model_residual_tolerance_pu"]
                        ),
                    )
                    precheck_rejection = RegionAttemptRejected(
                        f"Region {child_id} entered the registered Phase-I-first gate",
                        reason="phase_one_first_precheck",
                        master=prepared_master,
                        security_pairs=child_initial_pairs,
                        rounds=[],
                    )
                    precheck = _run_phase_one_attempt(
                        region_id=child_id,
                        masks=child_masks,
                        rejected=precheck_rejection,
                        case=case,
                        config=config,
                        deadline=deadline,
                        attempt_kind="pre_cost_lp",
                        time_limit_seconds=float(
                            config.runtime["precheck_phase_one_time_limit_seconds"]
                        ),
                        capacity_gate=capacity_gate,
                    )
                    precheck_record = precheck.record
                    payload["phase_one_prechecks"].append(
                        _phase_one_attempt_summary(precheck_record)
                    )
                    if precheck_record["prune_certified"]:
                        pruned_children[child_id] = precheck_record
                        tentative_outcomes.append(
                            {
                                "region_id": child_id,
                                "status": "phase_one_first_pruned",
                                "phase_one": precheck_record,
                            }
                        )
                        payload["pending_disjunctive_split"][
                            "completed_child_region_ids"
                        ].append(child_id)
                        payload["pending_disjunctive_split"]["tentative_outcomes"] = (
                            tentative_outcomes
                        )
                        save()
                        continue
                    phase_warm_start = precheck.source_native_primal
                    if phase_warm_start is not None:
                        warm_start_origin = "phase_one_zero_violation_primal_v1"
                    save()

                payload["active_stage"] = f"disjunctive_region_{child_id}"
                try:
                    child = _solve_region(
                        region_id=child_id,
                        masks=child_masks,
                        case=case,
                        network=network,
                        catalog=catalog,
                        config=config,
                        deadline=deadline,
                        initial_pairs=child_initial_pairs,
                        screener=screener,
                        checkpoint=save,
                        progress=save_region_progress,
                        candidate_policy=region_attempt_policy,
                        prepared_master=prepared_master,
                        initial_native_primal=phase_warm_start,
                        initial_warm_start_origin=warm_start_origin,
                    )
                except RegionAttemptRejected as rejected:
                    payload.pop("active_region_progress", None)
                    global_pairs.update(
                        (pair.pair_id, pair) for pair in rejected.security_pairs
                    )
                    if region_attempt_policy is None:
                        raise
                    payload["active_stage"] = f"phase_one_{child_id}"
                    phase_result = _run_phase_one_attempt(
                        region_id=child_id,
                        masks=child_masks,
                        rejected=rejected,
                        case=case,
                        config=config,
                        deadline=deadline,
                    )
                    phase_record = phase_result.record
                    payload["phase_one_fallback_attempts"].append(
                        _phase_one_attempt_summary(phase_record)
                    )
                    tentative_outcomes.append(
                        {
                            "region_id": child_id,
                            "status": (
                                "phase_one_pruned"
                                if phase_record["prune_certified"]
                                else "phase_one_not_certified"
                            ),
                            "phase_one": phase_record,
                        }
                    )
                    payload["pending_disjunctive_split"]["tentative_outcomes"] = (
                        tentative_outcomes
                    )
                    if phase_record["prune_certified"]:
                        pruned_children[child_id] = phase_record
                        payload["pending_disjunctive_split"][
                            "completed_child_region_ids"
                        ].append(child_id)
                        save()
                        continue
                    split_failed = True
                    break
                payload.pop("active_region_progress", None)
                solved_children[child_id] = child
                global_pairs.update((pair.pair_id, pair) for pair in child.security_pairs)
                tentative_outcomes.append(
                    {
                        "region_id": child_id,
                        "status": "solved",
                        "region": _region_record(child),
                    }
                )
                payload["pending_disjunctive_split"]["tentative_outcomes"] = (
                    tentative_outcomes
                )
                payload["pending_disjunctive_split"]["completed_child_region_ids"].append(child_id)
                save()
            if split_failed:
                failed_split_positions.setdefault(parent.region_id, set()).add(
                    split_position
                )
                payload["failed_disjunctive_split_attempts"].append(
                    {
                        **split_record,
                        "status": "rolled_back_phase_one_not_certified",
                        "parent_certificate_retained": True,
                        "outcomes": tentative_outcomes,
                    }
                )
                payload.pop("pending_disjunctive_split", None)
                persist_region_evidence()
                if len(payload["failed_disjunctive_split_attempts"]) >= int(
                    config.runtime["maximum_failed_split_attempts"]
                ):
                    payload["status"] = "incomplete_failed_split_attempt_limit"
                    break
                continue
            del frontier[parent.region_id]
            for child_id, child in solved_children.items():
                frontier[child_id] = child
                all_region_records.append(_region_record(child))
            payload["pruned_regions"].extend(
                pruned_children[child_id] for child_id in sorted(pruned_children)
            )
            if not frontier:
                raise ScopfError(
                    "All active regions were Phase-I pruned despite a secure incumbent"
                )
            payload["disjunctive_splits"].append(split_record)
            payload.pop("pending_disjunctive_split", None)
            persist_region_evidence()
            replay_and_checkpoint_frontier(f"independent_frontier_replay_after_{parent.region_id}")
            for child_id in (off_id, on_id):
                if best_primal is not None:
                    break
                child = solved_children.get(child_id)
                if child is None:
                    continue
                try_primal(child, np.rint(child.commitment), f"region_{child_id}_rounding")

        if str(payload.get("status", "")).startswith("incomplete_"):
            payload["refinement_stop_status"] = payload["status"]
        payload["all_solved_region_count"] = len(all_region_records)
        payload["solved_region_history"] = all_region_records
        payload["frontier_regions"] = [
            _region_record(frontier[region_id]) for region_id in sorted(frontier)
        ]
        if best_primal is None:
            payload["status"] = "incomplete_no_secure_primal"
            save()
            return payload
        payload["solution"] = best_primal["solution"]
        payload["objective"] = float(best_primal["objective"])
        payload["commitment_count"] = int(np.count_nonzero(best_primal["commitment"]))
        payload["pricing"] = best_primal["pricing"]
        payload["bound"] = min(
            region.lagrangian.conservative_lower_bound for region in frontier.values()
        )
        payload["relative_gap"] = _relative_gap(
            objective=float(payload["objective"]),
            lower_bound=float(payload["bound"]),
        )
        save()

        deadline.require(
            "independent primal and Lagrangian verification",
            reserve_seconds=float(config.runtime["serialization_reserve_seconds"]),
        )
        payload["active_stage"] = "independent_verification"
        verification_started = time.perf_counter()
        primal_verification = verify_serialized_solution(config, payload)
        payload["verification"] = primal_verification.as_dict()
        lagrangian_verification = verify_lagrangian_certificate_payload(config, payload)
        payload["lagrangian_verification"] = lagrangian_verification
        payload["timings_seconds"]["independent_verification"] = (
            time.perf_counter() - verification_started
        )
        relaxation_rounds = [
            round_record
            for region_record in all_region_records
            for round_record in region_record["constraint_generation_rounds"]
        ]
        primal_rounds = [
            round_record
            for repair in payload["primal_repairs"]
            for round_record in repair.get("constraint_generation_rounds", [])
        ]
        payload["timings_seconds"].update(
            {
                "relaxation_pdlp_adapter_wall": sum(
                    float(record["adapter_wall_time_seconds"]) for record in relaxation_rounds
                ),
                "relaxation_cupy_screening_wall": sum(
                    float(record.get("screen", {}).get("wall_time_seconds", 0.0))
                    for record in relaxation_rounds
                ),
                "fixed_commitment_pdlp_adapter_wall": sum(
                    float(record["adapter_wall_time_seconds"]) for record in primal_rounds
                ),
                "fixed_commitment_cupy_screening_wall": sum(
                    float(record.get("screen", {}).get("wall_time_seconds", 0.0))
                    for record in primal_rounds
                ),
                "cupy_lagrangian_evaluation_wall": sum(
                    float(record["gpu_lagrangian_evaluation"]["wall_time_seconds"])
                    for record in all_region_records
                ),
                "phase_one_precheck_pdlp_adapter_wall": sum(
                    float(record.get("adapter_wall_time_seconds", 0.0))
                    for record in payload.get("phase_one_prechecks", [])
                ),
                "phase_one_fallback_pdlp_adapter_wall": sum(
                    float(record.get("adapter_wall_time_seconds", 0.0))
                    for record in payload.get("phase_one_fallback_attempts", [])
                ),
                "phase_one_pdlp_adapter_wall": sum(
                    float(record.get("adapter_wall_time_seconds", 0.0))
                    for records in (
                        payload.get("phase_one_prechecks", []),
                        payload.get("phase_one_fallback_attempts", []),
                    )
                    for record in records
                ),
            }
        )
        payload["phase_one_first_outcomes"] = {
            "precheck_attempt_count": len(payload.get("phase_one_prechecks", [])),
            "cost_lp_attempts_avoided_by_certified_precheck": sum(
                int(bool(record.get("prune_certified")))
                for record in payload.get("phase_one_prechecks", [])
            ),
            "eligible_cost_lp_primal_warm_starts": sum(
                int(
                    bool(
                        record.get("source_feasible_warm_start", {}).get(
                            "eligible", False
                        )
                    )
                )
                for record in payload.get("phase_one_prechecks", [])
                if not bool(record.get("prune_certified"))
            ),
            "fallback_phase_one_attempt_count": len(
                payload.get("phase_one_fallback_attempts", [])
            ),
        }
        gap_passed = payload["relative_gap"] <= target_gap * (1.0 + 1e-9) + 1e-12
        payload["acceptance_gates"] = {
            "secure_primal_independently_verified": primal_verification.passed,
            "disjunctive_lagrangian_bound_independently_replayed": (
                lagrangian_verification["passed"]
            ),
            "requested_relative_gap": target_gap,
            "relative_gap": payload["relative_gap"],
            "gap_certified": gap_passed,
            "integer_solver_absent": True,
            "branch_and_bound_absent": True,
            "all_pruned_regions_have_replayed_phase_one_certificates": bool(
                lagrangian_verification["phase_one_pruned_region_count"]
                == len(payload.get("pruned_regions", []))
            ),
            "final_exhaustive_security_violation_pu": (
                primal_verification.maximum_security_violation_pu
            ),
        }
        if primal_verification.passed and lagrangian_verification["passed"] and gap_passed:
            payload["status"] = "optimality_gap_certified_gpu_lagrangian"
        elif not primal_verification.passed:
            payload["status"] = "failed_independent_primal_verification"
        elif not lagrangian_verification["passed"]:
            payload["status"] = "failed_independent_lagrangian_verification"
        else:
            payload["status"] = payload.get(
                "refinement_stop_status",
                "incomplete_requested_gap_not_certified",
            )
        cpu = payload["cpu_comparison"]
        payload["system_comparison"] = {
            "comparison_kind": "system_to_system_not_pure_gpu_speedup",
            "laptop_cpu_end_to_end_seconds": cpu["total_wall_time_seconds"],
            "dgx_spark_end_to_end_seconds": None,
            "laptop_cpu_objective": cpu["objective"],
            "dgx_spark_objective": payload["objective"],
            "laptop_cpu_gap": cpu["mip_gap"],
            "dgx_spark_gap": payload["relative_gap"],
        }
        save()

    payload["active_stage"] = "complete"
    payload["elapsed_seconds"] = deadline.elapsed
    return payload


def run_one_shot_gpu_lagrangian_experiment(
    config: RunConfig, *, output_path: Path
) -> dict[str, Any]:
    registration = validate_lagrangian_experiment_config(config)
    validate_platform(config, "dgx_spark")
    output = guard_output_path(output_path)
    experiment_root = (config.root / "results" / "experiments").resolve()
    if not output.resolve().is_relative_to(experiment_root):
        raise ScopfError("GPU Lagrangian output must be under results/experiments")
    if output.exists():
        raise ScopfError(f"GPU Lagrangian output already exists: {output}")
    identity = frozen_identity(config)
    suite_id = str(registration["benchmark"]["experiment_suite_id"])
    if suite_id != config.benchmark_id:
        raise ScopfError("GPU Lagrangian suite id must match its benchmark id")
    registry_path = guard_output_path(experiment_root / f"{suite_id}-run-registry.json")
    checkpoint_path = guard_output_path(
        config.root / "results" / "checkpoints" / f"{suite_id}-dgx_spark.json"
    )
    console_path = guard_output_path(
        config.root / "results" / "diagnostics" / f"{suite_id}-worker-console.log"
    )
    for path in (registry_path, checkpoint_path, console_path):
        if path.exists():
            raise ScopfError(f"GPU Lagrangian one-shot path already exists: {path}")
    registry = {
        "schema_version": "1.0.0",
        "experiment_suite_id": registration["benchmark"]["experiment_suite_id"],
        "runs": {
            "dgx_spark": {
                "status": "started",
                "started_at_utc": datetime.now(UTC).isoformat(),
                "host": platform.node(),
                "output": str(output),
                "checkpoint": str(checkpoint_path),
                "frozen_identity": identity,
            }
        },
    }
    write_json_atomic(registry, registry_path)
    command = [
        sys.executable,
        "-m",
        "activsg_scopf.cli",
        "_gpu_lagrangian_worker",
        "--config",
        str(config.path),
        "--output",
        str(output),
        "--checkpoint",
        str(checkpoint_path),
    ]
    deadline_seconds = float(config.runtime["deadline_seconds"])
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=config.root,
            check=False,
            capture_output=True,
            text=True,
            timeout=deadline_seconds,
        )
        wall = time.perf_counter() - started
        _write_console(console_path, completed.stdout, completed.stderr)
        result = (
            _read_json(output)
            if output.exists()
            else _read_json(checkpoint_path)
            if checkpoint_path.exists()
            else {
                "status": "failed_worker_without_result",
                "worker_returncode": completed.returncode,
                "worker_stderr": completed.stderr[-4000:],
            }
        )
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - started
        _write_console(console_path, exc.stdout, exc.stderr)
        result = _read_json(checkpoint_path) if checkpoint_path.exists() else {}
        result.update(
            {"status": "hard_deadline_exceeded", "worker_timeout_seconds": deadline_seconds}
        )
    result.update(
        {
            "official": False,
            "experiment": True,
            "one_shot": True,
            "frozen_identity": identity,
            "worker_console_log": str(console_path.relative_to(config.root)),
            "total_wall_time_seconds": wall,
            "benchmark_boundary": (
                "worker launch through raw loading, GPU PDLP primal/relaxation rounds, "
                "CuPy screening and Lagrangian evaluation, disjunctive refinement, "
                "independent CPU replay/verification, and result serialization"
            ),
        }
    )
    if "system_comparison" in result:
        result["system_comparison"]["dgx_spark_end_to_end_seconds"] = wall
    write_json_atomic(result, output)
    registry["runs"]["dgx_spark"].update(
        {
            "status": result.get("status"),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "total_wall_time_seconds": wall,
            "objective": result.get("objective"),
            "bound": result.get("bound"),
            "relative_gap": result.get("relative_gap"),
        }
    )
    write_json_atomic(registry, registry_path)
    return result


def run_gpu_lagrangian_worker_serialized(
    config: RunConfig, *, output_path: Path, checkpoint_path: Path
) -> dict[str, Any]:
    output = guard_output_path(output_path)
    checkpoint_file = guard_output_path(checkpoint_path)

    def checkpoint(payload: dict[str, Any]) -> None:
        write_json_atomic(payload, checkpoint_file)

    worker_started = time.perf_counter()
    try:
        result = run_gpu_lagrangian_experiment(config, checkpoint=checkpoint)
    except DeadlineExceeded as exc:
        result = _read_json(checkpoint_file) if checkpoint_file.exists() else {}
        result.update(
            {
                "status": "deadline_budget_exhausted",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    except Exception as exc:
        result = _read_json(checkpoint_file) if checkpoint_file.exists() else {}
        result.update(
            {
                "status": "failed_exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    result["elapsed_seconds"] = max(
        float(result.get("elapsed_seconds", 0.0)),
        time.perf_counter() - worker_started,
    )
    serialization_started = time.perf_counter()
    write_json_atomic(result, output)
    result.setdefault("timings_seconds", {})["result_serialization"] = (
        time.perf_counter() - serialization_started
    )
    write_json_atomic(result, output)
    return result
