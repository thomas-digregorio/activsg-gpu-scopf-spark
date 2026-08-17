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

    @classmethod
    def from_config(cls, config: RunConfig) -> PrimalCandidatePolicy:
        runtime = config.runtime
        return cls(
            total_seconds=float(runtime["maximum_primal_candidate_seconds"]),
            maximum_round_seconds=float(runtime["maximum_primal_candidate_round_seconds"]),
            minimum_round_seconds=float(runtime["minimum_primal_candidate_round_seconds"]),
            stagnation_window_rounds=int(runtime["primal_candidate_stagnation_window_rounds"]),
            minimum_relative_residual_improvement=float(
                runtime["primal_candidate_minimum_relative_residual_improvement"]
            ),
            dual_divergence_multiple=float(runtime["primal_candidate_dual_divergence_multiple"]),
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_seconds": self.total_seconds,
            "maximum_round_seconds": self.maximum_round_seconds,
            "minimum_round_seconds": self.minimum_round_seconds,
            "stagnation_window_rounds": self.stagnation_window_rounds,
            "minimum_relative_residual_improvement": (self.minimum_relative_residual_improvement),
            "dual_divergence_multiple": self.dual_divergence_multiple,
        }


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
        config.benchmark_id.endswith(("-v2", "-v3"))
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
) -> SolvedRegion:
    region_started = time.perf_counter()
    profile = config.raw["platforms"]["dgx_spark"]
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
    pairs_by_id = {pair.pair_id: pair for pair in initial_pairs}
    native_primal: np.ndarray | None = None
    native_dual: np.ndarray | None = None
    rounds: list[dict[str, Any]] = []
    final_screen: dict[str, Any] | None = None
    last_solve: ContinuousSolveResult | None = None
    usable_primal_seen = False
    infeasible_residuals: list[float] = []

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
                raise PrimalCandidateRejected(
                    f"Region {region_id} exhausted its {candidate_policy.total_seconds:g}s "
                    "primal-candidate budget"
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
        }
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
                round_record["candidate_gate"] = {
                    "rejected": True,
                    "reason": "solver_returned_no_usable_primal_vectors",
                    "solve_status": last_solve.status,
                }
                emit_progress()
                raise PrimalCandidateRejected(
                    f"Region {region_id} rejected candidate after round "
                    f"{round_number}: solver returned no usable primal vectors "
                    f"with status={last_solve.status}"
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
                    gate["rejected"] = True
                    gate["reason"] = rejection_reason
                    round_record["candidate_gate"] = gate
                    emit_progress()
                    raise PrimalCandidateRejected(
                        f"Region {region_id} rejected candidate after round "
                        f"{round_number}: {rejection_reason}"
                    )
                round_record["candidate_gate"] = gate
            if last_solve.status == "TimeLimit" and native_primal is not None:
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
                raise PrimalCandidateRejected(
                    f"Region {region_id} rejected candidate after round "
                    f"{round_number}: status={last_solve.status}, "
                    f"residual_pu={canonical_residual_pu:.6e}"
                )
            raise ScopfError(
                f"Region {region_id} PDLP primal is unusable: "
                f"status={last_solve.status}, residual_pu={canonical_residual_pu:.6e}"
            )
        usable_primal_seen = True
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
            security_pair_from_record(pair_record, catalog)
            for pair_record in record["security_pairs"]
        )
        add_reduced_security_pairs(master, network, pairs)
        if record.get("coefficient_cleanup_audit") != master.coefficient_cleanup_audit:
            raise ScopfError(f"Independent region {region_id} coefficient-cleanup audit mismatch")
        replayed = replay_lagrangian_certificate(master, record["lagrangian_certificate"], masks)
        recorded = float(record["lagrangian_certificate"]["conservative_lower_bound"])
        difference = abs(replayed.conservative_lower_bound - recorded)
        maximum_difference = max(maximum_difference, difference)
        if difference > float(config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]):
            raise ScopfError(f"Independent region {region_id} replay mismatch")
        leaves[region_id] = masks
        replayed_bounds[region_id] = replayed.conservative_lower_bound
    cover_passed = verify_disjunctive_cover(source_rows.size, payload["disjunctive_splits"], leaves)
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
        "disjunctive_cover_passed": cover_passed,
        "replayed_global_lower_bound": global_bound,
        "recorded_global_lower_bound": recorded_global,
        "global_bound_difference_dollars": global_difference,
        "maximum_region_replay_difference_dollars": maximum_difference,
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
        "frontier_regions": [],
        "primal_repairs": [],
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
            save()

        def replay_and_checkpoint_frontier(stage: str) -> None:
            payload["active_stage"] = stage
            replay = verify_lagrangian_certificate_payload(config, payload)
            if not replay["passed"]:
                raise ScopfError(f"Independent Lagrangian replay failed during {stage}")
            payload["bound"] = replay["replayed_global_lower_bound"]
            payload["bound_status"] = "independently_replayed_current_frontier"
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
            if config.benchmark_id.endswith("-v3")
            else None
        )
        payload["primal_candidate_policy"] = (
            candidate_policy.as_dict() if candidate_policy is not None else None
        )
        payload["primal_candidate_queue"] = []

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
                best_primal = {
                    **candidate_payload,
                    "commitment": candidate,
                    "fixed_region": solved,
                    "bus_prices": prices,
                }
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
        while True:
            if best_primal is None:
                gap = None
            else:
                lower_bound = min(
                    region.lagrangian.conservative_lower_bound for region in frontier.values()
                )
                upper_bound = float(best_primal["objective"])
                gap = (upper_bound - lower_bound) / max(1.0, abs(upper_bound))
                payload["objective"] = upper_bound
                payload["bound"] = lower_bound
                payload["relative_gap"] = gap
                if gap < -1e-9:
                    raise ScopfError("Lagrangian lower bound exceeds the secure incumbent")
                if gap <= target_gap:
                    break
            if len(frontier) >= maximum_regions:
                payload["status"] = "incomplete_frontier_region_limit"
                break
            deadline.require("disjunctive refinement")
            parent = min(
                frontier.values(),
                key=lambda region: region.lagrangian.conservative_lower_bound,
            )
            split_position = choose_split_generator(
                parent.commitment,
                parent.lagrangian.on_subproblem_values,
                parent.masks,
            )
            off_masks, on_masks = parent.masks.split(split_position)
            off_id = f"{parent.region_id}0"
            on_id = f"{parent.region_id}1"
            split_record = {
                "parent_region_id": parent.region_id,
                "off_child_region_id": off_id,
                "on_child_region_id": on_id,
                "generator_position": split_position,
                "generator_source_row": int(source_rows[split_position]) + 1,
            }
            payload["pending_disjunctive_split"] = {
                **split_record,
                "status": "solving_children_parent_certificate_retained",
                "completed_child_region_ids": [],
            }
            save()
            solved_children: dict[str, SolvedRegion] = {}
            for child_id, child_masks in ((off_id, off_masks), (on_id, on_masks)):
                payload["active_stage"] = f"disjunctive_region_{child_id}"
                child = _solve_region(
                    region_id=child_id,
                    masks=child_masks,
                    case=case,
                    network=network,
                    catalog=catalog,
                    config=config,
                    deadline=deadline,
                    initial_pairs=tuple(sorted(global_pairs.values())),
                    screener=screener,
                    checkpoint=save,
                    progress=save_region_progress,
                )
                payload.pop("active_region_progress", None)
                solved_children[child_id] = child
                global_pairs.update((pair.pair_id, pair) for pair in child.security_pairs)
                payload["pending_disjunctive_split"]["completed_child_region_ids"].append(child_id)
                save()
            del frontier[parent.region_id]
            for child_id in (off_id, on_id):
                child = solved_children[child_id]
                frontier[child_id] = child
                all_region_records.append(_region_record(child))
            payload["disjunctive_splits"].append(split_record)
            payload.pop("pending_disjunctive_split", None)
            persist_region_evidence()
            replay_and_checkpoint_frontier(f"independent_frontier_replay_after_{parent.region_id}")
            for child_id in (off_id, on_id):
                if best_primal is not None:
                    break
                child = solved_children[child_id]
                try_primal(child, np.rint(child.commitment), f"region_{child_id}_rounding")

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
        payload["pricing"] = {
            "status": "gpu_pdlp_fixed_commitment_dual",
            "definition": (
                "Demand-derivative dual prices from the final secure fixed-commitment "
                "cuOpt PDLP; not MILP duals"
            ),
            "bus_prices": [
                {
                    "bus": int(bus),
                    "price_per_mwh": float(price),
                    "price_per_pu_hour": float(price * case.base_mva),
                }
                for bus, price in zip(network.bus_ids, best_primal["bus_prices"], strict=True)
            ],
        }
        payload["bound"] = min(
            region.lagrangian.conservative_lower_bound for region in frontier.values()
        )
        payload["relative_gap"] = (float(payload["objective"]) - float(payload["bound"])) / max(
            1.0, abs(float(payload["objective"]))
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
                    float(record["screen"]["wall_time_seconds"]) for record in relaxation_rounds
                ),
                "fixed_commitment_pdlp_adapter_wall": sum(
                    float(record["adapter_wall_time_seconds"]) for record in primal_rounds
                ),
                "fixed_commitment_cupy_screening_wall": sum(
                    float(record["screen"]["wall_time_seconds"]) for record in primal_rounds
                ),
                "cupy_lagrangian_evaluation_wall": sum(
                    float(record["gpu_lagrangian_evaluation"]["wall_time_seconds"])
                    for record in all_region_records
                ),
            }
        )
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
            payload["status"] = "incomplete_requested_gap_not_certified"
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
