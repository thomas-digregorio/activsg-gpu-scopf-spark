"""One-shot exhaustive GPU LP-relaxation lower-bound experiment."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig
from .deadline import Deadline, PeakMemorySampler
from .environment import environment_manifest, validate_platform
from .errors import DeadlineExceeded, ProvenanceError, ScopfError
from .matpower import read_contingency_table, read_matpower_case
from .model import build_master
from .network import (
    build_contingency_catalog,
    build_network,
    contingency_catalog_report,
)
from .official import frozen_identity
from .paths import guard_output_path, guard_runtime_environment
from .provenance import build_source_manifest, write_json_atomic
from .screening import ContingencyScreener, add_security_pairs
from .solution import serialize_solution
from .solvers.cuopt_lp import (
    ContinuousSolveResult,
    derive_rate_a_angle_bounds,
    solve_cuopt_continuous_pdlp,
)
from .verify import verify_serialized_solution

LP_CERTIFICATE_KIND = "gpu_lp_relaxation_certificate"
LP_CERTIFICATE_POLICY = "full_fractional_n_minus_1_pdlp_fp64_v3"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read LP-certificate evidence {path}: {exc}") from exc


def _write_console(path: Path, stdout: str | bytes | None, stderr: str | bytes | None) -> None:
    def decode(value: str | bytes | None) -> str:
        return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(decode(stdout) + decode(stderr), encoding="utf-8")


def validate_lp_certificate_config(config: RunConfig) -> dict[str, Any]:
    """Fail closed if the registered continuous experiment was altered."""

    registered_identities = {
        "activsg2000-gpu-lp-certificate-v12": "experiment-2000-gpu-lp-certificate-v12",
        "activsg2000-gpu-lp-certificate-v13": "experiment-2000-gpu-lp-certificate-v13",
        "activsg2000-gpu-lp-certificate-v14": "experiment-2000-gpu-lp-certificate-v14",
        "activsg2000-gpu-lp-certificate-v15": "experiment-2000-gpu-lp-certificate-v15",
    }
    required_tag = registered_identities.get(config.benchmark_id)
    if required_tag is None:
        raise ScopfError(f"Unregistered GPU LP-certificate benchmark id: {config.benchmark_id!r}")
    observed_tag = str(config.raw["benchmark"].get("required_git_tag", ""))
    if observed_tag != required_tag:
        raise ScopfError(
            f"LP-certificate required tag changed: expected {required_tag!r}, "
            f"observed {observed_tag!r}"
        )
    experiment_suite_id = str(config.raw["benchmark"].get("experiment_suite_id", ""))
    if experiment_suite_id != config.benchmark_id:
        raise ScopfError(
            "LP-certificate experiment_suite_id must equal its registered benchmark id"
        )
    if config.case_name != "ACTIVSg2000":
        raise ScopfError("This LP-certificate experiment is registered only for ACTIVSg2000")
    if config.benchmark_kind != LP_CERTIFICATE_KIND:
        raise ScopfError("Configuration is not a GPU LP-relaxation certificate")
    is_v15 = config.benchmark_id == "activsg2000-gpu-lp-certificate-v15"
    required_deadline = 900.0 if is_v15 else 600.0
    if float(config.runtime.get("deadline_seconds", 0.0)) != required_deadline:
        raise ScopfError(
            "The LP-certificate end-to-end deadline must be exactly "
            f"{required_deadline:g} seconds"
        )
    required_runtime = {
        "verification_reserve_seconds": 60.0,
        "serialization_reserve_seconds": 15.0,
        "maximum_constraint_generation_rounds": 100,
    }
    if is_v15:
        required_runtime.update(
            {
                "maximum_lp_round_time_limit_seconds": 480.0,
                "followup_solve_reserve_seconds": 90.0,
                "minimum_lp_round_time_limit_seconds": 30.0,
            }
        )
    else:
        required_runtime["lp_round_time_limit_seconds"] = 120.0
    for key, expected in required_runtime.items():
        if config.runtime.get(key) != expected:
            raise ScopfError(
                f"LP-certificate runtime {key} changed: expected {expected!r}, "
                f"observed {config.runtime.get(key)!r}"
            )
    required_model = {
        "interval_hours": 1.0,
        "pwl_segments": 10,
        "mip_relative_gap_tolerance": 0.001,
        "model_residual_tolerance_pu": 1e-6,
        "security_violation_tolerance_pu": 1e-5,
        "lodf_validation_tolerance_pu": 1e-9,
        "lodf_validation_columns": 3,
        "lodf_build_chunk_columns": 256,
        "screen_chunk_columns": 256,
    }
    for key, expected in required_model.items():
        if config.model.get(key) != expected:
            raise ScopfError(
                f"LP-certificate model {key} changed: expected {expected!r}, "
                f"observed {config.model.get(key)!r}"
            )
    profile = config.raw["platforms"].get("dgx_spark", {})
    required_profile = {
        "solver": "cuopt",
        "screening": "cupy",
        "solver_threads": 0,
        "native_scaling_mode": "power_system_per_unit_v1",
        "lp_method": "pdlp",
        "pdlp_solver_mode": "stable3",
        "pdlp_precision": "fp64",
        "pdlp_optimality_tolerance": 1e-8,
        "dual_certificate_residual_tolerance": 1e-7,
    }
    if config.benchmark_id in {
        "activsg2000-gpu-lp-certificate-v14",
        "activsg2000-gpu-lp-certificate-v15",
    }:
        required_profile.update(
            {
                "per_constraint_residual": True,
                "redundant_angle_bounds": "rate_a_dc_shortest_path_v1",
            }
        )
    if is_v15:
        required_profile.update(
            {
                "presolve": 0,
                "pdlp_warm_start": "previous_primal_dual_zero_extend_rows_v1",
                "require_reported_reconstructed_dual_agreement": True,
                "recover_time_limit_vectors": True,
            }
        )
    for key, expected in required_profile.items():
        if profile.get(key) != expected:
            raise ScopfError(
                f"LP-certificate DGX profile {key} changed: expected {expected!r}, "
                f"observed {profile.get(key)!r}"
            )
    reference = config.raw["benchmark"].get("reference_incumbent", {})
    required_reference = {
        "objective": 1133047.8684341211,
        "result_sha256": "764bd04415828c02238417161fa74ff65d1cb6a32e838bb12e987a5152a4594a",
        "source_tag": "experiment-2000-gpu-commitment-trace-v11",
        "source_commit": "67d64dff6efcb62dcf38182c05b90e374a7df439",
        "independently_verified_n_minus_1": True,
    }
    for key, expected in required_reference.items():
        if reference.get(key) != expected:
            raise ScopfError(
                f"LP-certificate reference incumbent {key} changed: "
                f"expected {expected!r}, observed {reference.get(key)!r}"
            )
    return {
        "profile": profile,
        "reference_incumbent": reference,
        "experiment_suite_id": experiment_suite_id,
    }


def _solve_summary(result: ContinuousSolveResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "optimal": result.optimal,
        "primal_objective": result.primal_objective,
        "dual_objective": result.dual_objective,
        "solve_time_seconds": result.solve_time_seconds,
        "statistics": result.statistics,
    }


def allocate_lp_round_budget(
    available_solver_seconds: float,
    *,
    maximum_round_seconds: float,
    followup_reserve_seconds: float,
    minimum_round_seconds: float,
) -> float:
    """Allocate one solve while preserving a useful follow-up attempt when possible."""

    values = (
        available_solver_seconds,
        maximum_round_seconds,
        followup_reserve_seconds,
        minimum_round_seconds,
    )
    if not all(np.isfinite(value) for value in values):
        raise ScopfError("LP round-budget inputs must be finite")
    if available_solver_seconds <= 0 or maximum_round_seconds <= 0:
        raise ScopfError("LP round-budget inputs must be positive")
    if followup_reserve_seconds < 0 or minimum_round_seconds <= 0:
        raise ScopfError("LP round-budget reserves are invalid")
    if available_solver_seconds > followup_reserve_seconds + minimum_round_seconds:
        available_now = available_solver_seconds - followup_reserve_seconds
    else:
        available_now = available_solver_seconds
    return min(maximum_round_seconds, available_now)


def _fractional_commitment_summary(
    values: np.ndarray, commitment_columns: np.ndarray, *, tolerance: float
) -> dict[str, Any]:
    commitments = np.asarray(values[commitment_columns], dtype=np.float64)
    distances = np.abs(commitments - np.rint(commitments))
    fractional = distances > tolerance
    return {
        "online_generator_count": int(commitments.size),
        "fractional_commitment_count": int(np.count_nonzero(fractional)),
        "near_zero_count": int(np.count_nonzero(np.abs(commitments) <= tolerance)),
        "near_one_count": int(np.count_nonzero(np.abs(commitments - 1.0) <= tolerance)),
        "sum_fractional_commitments": float(np.sum(commitments)),
        "minimum_commitment": float(np.min(commitments)),
        "maximum_commitment": float(np.max(commitments)),
        "maximum_distance_from_binary": float(np.max(distances)),
    }


def run_lp_relaxation_certificate(
    config: RunConfig,
    *,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Solve/screen the fractional N-1 LP and compare its dual to one incumbent."""

    registration = validate_lp_certificate_config(config)
    profile = registration["profile"]
    reference = registration["reference_incumbent"]
    deadline = Deadline(
        total_seconds=float(config.runtime["deadline_seconds"]),
        verification_reserve_seconds=float(config.runtime["verification_reserve_seconds"]),
        serialization_reserve_seconds=float(config.runtime["serialization_reserve_seconds"]),
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "case_name": config.case_name,
        "benchmark_id": config.benchmark_id,
        "experiment_suite_id": registration["experiment_suite_id"],
        "experiment_policy": LP_CERTIFICATE_POLICY,
        "platform": "dgx_spark",
        "status": "running",
        "deadline_seconds": float(config.runtime["deadline_seconds"]),
        "model_class": "continuous_relaxation_of_single_hour_preventive_scuc",
        "integer_search_performed": False,
        "branch_and_bound_performed": False,
        "reference_incumbent": reference,
        "constraint_generation_rounds": [],
        "added_security_pair_ids": [],
        "timings_seconds": {},
    }
    memory_sampler = PeakMemorySampler(sample_gpu=True)

    def save_checkpoint() -> None:
        payload["elapsed_seconds"] = deadline.elapsed
        payload["peak_memory"] = {
            "process_rss_bytes": memory_sampler.peak_process_rss_bytes,
            "cupy_pool_used_bytes": memory_sampler.peak_gpu_pool_used_bytes,
            "cuda_device_memory_delta_bytes": (memory_sampler.peak_cuda_device_memory_delta_bytes),
        }
        if checkpoint is not None:
            checkpoint(payload)

    with memory_sampler:
        guard_runtime_environment(config.root)
        validate_platform(config, "dgx_spark")
        payload["environment"] = environment_manifest("dgx_spark")
        save_checkpoint()

        started = time.perf_counter()
        case = read_matpower_case(
            config.case_path,
            expected_sha256=config.raw["raw_inputs"]["case_sha256"],
        )
        contingency_table = read_contingency_table(
            config.contingency_path,
            expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
        )
        payload["timings_seconds"]["raw_input_loading"] = time.perf_counter() - started
        payload["source_manifest"] = build_source_manifest(case, contingency_table)
        save_checkpoint()

        started = time.perf_counter()
        network = build_network(case)
        catalog = build_contingency_catalog(
            case,
            network,
            contingency_table,
            validation_columns=int(config.model["lodf_validation_columns"]),
            validation_tolerance_pu=float(config.model["lodf_validation_tolerance_pu"]),
            chunk_columns=int(config.model["lodf_build_chunk_columns"]),
        )
        master = build_master(case, network, segments=int(config.model["pwl_segments"]))
        redundant_bounds = None
        redundant_bounds_policy = profile.get("redundant_angle_bounds")
        if redundant_bounds_policy is not None:
            if redundant_bounds_policy != "rate_a_dc_shortest_path_v1":
                raise ScopfError(
                    f"Unsupported redundant angle-bound policy: {redundant_bounds_policy!r}"
                )
            redundant_bounds = derive_rate_a_angle_bounds(
                network,
                master.index.theta_by_bus,
                total_columns=master.canonical.num_columns,
            )
            payload["redundant_angle_bounds"] = redundant_bounds.audit
        payload["timings_seconds"]["model_and_factor_build"] = time.perf_counter() - started
        _, _, _, original_integrality = master.canonical.column_arrays()
        payload["relaxation_identity"] = {
            "original_integer_columns": int(np.count_nonzero(original_integrality)),
            "continuous_solver_integer_columns": 0,
            "relaxed_columns": [
                master.canonical.variable_names[int(index)]
                for index in np.flatnonzero(original_integrality)
            ],
            "physical_constraints_changed": False,
            "cost_coefficients_changed": False,
            "exact_source_pmin_changed": False,
        }
        payload["contingencies"] = contingency_catalog_report(catalog)
        payload["base_model_dimensions"] = {
            "columns": master.canonical.num_columns,
            "rows": master.canonical.num_rows,
            "nonzeros": int(master.canonical.matrix_csr().nnz),
        }
        screener = ContingencyScreener(
            network,
            catalog,
            backend="cupy",
            chunk_columns=int(config.model["screen_chunk_columns"]),
        )
        save_checkpoint()

        added_pair_ids: set[str] = set()
        last_solve: ContinuousSolveResult | None = None
        final_screen_passed = False
        best_verified_bound: float | None = None
        best_verified_bound_round: int | None = None
        native_primal_start: np.ndarray | None = None
        native_row_dual_start: np.ndarray | None = None
        solve_wall_total = 0.0
        screen_wall_total = 0.0
        incumbent = float(reference["objective"])
        requested_gap = float(config.model["mip_relative_gap_tolerance"])
        required_bound = incumbent - requested_gap * abs(incumbent)
        use_dynamic_budget = (
            config.benchmark_id == "activsg2000-gpu-lp-certificate-v15"
        )
        commitment_columns = np.asarray(
            [
                master.index.commitment_by_generator[int(generator)]
                for generator in master.index.generator_source_rows
            ],
            dtype=np.int64,
        )
        for round_number in range(
            1, int(config.runtime["maximum_constraint_generation_rounds"]) + 1
        ):
            deadline.require("continuous restricted-master solve")
            available_solver_budget = deadline.solver_budget()
            if use_dynamic_budget:
                solver_budget = allocate_lp_round_budget(
                    available_solver_budget,
                    maximum_round_seconds=float(
                        config.runtime["maximum_lp_round_time_limit_seconds"]
                    ),
                    followup_reserve_seconds=float(
                        config.runtime["followup_solve_reserve_seconds"]
                    ),
                    minimum_round_seconds=float(
                        config.runtime["minimum_lp_round_time_limit_seconds"]
                    ),
                )
            else:
                solver_budget = min(
                    available_solver_budget,
                    float(config.runtime["lp_round_time_limit_seconds"]),
                )
            final_screen_passed = False
            payload["active_stage"] = "continuous_restricted_master_solve"
            payload["active_constraint_generation_round"] = round_number
            payload["active_solver_budget_seconds"] = solver_budget
            save_checkpoint()
            started = time.perf_counter()
            last_solve = solve_cuopt_continuous_pdlp(
                master.canonical,
                time_limit_seconds=solver_budget,
                optimality_tolerance=float(profile["pdlp_optimality_tolerance"]),
                primal_feasibility_tolerance=float(config.model["model_residual_tolerance_pu"]),
                certificate_residual_tolerance=float(
                    profile["dual_certificate_residual_tolerance"]
                ),
                native_scaling_mode=str(profile["native_scaling_mode"]),
                native_base_mva=float(case.base_mva),
                log_to_console=True,
                per_constraint_residual=bool(profile.get("per_constraint_residual", False)),
                redundant_bounds=redundant_bounds,
                presolve=int(profile.get("presolve", -1)),
                initial_native_primal=(native_primal_start if use_dynamic_budget else None),
                initial_native_row_dual=(
                    native_row_dual_start if use_dynamic_budget else None
                ),
            )
            solve_wall = time.perf_counter() - started
            solve_wall_total += solve_wall
            round_payload: dict[str, Any] = {
                "round": round_number,
                "rows_before_solve": master.canonical.num_rows,
                "available_solver_budget_seconds": available_solver_budget,
                "solver_budget_seconds": solver_budget,
                "adapter_wall_time_seconds": solve_wall,
                "solve": _solve_summary(last_solve),
            }
            payload["constraint_generation_rounds"].append(round_payload)
            payload["objective"] = last_solve.primal_objective
            dual_certificate = last_solve.statistics.get("dual_certificate", {})
            dual_passed = bool(dual_certificate.get("passed", False))
            primal_feasible = bool(dual_certificate.get("primal_feasible", False))
            conservative_bound = dual_certificate.get("conservative_numerical_lower_bound")
            payload["dual_objective_candidate"] = last_solve.dual_objective
            if dual_passed and conservative_bound is not None:
                candidate_bound = float(conservative_bound)
                if best_verified_bound is None or candidate_bound > best_verified_bound:
                    best_verified_bound = candidate_bound
                    best_verified_bound_round = round_number
            payload["bound"] = best_verified_bound
            payload["best_verified_bound_round"] = best_verified_bound_round
            if use_dynamic_budget:
                native_primal_start = last_solve.native_primal
                native_row_dual_start = last_solve.native_row_dual
            round_payload["acceptance_progress"] = {
                "primal_feasible": primal_feasible,
                "dual_certificate_passed": dual_passed,
                "best_verified_bound": best_verified_bound,
                "required_reference_bound": required_bound,
                "reference_gap_bound_ready": bool(
                    best_verified_bound is not None
                    and best_verified_bound >= required_bound
                ),
            }
            error_status = str(last_solve.statistics.get("error_status", ""))
            recoverable_status = last_solve.status in {
                "Optimal",
                "FeasibleFound",
                "TimeLimit",
            }
            if error_status != "Success" or not recoverable_status:
                payload["status"] = "failed_continuous_solver_status"
                save_checkpoint()
                break
            if (
                last_solve.values is None
                or last_solve.primal_objective is None
                or not primal_feasible
            ):
                payload["status"] = "incomplete_continuous_primal_not_feasible"
                save_checkpoint()
                if use_dynamic_budget and last_solve.status == "TimeLimit":
                    continue
                break
            round_payload["fractional_commitment"] = _fractional_commitment_summary(
                last_solve.values,
                commitment_columns,
                tolerance=float(config.model["model_residual_tolerance_pu"]),
            )
            payload["solution"] = serialize_solution(last_solve.values, case, network, master)
            flow = last_solve.values[master.index.flow_by_active_branch]
            payload["active_stage"] = "fractional_exhaustive_contingency_screen"
            save_checkpoint()
            started = time.perf_counter()
            screened = screener.screen(
                np.asarray(flow, dtype=np.float64),
                tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
                already_added=added_pair_ids,
            )
            screen_wall = time.perf_counter() - started
            screen_wall_total += screen_wall
            round_payload["screen"] = {
                "wall_time_seconds": screen_wall,
                "evaluated_sides": screened.evaluated_pairs,
                "new_violated_pairs": len(screened.violations),
                "maximum_violation_pu": screened.maximum_violation_pu,
                "maximum_pair_id": screened.maximum_pair_id,
            }
            payload["last_completed_screen"] = round_payload["screen"]
            tolerance = float(config.model["security_violation_tolerance_pu"])
            if not screened.violations:
                final_screen_passed = screened.maximum_violation_pu <= tolerance
                bound_ready = bool(
                    best_verified_bound is not None
                    and best_verified_bound >= required_bound
                )
                round_payload["acceptance_progress"].update(
                    {
                        "final_exhaustive_screen_passed": final_screen_passed,
                        "both_preverification_gates_ready": bool(
                            final_screen_passed and bound_ready
                        ),
                    }
                )
                if final_screen_passed and bound_ready:
                    payload["status"] = "continuous_preverification_gates_passed"
                    save_checkpoint()
                    break
                if not final_screen_passed:
                    payload["status"] = "failed_enforced_pair_residual"
                    save_checkpoint()
                    break
                if last_solve.optimal:
                    payload["status"] = (
                        "lp_relaxation_exhaustive_bound_insufficient"
                        if dual_passed
                        else "failed_continuous_dual_certificate"
                    )
                    save_checkpoint()
                    break
                payload["status"] = "incomplete_reference_gap_bound_not_ready"
                save_checkpoint()
                continue
            add_security_pairs(
                master.canonical,
                master.index,
                network,
                screened.violations,
            )
            new_ids = [pair.pair_id for pair in screened.violations]
            added_pair_ids.update(new_ids)
            payload["added_security_pair_ids"].extend(new_ids)
            round_payload["added_pair_ids"] = new_ids
            save_checkpoint()
        else:
            payload["status"] = "incomplete_constraint_generation_round_limit"

        payload["timings_seconds"]["continuous_solver_rounds"] = solve_wall_total
        payload["timings_seconds"]["fractional_screening"] = screen_wall_total
        payload["constraint_generation_round_count"] = len(payload["constraint_generation_rounds"])
        payload["added_security_pair_count"] = len(added_pair_ids)
        payload["best_verified_bound_round"] = best_verified_bound_round
        payload["bound"] = best_verified_bound
        payload["final_model_dimensions"] = {
            "columns": master.canonical.num_columns,
            "rows": master.canonical.num_rows,
            "nonzeros": int(master.canonical.matrix_csr().nnz),
        }
        gap_bound_ready = bool(
            best_verified_bound is not None and best_verified_bound >= required_bound
        )
        if final_screen_passed and last_solve is not None:
            deadline.require(
                "independent fractional-solution verification",
                reserve_seconds=float(config.runtime["serialization_reserve_seconds"]),
            )
            payload["active_stage"] = "independent_fractional_verification"
            save_checkpoint()
            started = time.perf_counter()
            verification = verify_serialized_solution(config, payload, require_integrality=False)
            payload["timings_seconds"]["independent_verification"] = time.perf_counter() - started
            payload["independent_verification"] = verification.as_dict()
            dual_bound = (
                float(payload["bound"]) if payload["bound"] is not None else None
            )
            relative_gap = (
                (incumbent - dual_bound) / abs(incumbent)
                if dual_bound is not None
                else None
            )
            bound_margin = (
                dual_bound - required_bound if dual_bound is not None else None
            )
            gap_certified = bool(
                verification.passed
                and gap_bound_ready
                and relative_gap is not None
                and relative_gap <= requested_gap * (1.0 + 1e-9) + 1e-12
            )
            payload["reference_incumbent_gap_test"] = {
                "known_integer_feasible_objective": incumbent,
                "requested_relative_gap": requested_gap,
                "required_lower_bound": required_bound,
                "verified_numerical_lp_dual_bound": dual_bound,
                "relative_gap": relative_gap,
                "bound_margin_to_requirement": bound_margin,
                "certified": gap_certified,
            }
            payload["acceptance_gates"] = {
                "zero_integer_columns_in_solve": (
                    last_solve.statistics["native_integer_columns"] == 0
                ),
                "pdlp_explicitly_selected": (
                    last_solve.statistics["method_parameters_readback"]["method"] == 1
                ),
                "branch_and_bound_markers_absent": last_solve.statistics[
                    "native_log_branch_and_bound_markers_absent"
                ],
                "numerical_dual_certificate_passed": best_verified_bound is not None,
                "verified_bound_source_round": best_verified_bound_round,
                "reference_gap_bound_threshold_met": gap_bound_ready,
                "final_fractional_exhaustive_screen_passed": final_screen_passed,
                "independent_fractional_verification_passed": verification.passed,
                "reference_incumbent_gap_certified": gap_certified,
            }
            if gap_certified:
                payload["status"] = "lp_relaxation_gap_certified"
            elif not verification.passed:
                payload["status"] = "failed_independent_fractional_verification"
            elif dual_bound is None:
                payload["status"] = "failed_continuous_dual_certificate"
            else:
                payload["status"] = "lp_relaxation_exhaustive_bound_insufficient"
        save_checkpoint()

    payload["active_stage"] = "complete"
    payload["elapsed_seconds"] = deadline.elapsed
    return payload


def run_one_shot_lp_certificate(
    config: RunConfig,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze, register, and hard-limit exactly one DGX LP-certificate run."""

    registration = validate_lp_certificate_config(config)
    validate_platform(config, "dgx_spark")
    output = guard_output_path(output_path)
    experiment_root = (config.root / "results" / "experiments").resolve()
    if not output.resolve().is_relative_to(experiment_root):
        raise ScopfError("LP-certificate output must be under results/experiments")
    if output.exists():
        raise ScopfError(f"LP-certificate output already exists: {output}")
    identity = frozen_identity(config)
    suite_id = registration["experiment_suite_id"]
    if not suite_id:
        raise ScopfError("LP-certificate experiment_suite_id is empty")
    registry_path = guard_output_path(experiment_root / f"{suite_id}-run-registry.json")
    if registry_path.exists():
        raise ScopfError("LP-certificate one-shot registry already exists; rerun forbidden")
    checkpoint = guard_output_path(
        config.root / "results" / "checkpoints" / f"{config.benchmark_id}-dgx_spark.json"
    )
    console_path = guard_output_path(
        config.root / "results" / "diagnostics" / f"{config.benchmark_id}-worker-console.log"
    )
    for path in (checkpoint, console_path):
        if path.exists():
            raise ScopfError(f"LP-certificate one-shot path already exists: {path}")
    registry = {
        "schema_version": "1.0.0",
        "experiment_suite_id": suite_id,
        "runs": {
            "dgx_spark": {
                "status": "started",
                "started_at_utc": datetime.now(UTC).isoformat(),
                "host": platform.node(),
                "output": str(output),
                "checkpoint": str(checkpoint),
                "frozen_identity": identity,
            }
        },
    }
    write_json_atomic(registry, registry_path)
    command = [
        sys.executable,
        "-m",
        "activsg_scopf.cli",
        "_lp_certificate_worker",
        "--config",
        str(config.path),
        "--output",
        str(output),
        "--checkpoint",
        str(checkpoint),
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
        total_wall = time.perf_counter() - started
        _write_console(console_path, completed.stdout, completed.stderr)
        if output.exists():
            result = _read_json(output)
        else:
            result = _read_json(checkpoint) if checkpoint.exists() else {}
            result.update(
                {
                    "status": "failed_worker_without_result",
                    "worker_returncode": completed.returncode,
                    "worker_stdout": completed.stdout[-4000:],
                    "worker_stderr": completed.stderr[-4000:],
                }
            )
    except subprocess.TimeoutExpired as exc:
        total_wall = time.perf_counter() - started
        _write_console(console_path, exc.stdout, exc.stderr)
        result = _read_json(checkpoint) if checkpoint.exists() else {}
        result.update(
            {
                "status": "hard_deadline_exceeded",
                "worker_timeout_seconds": deadline_seconds,
            }
        )
    result.update(
        {
            "official": False,
            "experiment": True,
            "one_shot": True,
            "frozen_identity": identity,
            "worker_console_log": str(console_path.relative_to(config.root)),
            "total_wall_time_seconds": total_wall,
            "benchmark_boundary": (
                "worker launch through raw loading, continuous PDLP solve/screen rounds, "
                "independent fractional verification, and first result serialization; "
                f"hard {deadline_seconds:g}-second wall-clock deadline"
            ),
        }
    )
    write_json_atomic(result, output)
    record = registry["runs"]["dgx_spark"]
    record.update(
        {
            "status": str(result.get("status", "unknown")),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "total_wall_time_seconds": total_wall,
            "objective": result.get("objective"),
            "bound": result.get("bound"),
            "reference_gap_test": result.get("reference_incumbent_gap_test"),
        }
    )
    write_json_atomic(registry, registry_path)
    return result


def run_lp_certificate_worker_serialized(
    config: RunConfig,
    *,
    output_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Worker boundary that preserves partial evidence on timeout or failure."""

    output = guard_output_path(output_path)
    checkpoint_file = guard_output_path(checkpoint_path)

    def checkpoint(payload: dict[str, Any]) -> None:
        write_json_atomic(payload, checkpoint_file)

    try:
        result = run_lp_relaxation_certificate(config, checkpoint=checkpoint)
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
    serialization_started = time.perf_counter()
    write_json_atomic(result, output)
    result.setdefault("timings_seconds", {})["result_serialization"] = (
        time.perf_counter() - serialization_started
    )
    write_json_atomic(result, output)
    return result
