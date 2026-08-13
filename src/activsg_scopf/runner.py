"""End-to-end constraint generation for bounded benchmarks or unbounded experiments."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig, load_config
from .costs import pwl_approximation_report
from .deadline import Deadline, PeakMemorySampler
from .diagnostics import DiagnosticEventWriter
from .environment import environment_manifest, validate_platform
from .matpower import read_contingency_table, read_matpower_case
from .model import build_master
from .network import (
    build_contingency_catalog,
    build_network,
    contingency_catalog_report,
)
from .paths import guard_output_path, guard_runtime_environment
from .pricing import run_fixed_commitment_pricing
from .provenance import build_source_manifest
from .screening import ContingencyScreener, add_security_pairs
from .solution import serialize_solution
from .solvers import SolveResult, create_solver_session
from .verify import verify_serialized_solution

Checkpoint = Callable[[dict[str, Any]], None]


def _seconds_since(started: float) -> float:
    return time.perf_counter() - started


def _solve_summary(result: SolveResult) -> dict[str, Any]:
    return {
        "solver": result.solver,
        "solver_version": result.solver_version,
        "status": result.status,
        "optimal": result.optimal,
        "has_incumbent": result.has_incumbent,
        "objective": result.objective,
        "bound": result.bound,
        "mip_gap": result.mip_gap,
        "solve_time_seconds": result.solve_time_seconds,
        "statistics": result.statistics,
    }


def run_end_to_end(
    config_or_path: RunConfig | str | Path,
    *,
    platform_name: str,
    deadline_seconds: float | None = None,
    official: bool = False,
    checkpoint: Checkpoint | None = None,
) -> dict[str, Any]:
    config = (
        config_or_path
        if isinstance(config_or_path, RunConfig)
        else load_config(config_or_path)
    )
    configured_deadline_value = config.runtime["deadline_seconds"]
    unbounded = configured_deadline_value is None
    if unbounded:
        if deadline_seconds is not None:
            raise ValueError("An unbounded experiment cannot receive a deadline override")
        total_deadline = float("inf")
    else:
        configured_deadline = float(configured_deadline_value)
        total_deadline = min(configured_deadline, deadline_seconds or configured_deadline)
    deadline = Deadline(
        total_seconds=total_deadline,
        verification_reserve_seconds=(
            0.0
            if unbounded
            else float(config.runtime["verification_reserve_seconds"])
        ),
        serialization_reserve_seconds=(
            0.0
            if unbounded
            else float(config.runtime["serialization_reserve_seconds"])
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "case_name": config.case_name,
        "benchmark_id": config.benchmark_id,
        "interval_hours": 1.0,
        "official": official,
        "platform": platform_name,
        "status": "running",
        "timings_seconds": {},
        "constraint_generation_rounds": [],
        "added_security_pair_ids": [],
        "deadline_seconds": None if unbounded else total_deadline,
    }
    profile = config.raw["platforms"][platform_name]
    diagnostics_profile = profile.get("diagnostics", {})
    diagnostics_enabled = bool(diagnostics_profile.get("enabled", False))
    diagnostic_writer: DiagnosticEventWriter | None = None
    native_log_path: Path | None = None
    if diagnostics_enabled:
        diagnostic_root = config.root / "results" / "diagnostics"
        event_path = guard_output_path(
            diagnostic_root / f"{config.benchmark_id}-{platform_name}-events.jsonl"
        )
        native_log_path = guard_output_path(
            diagnostic_root / f"{config.benchmark_id}-{platform_name}-highs.log"
        )
        diagnostic_writer = DiagnosticEventWriter(event_path, started=deadline.started)
        payload["diagnostics"] = {
            "enabled": True,
            "event_log": str(event_path.relative_to(config.root)),
            "native_solver_log": str(native_log_path.relative_to(config.root)),
            "mip_logging_interval_seconds": float(
                diagnostics_profile.get("mip_logging_interval_seconds", 1.0)
            ),
        }

    def emit_diagnostic(event: str, **fields: Any) -> None:
        if diagnostic_writer is not None:
            diagnostic_writer.emit(event, **fields)

    emit_diagnostic(
        "end_to_end_started",
        benchmark_id=config.benchmark_id,
        platform=platform_name,
        deadline_seconds=None if unbounded else total_deadline,
    )
    memory_sampler = PeakMemorySampler(sample_gpu=platform_name == "dgx_spark")

    def save_checkpoint() -> None:
        payload["elapsed_seconds"] = deadline.elapsed
        payload["peak_memory"] = {
            "process_rss_bytes": memory_sampler.peak_process_rss_bytes,
            "cupy_pool_used_bytes": memory_sampler.peak_gpu_pool_used_bytes,
            "cuda_device_memory_delta_bytes": (
                memory_sampler.peak_cuda_device_memory_delta_bytes
            ),
        }
        if checkpoint is not None:
            checkpoint(payload)

    with memory_sampler as memory:
        guard_runtime_environment(config.root)
        validate_platform(config, platform_name)
        payload["environment"] = environment_manifest(platform_name)
        save_checkpoint()

        stage = time.perf_counter()
        emit_diagnostic("raw_input_loading_started")
        case = read_matpower_case(
            config.case_path,
            expected_sha256=config.raw["raw_inputs"]["case_sha256"],
        )
        contingency_table = read_contingency_table(
            config.contingency_path,
            expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
        )
        payload["timings_seconds"]["raw_input_loading"] = _seconds_since(stage)
        emit_diagnostic(
            "raw_input_loading_finished",
            wall_time_seconds=payload["timings_seconds"]["raw_input_loading"],
        )
        payload["source_manifest"] = build_source_manifest(case, contingency_table)
        save_checkpoint()

        stage = time.perf_counter()
        emit_diagnostic("model_and_factor_build_started")
        network = build_network(case)
        catalog = build_contingency_catalog(
            case,
            network,
            contingency_table,
            validation_columns=int(config.model["lodf_validation_columns"]),
            validation_tolerance_pu=float(config.model["lodf_validation_tolerance_pu"]),
            chunk_columns=int(config.model.get("lodf_build_chunk_columns", 256)),
        )
        master = build_master(case, network, segments=int(config.model["pwl_segments"]))
        payload["timings_seconds"]["model_and_factor_build"] = _seconds_since(stage)
        emit_diagnostic(
            "model_and_factor_build_finished",
            wall_time_seconds=payload["timings_seconds"]["model_and_factor_build"],
        )
        payload["contingencies"] = contingency_catalog_report(catalog)
        payload["pwl_costs"] = pwl_approximation_report(master.costs)
        payload["base_model_dimensions"] = {
            "columns": master.canonical.num_columns,
            "rows": master.canonical.num_rows,
            "nonzeros": int(master.canonical.matrix_csr().nnz),
        }
        save_checkpoint()

        screen_chunk_columns = int(config.model.get("screen_chunk_columns", 256))
        stage = time.perf_counter()
        emit_diagnostic("screen_workspace_build_started")
        screener = ContingencyScreener(
            network,
            catalog,
            backend=profile["screening"],
            chunk_columns=screen_chunk_columns,
        )
        payload["timings_seconds"]["screen_workspace_build"] = _seconds_since(stage)
        emit_diagnostic(
            "screen_workspace_build_finished",
            wall_time_seconds=payload["timings_seconds"]["screen_workspace_build"],
        )
        stage = time.perf_counter()
        emit_diagnostic("solver_session_build_started")
        solver_session = create_solver_session(
            master.canonical,
            solver=profile["solver"],
            mip_relative_gap=float(config.model["mip_relative_gap_tolerance"]),
            threads=int(profile["solver_threads"]),
            diagnostic_event=(
                emit_diagnostic if diagnostics_enabled and profile["solver"] == "highs" else None
            ),
            native_log_path=(
                native_log_path
                if diagnostics_enabled and profile["solver"] == "highs"
                else None
            ),
            mip_logging_interval_seconds=float(
                diagnostics_profile.get("mip_logging_interval_seconds", 1.0)
            ),
        )
        expected_session_mode = profile.get("solver_session")
        if expected_session_mode and solver_session.mode != expected_session_mode:
            raise ValueError(
                f"Configured solver session {expected_session_mode!r} does not match "
                f"adapter mode {solver_session.mode!r}"
            )
        payload["timings_seconds"]["solver_session_build"] = _seconds_since(stage)
        emit_diagnostic(
            "solver_session_build_finished",
            wall_time_seconds=payload["timings_seconds"]["solver_session_build"],
            session_mode=solver_session.mode,
        )
        payload["solver_session_mode"] = solver_session.mode
        save_checkpoint()
        added_pair_ids: set[str] = set()
        solve_total = 0.0
        screen_total = 0.0
        last_solve: SolveResult | None = None
        secure = False
        maximum_rounds = int(config.runtime["maximum_constraint_generation_rounds"])
        for round_number in range(1, maximum_rounds + 1):
            deadline.require("restricted-master solve", reserve_seconds=0.0)
            solver_budget = None if unbounded else deadline.solver_budget()
            payload["active_stage"] = "restricted_master_solve"
            payload["active_constraint_generation_round"] = round_number
            payload["active_solver_budget_seconds"] = solver_budget
            save_checkpoint()
            emit_diagnostic(
                "constraint_generation_round_started",
                round=round_number,
                rows_before_solve=master.canonical.num_rows,
                solver_budget_seconds=solver_budget,
            )
            stage = time.perf_counter()
            last_solve = solver_session.solve(time_limit_seconds=solver_budget)
            solve_elapsed = _seconds_since(stage)
            emit_diagnostic(
                "restricted_master_solve_finished",
                round=round_number,
                wall_time_seconds=solve_elapsed,
                solver_status=last_solve.status,
                objective=last_solve.objective,
                bound=last_solve.bound,
                mip_gap=last_solve.mip_gap,
            )
            solve_total += solve_elapsed
            round_payload: dict[str, Any] = {
                "round": round_number,
                "rows_before_solve": master.canonical.num_rows,
                "solver_budget_seconds": solver_budget,
                "adapter_wall_time_seconds": solve_elapsed,
                "solve": _solve_summary(last_solve),
            }
            payload["constraint_generation_rounds"].append(round_payload)
            payload.update(
                {
                    "objective": last_solve.objective,
                    "bound": last_solve.bound,
                    "mip_gap": last_solve.mip_gap,
                    "solver_status": last_solve.status,
                }
            )
            if not last_solve.has_incumbent or last_solve.values is None:
                payload["status"] = "incomplete_no_incumbent"
                save_checkpoint()
                break
            payload["solution"] = serialize_solution(
                last_solve.values, case, network, master
            )
            flow = last_solve.values[master.index.flow_by_active_branch]
            payload["active_stage"] = "exhaustive_contingency_screen"
            save_checkpoint()
            stage = time.perf_counter()
            emit_diagnostic(
                "exhaustive_contingency_screen_started",
                round=round_number,
            )
            screened = screener.screen(
                np.asarray(flow, dtype=np.float64),
                tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
                already_added=added_pair_ids,
            )
            screen_elapsed = _seconds_since(stage)
            screen_total += screen_elapsed
            emit_diagnostic(
                "exhaustive_contingency_screen_finished",
                round=round_number,
                wall_time_seconds=screen_elapsed,
                evaluated_sides=screened.evaluated_pairs,
                new_violated_pairs=len(screened.violations),
                maximum_violation_pu=screened.maximum_violation_pu,
            )
            round_payload["screen"] = {
                "wall_time_seconds": screen_elapsed,
                "evaluated_sides": screened.evaluated_pairs,
                "new_violated_pairs": len(screened.violations),
                "maximum_violation_pu": screened.maximum_violation_pu,
                "maximum_pair_id": screened.maximum_pair_id,
            }
            payload["last_completed_screen"] = round_payload["screen"]
            save_checkpoint()
            if not last_solve.optimal:
                payload["status"] = "incomplete_restricted_master_not_optimal"
                break
            tolerance = float(config.model["security_violation_tolerance_pu"])
            if not screened.violations:
                if screened.maximum_violation_pu <= tolerance:
                    secure = True
                    break
                payload["status"] = "failed_enforced_pair_residual"
                break
            add_security_pairs(
                master.canonical,
                master.index,
                network,
                screened.violations,
            )
            emit_diagnostic(
                "security_pairs_added",
                round=round_number,
                pair_count=len(screened.violations),
                rows_after_add=master.canonical.num_rows,
            )
            new_ids = [pair.pair_id for pair in screened.violations]
            added_pair_ids.update(new_ids)
            payload["added_security_pair_ids"].extend(new_ids)
            round_payload["added_pair_ids"] = new_ids
            save_checkpoint()
        else:
            payload["status"] = "incomplete_constraint_generation_round_limit"

        payload["timings_seconds"]["solver_rounds"] = solve_total
        payload["timings_seconds"]["screening"] = screen_total
        payload["constraint_generation_round_count"] = len(
            payload["constraint_generation_rounds"]
        )
        payload["final_model_dimensions"] = {
            "columns": master.canonical.num_columns,
            "rows": master.canonical.num_rows,
            "nonzeros": int(master.canonical.matrix_csr().nnz),
        }
        payload["added_security_pairs"] = len(added_pair_ids)
        if secure and last_solve is not None and last_solve.values is not None:
            deadline.require(
                "independent verification",
                reserve_seconds=float(config.runtime["serialization_reserve_seconds"]),
            )
            stage = time.perf_counter()
            payload["active_stage"] = "independent_verification"
            save_checkpoint()

            emit_diagnostic("independent_verification_started")
            verification = verify_serialized_solution(config, payload)
            payload["timings_seconds"]["independent_verification"] = _seconds_since(stage)
            payload["verification"] = verification.as_dict()
            payload["maximum_model_residual_pu"] = verification.maximum_model_residual_pu
            payload["final_exhaustive_violation_pu"] = (
                verification.maximum_security_violation_pu
            )
            payload["commitment_count"] = sum(
                float(record["commitment"]) > 0.5
                for record in payload["solution"]["generators"]
            )
            payload["status"] = (
                "optimal_verified" if verification.passed else "failed_independent_verification"
            )
            emit_diagnostic(
                "independent_verification_finished",
                passed=verification.passed,
                wall_time_seconds=payload["timings_seconds"][
                    "independent_verification"
                ],
            )
            save_checkpoint()

            pricing_profile = config.raw["benchmark"].get("pricing", {})
            if verification.passed and bool(pricing_profile.get("enabled", False)):
                payload["active_stage"] = "fixed_commitment_pricing"
                save_checkpoint()
                emit_diagnostic("fixed_commitment_pricing_started")
                stage = time.perf_counter()
                try:
                    payload["pricing"] = run_fixed_commitment_pricing(
                        case,
                        network,
                        catalog,
                        master,
                        last_solve.values,
                        already_added_pair_ids=added_pair_ids,
                        security_tolerance_pu=float(
                            config.model["security_violation_tolerance_pu"]
                        ),
                        screen_chunk_columns=screen_chunk_columns,
                        solver_threads=int(profile["solver_threads"]),
                        maximum_rounds=int(
                            pricing_profile.get(
                                "maximum_constraint_generation_rounds", maximum_rounds
                            )
                        ),
                    )
                    payload["timings_seconds"]["fixed_commitment_pricing"] = (
                        _seconds_since(stage)
                    )
                    emit_diagnostic(
                        "fixed_commitment_pricing_finished",
                        status=payload["pricing"]["status"],
                        wall_time_seconds=payload["timings_seconds"][
                            "fixed_commitment_pricing"
                        ],
                    )
                except Exception as exc:
                    payload["timings_seconds"]["fixed_commitment_pricing"] = (
                        _seconds_since(stage)
                    )
                    payload["pricing"] = {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    payload["status"] = "optimal_verified_pricing_failed"
                    emit_diagnostic(
                        "fixed_commitment_pricing_failed",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                save_checkpoint()

        payload["active_stage"] = "result_serialization"

    payload["peak_memory"] = {
        "process_rss_bytes": memory.peak_process_rss_bytes,
        "cupy_pool_used_bytes": memory.peak_gpu_pool_used_bytes,
        "cuda_device_memory_delta_bytes": memory.peak_cuda_device_memory_delta_bytes,
    }
    payload["total_wall_time_seconds_before_serialization"] = deadline.elapsed
    emit_diagnostic(
        "end_to_end_finished",
        status=payload["status"],
        elapsed_seconds_before_serialization=deadline.elapsed,
    )
    if diagnostic_writer is not None:
        diagnostic_writer.close()
    return payload
