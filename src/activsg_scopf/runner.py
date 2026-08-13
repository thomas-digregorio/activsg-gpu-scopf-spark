"""End-to-end constraint generation under one global deadline."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig, load_config
from .costs import pwl_approximation_report
from .deadline import Deadline, PeakMemorySampler
from .environment import environment_manifest, validate_platform
from .matpower import read_contingency_table, read_matpower_case
from .model import build_master
from .network import (
    build_contingency_catalog,
    build_network,
    contingency_catalog_report,
)
from .paths import guard_runtime_environment
from .provenance import build_source_manifest
from .screening import add_security_pairs, screen_contingencies
from .solution import serialize_solution
from .solvers import SolveResult, solve_canonical
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
    configured_deadline = float(config.runtime["deadline_seconds"])
    total_deadline = min(configured_deadline, deadline_seconds or configured_deadline)
    deadline = Deadline(
        total_seconds=total_deadline,
        verification_reserve_seconds=float(config.runtime["verification_reserve_seconds"]),
        serialization_reserve_seconds=float(config.runtime["serialization_reserve_seconds"]),
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "case_name": "ACTIVSg500",
        "interval_hours": 1.0,
        "official": official,
        "platform": platform_name,
        "status": "running",
        "timings_seconds": {},
        "constraint_generation_rounds": [],
        "added_security_pair_ids": [],
    }

    def save_checkpoint() -> None:
        payload["elapsed_seconds"] = deadline.elapsed
        if checkpoint is not None:
            checkpoint(payload)

    with PeakMemorySampler(sample_gpu=platform_name == "dgx_spark") as memory:
        guard_runtime_environment(config.root)
        validate_platform(config, platform_name)
        payload["environment"] = environment_manifest(platform_name)
        save_checkpoint()

        stage = time.perf_counter()
        case = read_matpower_case(
            config.case_path,
            expected_sha256=config.raw["raw_inputs"]["case_sha256"],
        )
        contingency_table = read_contingency_table(
            config.contingency_path,
            expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
        )
        payload["timings_seconds"]["raw_input_loading"] = _seconds_since(stage)
        payload["source_manifest"] = build_source_manifest(case, contingency_table)
        save_checkpoint()

        stage = time.perf_counter()
        network = build_network(case)
        catalog = build_contingency_catalog(
            case,
            network,
            contingency_table,
            validation_columns=int(config.model["lodf_validation_columns"]),
            validation_tolerance_pu=float(config.model["lodf_validation_tolerance_pu"]),
        )
        master = build_master(case, network, segments=int(config.model["pwl_segments"]))
        payload["timings_seconds"]["model_and_factor_build"] = _seconds_since(stage)
        payload["contingencies"] = contingency_catalog_report(catalog)
        payload["pwl_costs"] = pwl_approximation_report(master.costs)
        payload["base_model_dimensions"] = {
            "columns": master.canonical.num_columns,
            "rows": master.canonical.num_rows,
            "nonzeros": int(master.canonical.matrix_csr().nnz),
        }
        save_checkpoint()

        profile = config.raw["platforms"][platform_name]
        added_pair_ids: set[str] = set()
        solve_total = 0.0
        screen_total = 0.0
        last_solve: SolveResult | None = None
        secure = False
        maximum_rounds = int(config.runtime["maximum_constraint_generation_rounds"])
        for round_number in range(1, maximum_rounds + 1):
            deadline.require("restricted-master solve", reserve_seconds=0.0)
            solver_budget = deadline.solver_budget()
            stage = time.perf_counter()
            last_solve = solve_canonical(
                master.canonical,
                solver=profile["solver"],
                time_limit_seconds=solver_budget,
                mip_relative_gap=float(config.model["mip_relative_gap_tolerance"]),
                threads=int(profile["solver_threads"]),
            )
            solve_elapsed = _seconds_since(stage)
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
            stage = time.perf_counter()
            screened = screen_contingencies(
                np.asarray(flow, dtype=np.float64),
                network,
                catalog,
                tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
                backend=profile["screening"],
                already_added=added_pair_ids,
            )
            screen_elapsed = _seconds_since(stage)
            screen_total += screen_elapsed
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
                catalog,
                screened.violations,
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
            save_checkpoint()

    payload["peak_memory"] = {
        "process_rss_bytes": memory.peak_process_rss_bytes,
        "cupy_pool_used_bytes": memory.peak_gpu_pool_used_bytes,
        "cuda_device_memory_delta_bytes": memory.peak_cuda_device_memory_delta_bytes,
    }
    payload["total_wall_time_seconds_before_serialization"] = deadline.elapsed
    return payload
