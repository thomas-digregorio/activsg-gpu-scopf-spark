#!/usr/bin/env python3
"""Probe the centered, unit-scaled GPU multiplier LP on v13 bottleneck leaves."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from activsg_scopf.commitment_cuts import (
    add_commitment_upper_cuts,
    commitment_upper_cut_from_record,
)
from activsg_scopf.config import load_config
from activsg_scopf.lagrangian import (
    build_lagrangian_multiplier_delta_search_model,
    evaluate_lagrangian_bound,
    expand_lagrangian_multiplier_delta_candidate,
    replay_lagrangian_certificate,
)
from activsg_scopf.lagrangian_experiment import (
    _certificate_dual_arrays,
    _masks_from_record,
    _solve_summary,
)
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.provenance import write_json_atomic
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    security_pair_from_record,
)
from activsg_scopf.solvers.cuopt_lp import solve_cuopt_continuous_pdlp

RADIUS_SCHEDULE = (
    (10.0, 1_000.0, 8.0),
    (100.0, 10_000.0, 8.0),
    (1_000.0, 100_000.0, 8.0),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    started = time.perf_counter()
    root = Path("/workspace")
    config = load_config(root / "configs" / "activsg2000-gpu-lagrangian-v13.json")
    prior_path = (
        root
        / "results"
        / "experiments"
        / "activsg2000-gpu-lagrangian-v13-dgx-spark.json"
    )
    prior = _read_json(prior_path)
    if not bool(prior.get("lagrangian_verification", {}).get("passed")):
        raise RuntimeError("v13 fixture lacks independent Lagrangian replay")
    case = read_matpower_case(
        config.case_path,
        expected_sha256=config.raw["raw_inputs"]["case_sha256"],
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
    frontier = list(prior["frontier_regions"])
    minimum_prior_bound = min(
        float(record["lagrangian_certificate"]["conservative_lower_bound"])
        for record in frontier
    )
    bottlenecks = sorted(
        (
            record
            for record in frontier
            if abs(
                float(record["lagrangian_certificate"]["conservative_lower_bound"])
                - minimum_prior_bound
            )
            <= 1e-6
        ),
        key=lambda record: str(record["region_id"]),
    )
    records: list[dict[str, Any]] = []
    for frontier_record in bottlenecks:
        region_started = time.perf_counter()
        region_id = str(frontier_record["region_id"])
        masks = _masks_from_record(frontier_record, source_rows)
        master = build_reduced_master(
            case,
            network,
            segments=int(config.model["pwl_segments"]),
            coefficient_zero_tolerance=float(
                config.model["reduced_coefficient_zero_tolerance"]
            ),
        )
        pairs = tuple(
            security_pair_from_record(
                pair_record,
                catalog,
                lodf_absolute_tolerance=float(
                    config.model["serialized_lodf_replay_tolerance"]
                ),
            )
            for pair_record in frontier_record["security_pairs"]
        )
        equivalence = frontier_record.get("security_row_equivalence", {})
        add_reduced_security_pairs(
            master,
            network,
            pairs,
            expected_representative_by_pair_id=equivalence.get(
                "representative_by_pair_id"
            ),
            equivalence_replay_tolerance=float(
                config.model["security_equivalence_replay_tolerance"]
            ),
        )
        commitment_cuts = tuple(
            commitment_upper_cut_from_record(cut_record, source_rows)
            for cut_record in frontier_record.get("commitment_upper_cuts", [])
        )
        add_commitment_upper_cuts(master, commitment_cuts)
        inherited = replay_lagrangian_certificate(
            master,
            frontier_record["lagrangian_certificate"],
            masks,
            commitment_cuts_by_id={cut.cut_id: cut for cut in commitment_cuts},
        )
        current = inherited
        current_row_dual, current_cut_dual = _certificate_dual_arrays(
            master, current, commitment_cuts
        )
        passes: list[dict[str, Any]] = []
        for pass_number, (
            coupling_radius,
            cut_radius,
            solve_budget,
        ) in enumerate(RADIUS_SCHEDULE, start=1):
            search = build_lagrangian_multiplier_delta_search_model(
                master,
                current_row_dual,
                masks,
                commitment_cuts=commitment_cuts,
                commitment_cut_dual=current_cut_dual,
                maximum_new_violated_coupling_rows=384,
                coupling_trust_radius=coupling_radius,
                commitment_cut_trust_radius=cut_radius,
            )
            solve_started = time.perf_counter()
            solve = solve_cuopt_continuous_pdlp(
                search.canonical,
                time_limit_seconds=solve_budget,
                optimality_tolerance=1e-8,
                primal_feasibility_tolerance=1e-7,
                certificate_residual_tolerance=1e-7,
                native_scaling_mode="none",
                native_base_mva=float(case.base_mva),
                log_to_console=True,
                per_constraint_residual=True,
                presolve=0,
                initial_native_primal=search.initial_values,
                pdlp_solver_mode=1,
            )
            solve_wall = time.perf_counter() - solve_started
            pass_record: dict[str, Any] = {
                "pass": pass_number,
                "coupling_trust_radius": coupling_radius,
                "commitment_cut_trust_radius": cut_radius,
                "search_solver_budget_seconds": solve_budget,
                "search_solver_wall_time_seconds": solve_wall,
                "center_conservative_lower_bound": (
                    current.conservative_lower_bound
                ),
                "search_model": search.audit,
                "search_solve": _solve_summary(solve),
                "search_lp_solution_used_as_bound": False,
            }
            if solve.values is None or not np.all(np.isfinite(solve.values)):
                pass_record["status"] = "no_finite_search_vector"
                passes.append(pass_record)
                continue
            candidate_row_dual, candidate_cut_dual, reconstruction = (
                expand_lagrangian_multiplier_delta_candidate(
                    master, search, solve.values
                )
            )
            candidate = evaluate_lagrangian_bound(
                master,
                candidate_row_dual,
                masks,
                safety_margin_dollars=float(
                    config.raw["benchmark"]["certificate_safety_margin_dollars"]
                ),
                commitment_cuts=commitment_cuts,
                commitment_cut_dual=candidate_cut_dual,
            )
            prior_pass_bound = current.conservative_lower_bound
            accepted = (
                candidate.conservative_lower_bound
                > current.conservative_lower_bound
            )
            if accepted:
                current = candidate
                current_row_dual, current_cut_dual = _certificate_dual_arrays(
                    master, current, commitment_cuts
                )
            pass_record.update(
                {
                    "status": (
                        "exact_candidate_accepted"
                        if accepted
                        else "exact_candidate_rejected_monotone"
                    ),
                    "candidate_conservative_lower_bound": (
                        candidate.conservative_lower_bound
                    ),
                    "exact_candidate_improvement_over_center_dollars": (
                        candidate.conservative_lower_bound - prior_pass_bound
                    ),
                    "accepted_conservative_lower_bound": (
                        current.conservative_lower_bound
                    ),
                    "candidate_reconstruction": reconstruction,
                    "search_predicted_improvement_dollars": (
                        None
                        if solve.primal_objective is None
                        else -float(solve.primal_objective)
                        * search.objective_normalizer
                    ),
                }
            )
            passes.append(pass_record)

        serialized = current.as_dict(source_rows, compact=True)
        replayed = replay_lagrangian_certificate(
            master,
            serialized,
            masks,
            commitment_cuts_by_id={cut.cut_id: cut for cut in commitment_cuts},
        )
        replay_difference = abs(
            replayed.conservative_lower_bound - current.conservative_lower_bound
        )
        if replay_difference > float(
            config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]
        ):
            raise RuntimeError(f"Centered search certificate failed replay for {region_id}")
        records.append(
            {
                "region_id": region_id,
                "prior_conservative_lower_bound": inherited.conservative_lower_bound,
                "selected_conservative_lower_bound": (
                    current.conservative_lower_bound
                ),
                "improvement_dollars": (
                    current.conservative_lower_bound
                    - inherited.conservative_lower_bound
                ),
                "serialized_certificate_replay_difference_dollars": (
                    replay_difference
                ),
                "passes": passes,
                "region_wall_time_seconds": time.perf_counter() - region_started,
            }
        )

    output = {
        "diagnostic_id": "activsg2000-v19-centered-dual-lp-probe",
        "status": "passed",
        "source_case_sha256": config.raw["raw_inputs"]["case_sha256"],
        "source_contingency_sha256": config.raw["raw_inputs"][
            "contingency_sha256"
        ],
        "source_gpu_result": str(prior_path.relative_to(root)),
        "source_gpu_lagrangian_replay_passed": True,
        "cpu_problem_solution_data_used": False,
        "search_lp_solution_used_as_bound": False,
        "exact_nonsmoothed_host_replay_is_bound_authority": True,
        "exact_source_pmin_pmax_changed": False,
        "mathematical_original_integer_feasible_set_changed": False,
        "radius_schedule": [list(values) for values in RADIUS_SCHEDULE],
        "bottleneck_region_count": len(bottlenecks),
        "prior_global_bound": minimum_prior_bound,
        "polished_bottleneck_bound": min(
            float(record["selected_conservative_lower_bound"])
            for record in records
        ),
        "minimum_bottleneck_improvement_dollars": min(
            float(record["improvement_dollars"]) for record in records
        ),
        "regions": records,
        "total_wall_time_seconds": time.perf_counter() - started,
    }
    output_path = (
        root
        / "results"
        / "diagnostics"
        / "activsg2000-v19-centered-dual-lp-probe.json"
    )
    write_json_atomic(output, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "status": output["status"],
                "prior_global_bound": output["prior_global_bound"],
                "polished_bottleneck_bound": output["polished_bottleneck_bound"],
                "minimum_bottleneck_improvement_dollars": output[
                    "minimum_bottleneck_improvement_dollars"
                ],
                "total_wall_time_seconds": output["total_wall_time_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
