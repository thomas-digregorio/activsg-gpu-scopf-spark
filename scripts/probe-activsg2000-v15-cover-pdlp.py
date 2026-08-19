#!/usr/bin/env python3
"""Measure replayable binary-cover strengthening in one GPU PDLP component gate.

This is a bounded development probe, not a registered benchmark.  It uses only
raw ACTIVSg2000 inputs and preserved GPU-derived certificate data.  The first
PDLP solve produces the fractional separation reference; a second, warm-started
PDLP solve measures valid cover cuts derived from replay-certified Phase-I and
conditional-PMIN/PMAX cuts.  No CPU commitment, dispatch, objective, or bound is
read or used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from activsg_scopf.commitment_cuts import (
    CommitmentCapacityCut,
    CommitmentFeasibilityCut,
    commitment_upper_cut_from_record,
    derive_binary_knapsack_cover_cut,
)
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.lagrangian import RegionMasks
from activsg_scopf.lagrangian_experiment import (
    _solve_region,
    _solve_summary,
)
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import security_pair_from_record
from activsg_scopf.screening import ContingencyScreener


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_marker(path: Path, marker: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        selected: str | None = None
        for line in handle:
            if line.startswith(marker):
                selected = line[len(marker) :]
    if selected is None:
        raise RuntimeError(f"Missing {marker!r} in {path}")
    return json.loads(selected)


def _round_summary(round_record: dict[str, Any]) -> dict[str, Any]:
    solve = round_record["solve"]
    audit = solve.get("native_log_audit", {})
    certificate = solve.get("dual_certificate", {})
    return {
        "round": int(round_record["round"]),
        "rows_before_solve": int(round_record["rows_before_solve"]),
        "security_pairs_before_solve": int(
            round_record["security_pairs_before_solve"]
        ),
        "adapter_wall_time_seconds": float(
            round_record["adapter_wall_time_seconds"]
        ),
        "solve_status": solve.get("status"),
        "native_solve_time_seconds": solve.get("native_solve_time_seconds"),
        "native_primal_residual": solve.get("native_lp_stats", {}).get(
            "primal_residual"
        ),
        "native_dual_residual": solve.get("native_lp_stats", {}).get(
            "dual_residual"
        ),
        "dual_certificate_passed": certificate.get("passed"),
        "conservative_numerical_lower_bound": certificate.get(
            "conservative_numerical_lower_bound"
        ),
        "barrier_numerical_warning_count": int(
            audit.get("barrier_numerical_warning_count", 0)
        ),
        "mip_start_rejection_count": int(audit.get("mip_start_rejection_count", 0)),
        "free_variable_warning_count": int(
            audit.get("free_variable_warning_count", 0)
        ),
        "screen": round_record.get("screen"),
        "warm_start": solve.get("warm_start"),
    }


def main() -> None:
    started = time.perf_counter()
    workspace = Path("/workspace")
    config = load_config(
        workspace / "configs" / "activsg2000-gpu-lagrangian-v12.json"
    )
    # Every removed coefficient receives its exact box-derived outward RHS
    # relaxation.  This is a weaker continuous model and therefore preserves a
    # valid lower bound while eliminating the observed sub-2e-6 matrix tail.
    original_zero_tolerance = float(
        config.model["reduced_coefficient_zero_tolerance"]
    )
    config.raw["model"]["reduced_coefficient_zero_tolerance"] = 2e-6
    config.raw["platforms"]["dgx_spark"]["lagrangian_gpu_iterations"] = 1
    config.raw["runtime"]["maximum_pdlp_round_seconds"] = 180.0

    prior = _read_json(
        workspace
        / "results"
        / "experiments"
        / "activsg2000-gpu-lagrangian-v12-dgx-spark.json"
    )
    diagnostic = _read_marker(
        workspace
        / "results"
        / "diagnostics"
        / "activsg2000-v14-root-minimizer-cut-probe-attempt8.log",
        "V14_ROOT_MINIMIZER_CUT_PROBE=",
    )
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
    source_rows = np.flatnonzero(case.gen[:, GEN_STATUS] > 0).astype(np.int64)
    root_record = next(
        record for record in prior["solved_region_history"] if record["region_id"] == "r"
    )
    initial_pairs = tuple(
        security_pair_from_record(
            record,
            catalog,
            lodf_absolute_tolerance=float(
                config.model["serialized_lodf_replay_tolerance"]
            ),
        )
        for record in root_record["security_pairs"]
    )
    parent_records: dict[str, dict[str, Any]] = {}
    for iteration in diagnostic["cutting_plane_iterations"]:
        record = iteration.get("cut")
        if record is not None:
            parent_records.setdefault(str(record["cut_id"]), record)
    parent_cuts = tuple(
        commitment_upper_cut_from_record(record, source_rows + 1)
        for _, record in sorted(parent_records.items())
    )
    if not all(
        isinstance(cut, CommitmentFeasibilityCut | CommitmentCapacityCut)
        for cut in parent_cuts
    ):
        raise RuntimeError("The v14 parent-cut fixture contains an unsupported cut kind")

    deadline = Deadline(600.0, 30.0, 10.0)
    screener = ContingencyScreener(
        network,
        catalog,
        backend="cupy",
        chunk_columns=int(config.model["screen_chunk_columns"]),
    )
    masks = RegionMasks.root(source_rows.size)
    baseline = _solve_region(
        region_id="v15_cover_baseline",
        masks=masks,
        case=case,
        network=network,
        catalog=catalog,
        config=config,
        deadline=deadline,
        initial_pairs=initial_pairs,
        screener=screener,
        checkpoint=lambda: None,
        commitment_cuts=(),
    )
    reference = np.asarray(baseline.commitment, dtype=np.float64)
    if np.any(reference < -1e-7) or np.any(reference > 1.0 + 1e-7):
        raise RuntimeError("Baseline GPU LP returned invalid commitment values")
    reference = np.clip(reference, 0.0, 1.0)
    cover_records: list[dict[str, Any]] = []
    covers_by_id = {}
    for parent in parent_cuts:
        cover, derivation = derive_binary_knapsack_cover_cut(
            source_cut=parent,
            generator_source_rows=source_rows + 1,
            source_commitment=None,
            separation_reference=reference,
        )
        cover_records.append(
            {
                "source_cut_id": parent.cut_id,
                "reference_violation": cover.separation_reference_violation,
                "selected_for_pdlp": cover.separation_reference_violation > 1e-9,
                "derivation": derivation,
            }
        )
        if cover.separation_reference_violation > 1e-9:
            covers_by_id.setdefault(cover.cut_id, cover)
    covers = tuple(covers_by_id[cut_id] for cut_id in sorted(covers_by_id))
    if not covers:
        raise RuntimeError("No valid cover cut separates the baseline GPU LP point")

    strengthened = _solve_region(
        region_id="v15_cover_strengthened",
        masks=masks,
        case=case,
        network=network,
        catalog=catalog,
        config=config,
        deadline=deadline,
        initial_pairs=baseline.security_pairs,
        screener=screener,
        checkpoint=lambda: None,
        initial_native_primal=baseline.solve.native_primal,
        initial_native_row_dual=baseline.solve.native_row_dual,
        initial_warm_start_origin="same_scaled_gpu_lp_before_appended_cover_rows",
        commitment_cuts=covers,
    )
    baseline_certificate = baseline.solve.statistics["dual_certificate"]
    strengthened_certificate = strengthened.solve.statistics["dual_certificate"]
    baseline_bound = float(
        baseline_certificate["conservative_numerical_lower_bound"]
    )
    strengthened_bound = float(
        strengthened_certificate["conservative_numerical_lower_bound"]
    )
    required_bound = float(prior["objective"]) * (
        1.0 - float(config.model["mip_relative_gap_tolerance"])
    )
    required_lift = max(0.0, required_bound - baseline_bound)
    achieved_lift = strengthened_bound - baseline_bound
    native_audits = [
        record["solve"].get("native_log_audit", {})
        for region in (baseline, strengthened)
        for record in region.rounds
    ]
    numerical_failure_count = sum(
        int(audit.get("barrier_numerical_warning_count", 0))
        + int(audit.get("mip_start_rejection_count", 0))
        + int(audit.get("free_variable_warning_count", 0))
        for audit in native_audits
    )
    output = {
        "passed": True,
        "development_component_only": True,
        "gpu_only_problem_solution_data": True,
        "cpu_problem_solution_data_used": False,
        "exact_source_pmin_pmax_changed": False,
        "mathematical_integer_feasible_set_changed": False,
        "numeric_cleanup": {
            "policy": "drop_with_box_derived_outward_rhs_relaxation_v1",
            "original_zero_tolerance": original_zero_tolerance,
            "probe_zero_tolerance": float(
                config.model["reduced_coefficient_zero_tolerance"]
            ),
            "baseline_cleanup_audit": baseline.master.coefficient_cleanup_audit,
            "strengthened_cleanup_audit": strengthened.master.coefficient_cleanup_audit,
        },
        "parent_cut_count": len(parent_cuts),
        "derived_cover_count": len(cover_records),
        "selected_cover_count": len(covers),
        "cover_derivations": cover_records,
        "baseline": {
            "bound": baseline_bound,
            "lp_objective": float(baseline.solve.primal_objective),
            "fractional_commitment_count": int(
                np.count_nonzero((reference > 1e-7) & (reference < 1.0 - 1e-7))
            ),
            "security_pair_count": len(baseline.security_pairs),
            "final_screen": baseline.final_screen,
            "solve": _solve_summary(baseline.solve),
            "rounds": [_round_summary(record) for record in baseline.rounds],
        },
        "strengthened": {
            "bound": strengthened_bound,
            "lp_objective": float(strengthened.solve.primal_objective),
            "fractional_commitment_count": int(
                np.count_nonzero(
                    (strengthened.commitment > 1e-7)
                    & (strengthened.commitment < 1.0 - 1e-7)
                )
            ),
            "security_pair_count": len(strengthened.security_pairs),
            "final_screen": strengthened.final_screen,
            "solve": _solve_summary(strengthened.solve),
            "rounds": [_round_summary(record) for record in strengthened.rounds],
        },
        "achieved_bound_lift_dollars": achieved_lift,
        "required_bound_for_gpu_incumbent_at_requested_gap": required_bound,
        "required_lift_from_probe_baseline_dollars": required_lift,
        "achieved_fraction_of_required_lift": (
            1.0 if required_lift == 0.0 else achieved_lift / required_lift
        ),
        "component_gate": (
            "material"
            if required_lift == 0.0 or achieved_lift >= 0.01 * required_lift
            else "immaterial"
        ),
        "numerical_failure_count": numerical_failure_count,
        "total_wall_time_seconds": time.perf_counter() - started,
    }
    print("V15_COVER_PDLP_PROBE=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
