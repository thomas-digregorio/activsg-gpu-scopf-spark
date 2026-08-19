#!/usr/bin/env python3
"""Bounded DGX probe of v12 cut cleanup and child Phase-I warm starts.

The probe reconstructs one child that was numerically inconclusive in the
preserved v11 GPU run.  That result is a development fixture only: the
registered v12 experiment never reads it and starts from immutable raw input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from activsg_scopf.commitment_cuts import (
    CommitmentFeasibilityCut,
    build_commitment_cardinality_cut,
    commitment_upper_cut_from_record,
    derive_commitment_feasibility_cut,
)
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.fixed_commitment import build_fixed_commitment_projection
from activsg_scopf.lagrangian import replay_lagrangian_certificate
from activsg_scopf.lagrangian_experiment import (
    RegionAttemptRejected,
    _masks_from_record,
    _prepare_region_master,
    _region_pmin_pmax_capacity_gate,
    _run_phase_one_attempt,
    _solve_phase_one_lagrangian_region,
    _validate_prepared_region_master,
)
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.phase_one import (
    build_phase_one_model,
    phase_one_certificate,
    phase_one_semantic_row_key,
)
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    fix_commitments,
    security_pair_from_record,
)
from activsg_scopf.screening import ContingencyScreener


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_v11_feasibility_cuts(
    *,
    prior: dict[str, Any],
    case: Any,
    network: Any,
    catalog: Any,
    config: Any,
    source_rows: np.ndarray,
) -> tuple[dict[str, CommitmentFeasibilityCut], list[dict[str, Any]]]:
    source_lookup = {int(row): position for position, row in enumerate(source_rows)}
    replacements: dict[str, CommitmentFeasibilityCut] = {}
    audits: list[dict[str, Any]] = []
    for derivation in prior["commitment_feasibility_cuts"]:
        serialized = derivation["cut"]
        binary = np.zeros(source_rows.size, dtype=np.int8)
        binary[
            np.asarray(
                [source_lookup[int(row)] for row in derivation["source_commitment_generator_rows"]],
                dtype=np.int64,
            )
        ] = 1
        if hashlib.sha256(binary.tobytes()).hexdigest() != serialized["source_commitment_sha256"]:
            raise RuntimeError("Development cut fixture commitment hash changed")
        cut_master = build_reduced_master(
            case,
            network,
            segments=int(config.model["pwl_segments"]),
            coefficient_zero_tolerance=float(config.model["reduced_coefficient_zero_tolerance"]),
        )
        pairs = tuple(
            security_pair_from_record(
                record,
                catalog,
                lodf_absolute_tolerance=float(config.model["serialized_lodf_replay_tolerance"]),
            )
            for record in derivation["security_pairs"]
        )
        add_reduced_security_pairs(cut_master, network, pairs)
        fix_commitments(cut_master, binary == 0, binary == 1)
        projection = build_fixed_commitment_projection(cut_master, binary)
        phase_model = build_phase_one_model(
            projection.canonical,
            base_mva=float(case.base_mva),
            maximum_violation_pu=float(config.runtime["phase_one_maximum_violation_pu"]),
        )
        by_semantic = {
            phase_one_semantic_row_key(name): row for row, name in enumerate(phase_model.row_names)
        }
        phase_dual = np.zeros(phase_model.num_rows, dtype=np.float64)
        for item in derivation["nonzero_phase_row_duals"]:
            semantic = f"phase1__{item['side']}__{item['row_name']}"
            phase_dual[by_semantic[semantic]] = float(item["canonical_row_dual"])
        recorded_replay = derivation["phase_one_replay"]
        certificate = phase_one_certificate(
            phase_model,
            phase_dual,
            safety_margin_pu=float(recorded_replay["safety_margin_pu"]),
            infeasibility_threshold_pu=float(recorded_replay["infeasibility_threshold_pu"]),
        )
        cleaned, audit = derive_commitment_feasibility_cut(
            master=cut_master,
            phase_model=phase_model,
            phase_certificate=certificate,
            source_commitment=binary,
            replay_tolerance_pu=float(config.model["phase_one_replay_tolerance_pu"]),
            coefficient_zero_tolerance=float(
                config.runtime["feasibility_cut_coefficient_zero_tolerance"]
            ),
        )
        replacements[str(serialized["cut_id"])] = cleaned
        audits.append(audit["coefficient_cleanup"])
    return replacements, audits


def main() -> None:
    started = time.perf_counter()
    root = Path("/workspace")
    config = load_config(root / "configs" / "activsg2000-gpu-lagrangian-v12.json")
    prior = _read_json(
        root / "results" / "experiments" / "activsg2000-gpu-lagrangian-v11-dgx-spark.json"
    )
    if (
        prior["source_manifest"]["source_identity"]["case_sha256"]
        != config.raw["raw_inputs"]["case_sha256"]
    ):
        raise RuntimeError("Development v11 fixture case hash changed")

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
    cleaned_by_old_id, cleanup_audits = _clean_v11_feasibility_cuts(
        prior=prior,
        case=case,
        network=network,
        catalog=catalog,
        config=config,
        source_rows=source_rows,
    )
    if not any(audit["dropped_coefficient_count"] > 0 for audit in cleanup_audits):
        raise RuntimeError("Real-size v12 cut cleanup did not remove coefficient dust")

    parent_record = next(
        record for record in prior["solved_region_history"] if record["region_id"] == "r_s001_0"
    )
    parent_masks = _masks_from_record(parent_record, source_rows)
    parent_pairs = tuple(
        security_pair_from_record(
            record,
            catalog,
            lodf_absolute_tolerance=float(config.model["serialized_lodf_replay_tolerance"]),
        )
        for record in parent_record["security_pairs"]
    )
    parent_cuts = tuple(
        cleaned_by_old_id.get(
            str(record["cut_id"]),
            commitment_upper_cut_from_record(record, source_rows),
        )
        for record in parent_record["commitment_upper_cuts"]
    )
    parent_master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=parent_masks,
        initial_pairs=parent_pairs,
        commitment_cuts=parent_cuts,
    )
    _validate_prepared_region_master(
        parent_master,
        parent_masks,
        parent_pairs,
        parent_cuts,
    )
    parent_certificate = copy.deepcopy(parent_record["lagrangian_certificate"])
    for item in parent_certificate["commitment_cut_duals"]:
        replacement = cleaned_by_old_id.get(str(item["cut_id"]))
        if replacement is not None:
            item["cut_id"] = replacement.cut_id
    parent_lagrangian = replay_lagrangian_certificate(
        parent_master,
        parent_certificate,
        parent_masks,
        commitment_cuts_by_id={cut.cut_id: cut for cut in parent_cuts},
    )
    parent = SimpleNamespace(
        master=parent_master,
        lagrangian=parent_lagrangian,
        commitment_cuts=parent_cuts,
    )

    failed = prior["failed_disjunctive_split_attempts"][0]
    source_lookup = {int(row): position for position, row in enumerate(source_rows)}
    subset_positions = np.asarray(
        [source_lookup[int(row)] for row in failed["subset_source_rows"]],
        dtype=np.int64,
    )
    child_cut = build_commitment_cardinality_cut(
        generator_source_rows=source_rows,
        subset_positions=subset_positions,
        subset_id=str(failed["subset_id"]),
        branch_side="at_most",
        integer_threshold=int(failed["floor_value"]),
    )
    if child_cut.cut_id != failed["at_most_cut_id"]:
        raise RuntimeError("Development child cardinality identity changed")
    child_cuts = parent_cuts + (child_cut,)
    child_master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=parent_masks,
        initial_pairs=parent_pairs,
        commitment_cuts=child_cuts,
    )
    _validate_prepared_region_master(
        child_master,
        parent_masks,
        parent_pairs,
        child_cuts,
    )
    capacity_gate = _region_pmin_pmax_capacity_gate(
        case=case,
        master=child_master,
        masks=parent_masks,
        tolerance_pu=float(config.model["model_residual_tolerance_pu"]),
    )
    rejected = RegionAttemptRejected(
        "v12 bounded child probe",
        reason="phase_one_first_precheck",
        master=child_master,
        security_pairs=parent_pairs,
        rounds=[],
    )
    deadline = Deadline(45.0, 0.0, 0.0)
    phase = _run_phase_one_attempt(
        region_id=str(failed["off_child_region_id"]),
        masks=parent_masks,
        rejected=rejected,
        case=case,
        config=config,
        deadline=deadline,
        attempt_kind="pre_cost_lp",
        time_limit_seconds=float(config.runtime["precheck_phase_one_time_limit_seconds"]),
        capacity_gate=capacity_gate,
        commitment_cuts=child_cuts,
    )
    if phase.record["prune_certified"] or phase.source_native_primal is None:
        raise RuntimeError("v12 child Phase I did not return its expected feasible primal")
    child, prune = _solve_phase_one_lagrangian_region(
        region_id=str(failed["off_child_region_id"]),
        masks=parent_masks,
        parent=parent,
        master=child_master,
        initial_pairs=parent_pairs,
        initial_precheck=phase,
        case=case,
        network=network,
        config=config,
        deadline=deadline,
        screener=ContingencyScreener(
            network,
            catalog,
            backend="cupy",
            chunk_columns=int(config.model["screen_chunk_columns"]),
        ),
        checkpoint=lambda: None,
        commitment_cuts=child_cuts,
    )
    if prune is not None or child is None:
        raise RuntimeError("v12 preserved-primal child engine did not return a solved child")
    matrix_data = np.abs(child_master.canonical.matrix_csr().data)
    cost_summary = child.solve.statistics.get("cost_pdlp_solve", {})
    warm = cost_summary.get("warm_start", {})
    if not warm.get("initial_primal_submitted") or not warm.get("initial_dual_submitted"):
        raise RuntimeError("v12 child cost-dual solve did not consume both registered starts")
    if child.solve.statistics.get("cost_pdlp_returned_primal_used") is not False:
        raise RuntimeError("v12 child used the non-authoritative cost-PDLP primal")
    if child.final_screen["new_violated_pairs"] != 0:
        raise RuntimeError("v12 child did not preserve an exhaustively screened Phase-I primal")
    print(
        "V12_PRESERVED_PHASE_ONE_PROBE="
        + json.dumps(
            {
                "passed": True,
                "development_component_only": True,
                "registered_v12_reads_prior_result": False,
                "fixture": "preserved_gpu_v11_child_only",
                "child_region_id": child.region_id,
                "cleaned_global_cut_count": len(cleaned_by_old_id),
                "dropped_cut_coefficient_count": sum(
                    int(audit["dropped_coefficient_count"]) for audit in cleanup_audits
                ),
                "minimum_nonzero_child_matrix_coefficient": float(
                    np.min(matrix_data[matrix_data > 0.0])
                ),
                "phase_one": {
                    "adapter_wall_time_seconds": phase.record["adapter_wall_time_seconds"],
                    "registered_optimality_tolerance": phase.record[
                        "registered_optimality_tolerance"
                    ],
                    "source_model_residual_pu": phase.record["source_feasible_warm_start"][
                        "source_model_residual_pu"
                    ],
                },
                "preserved_phase_one_primal": {
                    "status": child.solve.status,
                    "canonical_model_residual_pu": child.master.canonical.max_row_violation(
                        np.asarray(child.solve.values, dtype=np.float64)
                    )
                    / float(case.base_mva),
                    "final_exhaustive_screen": child.final_screen,
                    "cost_pdlp_returned_primal_used": child.solve.statistics[
                        "cost_pdlp_returned_primal_used"
                    ],
                },
                "cost_dual_seed": {
                    "adapter_wall_time_seconds": child.rounds[-1]["cost_dual_seed"][
                        "adapter_wall_time_seconds"
                    ],
                    "dual_seed_eligible": child.rounds[-1]["cost_dual_seed"][
                        "dual_seed_eligible"
                    ],
                    "selected_seed": child.rounds[-1]["cost_dual_seed"]["selected_seed"],
                    "warm_start": warm,
                    "native_log_audit": cost_summary.get("native_log_audit"),
                },
                "parent_dual_mapping": child.gpu_lagrangian["parent_dual_mapping"],
                "certificate_bound": child.lagrangian.conservative_lower_bound,
                "total_probe_wall_time_seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
