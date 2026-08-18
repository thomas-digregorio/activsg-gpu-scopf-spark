#!/usr/bin/env python3
"""Bounded DGX probe for v10 cardinality rows and dual-only child starts.

This is a development component check, not an end-to-end benchmark.  It uses
the preserved GPU v15 LP primal only to avoid paying for another root solve
from a cold vector.  The registered v10 experiment does not read this file or
any prior result and therefore starts from the immutable raw inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from activsg_scopf.cardinality import choose_cardinality_split, commitment_branch_subsets
from activsg_scopf.config import load_config
from activsg_scopf.lagrangian import (
    RegionMasks,
    evaluate_lagrangian_bound,
    evaluate_lagrangian_bound_cupy,
)
from activsg_scopf.lagrangian_experiment import (
    _map_native_row_dual_by_identity,
    _prepare_region_master,
    _validate_prepared_region_master,
)
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import (
    add_reduced_security_pairs,
    build_reduced_master,
    security_pair_from_record,
)
from activsg_scopf.solvers.cuopt import native_scaling_vectors
from activsg_scopf.solvers.cuopt_lp import solve_cuopt_continuous_pdlp


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reduced_values_from_serialized_solution(master: Any, solution: dict[str, Any]) -> np.ndarray:
    by_row = {int(record["source_row"]): record for record in solution["generators"]}
    values = np.zeros(master.canonical.num_columns, dtype=np.float64)
    for source_index in master.index.generator_source_rows:
        source_row = int(source_index) + 1
        record = by_row[source_row]
        values[master.index.commitment_by_generator[int(source_index)]] = float(
            record["commitment"]
        )
        values[master.index.dispatch_by_generator[int(source_index)]] = float(
            record["dispatch_mw"]
        )
        for column, dispatch in zip(
            master.index.segments_by_generator[int(source_index)],
            record["segment_dispatch_mw"],
            strict=True,
        ):
            if column is not None:
                values[column] = float(dispatch)
    return values


def _numeric_warning_counts(result: Any) -> dict[str, int]:
    audit = result.statistics.get("native_log_audit", {})
    return {
        key: int(audit.get(key, 0))
        for key in (
            "barrier_numerical_warning_count",
            "free_variable_warning_count",
            "mip_start_rejection_count",
        )
    }


def main() -> None:
    root = Path("/workspace")
    config = load_config(root / "configs" / "activsg2000-gpu-lagrangian-v10.json")
    gpu_lp = _read_json(
        root
        / "results"
        / "experiments"
        / "activsg2000-gpu-lp-certificate-v15-dgx-spark.json"
    )
    gpu_v9 = _read_json(
        root
        / "results"
        / "experiments"
        / "activsg2000-gpu-lagrangian-v9-dgx-spark.json"
    )
    if gpu_lp["source_manifest"]["source_identity"]["case_sha256"] != config.raw[
        "raw_inputs"
    ]["case_sha256"]:
        raise RuntimeError("GPU LP component fixture case hash changed")
    if gpu_v9["source_manifest"]["source_identity"]["case_sha256"] != config.raw[
        "raw_inputs"
    ]["case_sha256"]:
        raise RuntimeError("GPU v9 component fixture case hash changed")

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
    root_record = gpu_v9["solved_region_history"][0]
    pairs = tuple(
        security_pair_from_record(
            record,
            catalog,
            lodf_absolute_tolerance=float(config.model["serialized_lodf_replay_tolerance"]),
        )
        for record in root_record["security_pairs"]
    )
    root_master = build_reduced_master(
        case,
        network,
        segments=int(config.model["pwl_segments"]),
        coefficient_zero_tolerance=float(config.model["reduced_coefficient_zero_tolerance"]),
    )
    add_reduced_security_pairs(
        root_master,
        network,
        pairs,
        expected_representative_by_pair_id=root_record["security_row_equivalence"][
            "representative_by_pair_id"
        ],
        equivalence_replay_tolerance=float(
            config.model["security_equivalence_replay_tolerance"]
        ),
    )
    root_values = _reduced_values_from_serialized_solution(root_master, gpu_lp["solution"])
    root_residual_pu = root_master.canonical.max_row_violation(root_values) / float(
        case.base_mva
    )
    column_scale, _ = native_scaling_vectors(
        root_master.canonical,
        mode=str(config.raw["platforms"]["dgx_spark"]["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    root_solve = solve_cuopt_continuous_pdlp(
        root_master.canonical,
        time_limit_seconds=15.0,
        optimality_tolerance=float(
            config.raw["platforms"]["dgx_spark"]["pdlp_optimality_tolerance"]
        ),
        primal_feasibility_tolerance=float(config.model["model_residual_tolerance_pu"]),
        certificate_residual_tolerance=float(
            config.raw["platforms"]["dgx_spark"]["dual_certificate_residual_tolerance"]
        ),
        native_scaling_mode=str(
            config.raw["platforms"]["dgx_spark"]["native_scaling_mode"]
        ),
        native_base_mva=float(case.base_mva),
        log_to_console=True,
        per_constraint_residual=True,
        presolve=0,
        initial_native_primal=root_values / column_scale,
        pdlp_solver_mode=1,
    )
    if root_solve.native_row_dual is None:
        raise RuntimeError("Root component solve returned no native row dual")

    online_rows = np.flatnonzero(case.gen[:, GEN_STATUS] > 0).astype(np.int64) + 1
    commitment_by_row = {
        int(record["source_row"]): float(record["commitment"])
        for record in gpu_lp["solution"]["generators"]
    }
    lp_commitment = np.asarray(
        [commitment_by_row[int(row)] for row in online_rows], dtype=np.float64
    )
    subsets = commitment_branch_subsets(root_master)
    split = choose_cardinality_split(
        master=root_master,
        commitments=lp_commitment,
        subsets=subsets,
        existing_cut_ids=set(),
    )
    if len(split.subset.source_rows) < 2:
        raise RuntimeError("Cardinality probe selected a singleton subset")
    child_records: list[dict[str, Any]] = []
    for side, cut in (
        ("at_most", split.at_most_cut),
        ("at_least", split.at_least_cut),
    ):
        child = _prepare_region_master(
            case=case,
            network=network,
            config=config,
            masks=RegionMasks.root(online_rows.size),
            initial_pairs=pairs,
            commitment_cuts=(cut,),
        )
        _validate_prepared_region_master(
            child,
            RegionMasks.root(online_rows.size),
            pairs,
            (cut,),
        )
        mapped_dual, mapping = _map_native_row_dual_by_identity(
            root_master.canonical,
            child.canonical,
            np.asarray(root_solve.native_row_dual, dtype=np.float64),
            scaling_mode=str(
                config.raw["platforms"]["dgx_spark"]["native_scaling_mode"]
            ),
            base_mva=float(case.base_mva),
        )
        child_solve = solve_cuopt_continuous_pdlp(
            child.canonical,
            time_limit_seconds=8.0,
            optimality_tolerance=float(
                config.raw["platforms"]["dgx_spark"]["pdlp_optimality_tolerance"]
            ),
            primal_feasibility_tolerance=float(config.model["model_residual_tolerance_pu"]),
            certificate_residual_tolerance=float(
                config.raw["platforms"]["dgx_spark"]["dual_certificate_residual_tolerance"]
            ),
            native_scaling_mode=str(
                config.raw["platforms"]["dgx_spark"]["native_scaling_mode"]
            ),
            native_base_mva=float(case.base_mva),
            log_to_console=True,
            per_constraint_residual=True,
            presolve=0,
            initial_native_row_dual=mapped_dual,
            pdlp_solver_mode=1,
        )
        warnings = _numeric_warning_counts(child_solve)
        if any(warnings.values()):
            raise RuntimeError(f"Child {side} emitted a native numerical warning: {warnings}")
        warm = child_solve.statistics["warm_start"]
        if warm["initial_primal_submitted"] or not warm["initial_dual_submitted"]:
            raise RuntimeError(f"Child {side} violated the dual-only warm-start contract")
        child_records.append(
            {
                "side": side,
                "status": child_solve.status,
                "error_status": child_solve.statistics["error_status"],
                "solution_vectors_available": child_solve.statistics[
                    "solution_vectors_available"
                ],
                "numeric_warning_counts": warnings,
                "warm_start": warm,
                "dual_mapping": mapping,
                "cut_row_scale": native_scaling_vectors(
                    child.canonical,
                    mode=str(
                        config.raw["platforms"]["dgx_spark"]["native_scaling_mode"]
                    ),
                    base_mva=float(case.base_mva),
                )[1][child.canonical.row_names.index(cut.cut_id)],
            }
        )

    zero_dual = np.zeros(root_master.canonical.num_rows, dtype=np.float64)
    cut_dual = np.asarray([-1.0], dtype=np.float64)
    gpu_replay = evaluate_lagrangian_bound_cupy(
        root_master,
        zero_dual,
        RegionMasks.root(online_rows.size),
        commitment_cuts=(split.at_most_cut,),
        commitment_cut_dual=cut_dual,
    )
    cpu_replay = evaluate_lagrangian_bound(
        root_master,
        zero_dual,
        RegionMasks.root(online_rows.size),
        safety_margin_dollars=0.0,
        commitment_cuts=(split.at_most_cut,),
        commitment_cut_dual=cut_dual,
    )
    replay_difference = abs(
        float(gpu_replay["raw_lower_bound"]) - cpu_replay.raw_lower_bound
    )
    if replay_difference > 1e-6:
        raise RuntimeError("GPU/CPU cardinality-cut Lagrangian replay mismatch")

    print(
        "V10_CARDINALITY_PROBE="
        + json.dumps(
            {
                "passed": True,
                "development_component_only": True,
                "registered_v10_reads_prior_result": False,
                "root_fixture_source": "preserved_gpu_lp_v15_only",
                "root_fixture_residual_pu": root_residual_pu,
                "root_solve_status": root_solve.status,
                "root_numeric_warning_counts": _numeric_warning_counts(root_solve),
                "split": split.as_dict(),
                "children": child_records,
                "gpu_cpu_cut_replay_difference_dollars": replay_difference,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
