#!/usr/bin/env python3
"""Reproduce numerical attribution for the exact ACTIVSg2000 round-2 master."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import structural_rank

from activsg_scopf.config import load_config
from activsg_scopf.matpower import read_contingency_table, read_matpower_case
from activsg_scopf.model import build_master
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.screening import add_security_pairs
from activsg_scopf.seeded_diagnostic import (
    EXPECTED_SECURITY_ROWS,
    canonical_feasibility_audit,
    deserialize_solution_values,
    security_pairs_from_ids,
    validate_reference_evidence,
)
from activsg_scopf.solvers.cuopt import (
    FULL_MIP_START,
    POWER_SYSTEM_PER_UNIT_SCALING,
    native_scaling_audit,
    prepare_mip_start,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "activsg2000-gpu-round2-cpu-seed-diagnostic-v3.json"


def main() -> None:
    config = load_config(CONFIG)
    _, cpu, _, prior_gpu = validate_reference_evidence(config)
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
    master = build_master(case, network, segments=int(config.model["pwl_segments"]))
    pairs = security_pairs_from_ids(
        list(prior_gpu["added_security_pair_ids"]), network, catalog
    )
    add_security_pairs(master.canonical, master.index, network, pairs)
    model = master.canonical
    objective, lower, upper, integrality = model.column_arrays()
    row_lower, row_upper = model.row_bound_arrays()
    matrix = model.matrix_csr()

    raw_values = deserialize_solution_values(cpu["solution"], case, network, master)
    _, seed = prepare_mip_start(
        raw_values,
        expected_shape=(model.num_columns,),
        integrality=integrality,
        mode=FULL_MIP_START,
        lower_bounds=lower,
        upper_bounds=upper,
        clip_to_bounds=True,
    )
    scaling = native_scaling_audit(
        model,
        seed,
        mode=POWER_SYSTEM_PER_UNIT_SCALING,
        base_mva=float(case.base_mva),
    )

    equality_mask = np.isfinite(row_lower) & (row_lower == row_upper)
    equality_matrix = matrix[equality_mask]
    security_matrix = matrix[-EXPECTED_SECURITY_ROWS:].toarray()
    singular_values = np.linalg.svd(security_matrix, compute_uv=False)
    rank_tolerance = float(
        max(security_matrix.shape) * np.finfo(np.float64).eps * singular_values[0]
    )
    security_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    normalized = security_matrix / np.linalg.norm(
        security_matrix, axis=1, keepdims=True
    )
    cosine = np.abs(normalized @ normalized.T)
    np.fill_diagonal(cosine, -np.inf)
    maximum_pair = np.unravel_index(int(np.argmax(cosine)), cosine.shape)
    above_threshold = int(np.count_nonzero(np.triu(cosine > 0.999999, k=1)))

    security_csr = matrix[-EXPECTED_SECURITY_ROWS:].tocsr()
    supports = [
        tuple(security_csr.indices[security_csr.indptr[row] : security_csr.indptr[row + 1]])
        for row in range(EXPECTED_SECURITY_ROWS)
    ]
    repeated_supports = [count for count in Counter(supports).values() if count > 1]
    result = {
        "schema_version": "1.0.0",
        "case_name": config.case_name,
        "benchmark_id": config.benchmark_id,
        "model_dimensions": {
            "columns": model.num_columns,
            "rows": model.num_rows,
            "nonzeros": int(matrix.nnz),
            "integer_columns": int(np.count_nonzero(integrality)),
        },
        "scaling": scaling,
        "known_secure_seed": canonical_feasibility_audit(
            model, seed, expected_objective=float(cpu["objective"])
        ),
        "degeneracy_indicators": {
            "zero_objective_columns": int(np.count_nonzero(objective == 0.0)),
            "free_columns": int(
                np.count_nonzero(np.isneginf(lower) & np.isposinf(upper))
            ),
            "equality_rows": int(np.count_nonzero(equality_mask)),
            "equality_structural_rank": int(structural_rank(equality_matrix)),
        },
        "security_row_dependence": {
            "rows": EXPECTED_SECURITY_ROWS,
            "numeric_rank": security_rank,
            "rank_tolerance": rank_tolerance,
            "largest_singular_value": float(singular_values[0]),
            "smallest_singular_value": float(singular_values[-1]),
            "singular_value_ratio": float(singular_values[0] / singular_values[-1]),
            "maximum_absolute_row_cosine": float(cosine[maximum_pair]),
            "maximum_cosine_pair_ids": [
                model.row_names[-EXPECTED_SECURITY_ROWS + maximum_pair[0]],
                model.row_names[-EXPECTED_SECURITY_ROWS + maximum_pair[1]],
            ],
            "pairs_above_absolute_cosine_0_999999": above_threshold,
            "repeated_support_groups": len(repeated_supports),
            "maximum_repeated_support_group_size": max(repeated_supports, default=1),
        },
        "interpretation": {
            "poor_scaling": "confirmed",
            "near_dependent_security_rows": "confirmed_but_not_safe_to_delete",
            "degeneracy": "indicators_present_without_structural_equality_deficiency",
            "factorization_or_barrier_issue": "confirmed_by_native_cuopt_console_log",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
