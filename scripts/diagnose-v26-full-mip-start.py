"""Diagnose the native-coordinate residual of a serialized secure GPU incumbent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from activsg_scopf.config import load_config
from activsg_scopf.matpower import read_contingency_table, read_matpower_case
from activsg_scopf.model import build_master
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import security_pair_from_record
from activsg_scopf.screening import add_security_pairs
from activsg_scopf.seeded_diagnostic import deserialize_solution_values
from activsg_scopf.solvers.cuopt import native_scaling_vectors
from activsg_scopf.solvers.cuopt_lp import derive_rate_a_angle_bounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result = json.loads(args.result.read_text(encoding="utf-8"))
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
    pair_records = result["shared_master_evidence"]["security_pairs"]
    pairs = tuple(
        security_pair_from_record(
            record,
            catalog,
            lodf_absolute_tolerance=float(
                config.model["serialized_lodf_replay_tolerance"]
            ),
        )
        for record in pair_records
    )
    add_security_pairs(master.canonical, master.index, network, pairs)
    values = deserialize_solution_values(
        result["secure_incumbent_checkpoint"]["solution"],
        case,
        network,
        master,
    )

    redundant = derive_rate_a_angle_bounds(
        network,
        master.index.theta_by_bus,
        total_columns=master.canonical.num_columns,
    )
    _, lower, upper, _ = master.canonical.column_arrays()
    for column in np.flatnonzero(
        (redundant.lower > lower) | (redundant.upper < upper)
    ):
        position = int(column)
        master.canonical.column_lower[position] = max(
            float(lower[position]), float(redundant.lower[position])
        )
        master.canonical.column_upper[position] = min(
            float(upper[position]), float(redundant.upper[position])
        )

    column_scale, row_scale = native_scaling_vectors(
        master.canonical,
        mode=str(config.raw["platforms"]["dgx_spark"]["native_scaling_mode"]),
        base_mva=float(case.base_mva),
    )
    del column_scale
    activity = np.asarray(master.canonical.matrix_csr() @ values, dtype=np.float64)
    row_lower, row_upper = master.canonical.row_bound_arrays()
    lower_violation = np.where(
        np.isfinite(row_lower), row_lower - activity, -np.inf
    )
    upper_violation = np.where(
        np.isfinite(row_upper), activity - row_upper, -np.inf
    )
    canonical_violation = np.maximum(lower_violation, upper_violation)
    native_violation = canonical_violation * row_scale
    order = np.argsort(native_violation)[::-1][:10]
    records = []
    for row in order:
        position = int(row)
        side = "lower" if lower_violation[position] >= upper_violation[position] else "upper"
        records.append(
            {
                "row": position,
                "name": master.canonical.row_names[position],
                "side": side,
                "activity": float(activity[position]),
                "lower": float(row_lower[position]),
                "upper": float(row_upper[position]),
                "row_scale": float(row_scale[position]),
                "canonical_violation": float(max(canonical_violation[position], 0.0)),
                "canonical_violation_pu": float(
                    max(canonical_violation[position], 0.0) / case.base_mva
                ),
                "native_violation": float(max(native_violation[position], 0.0)),
            }
        )
    print(
        json.dumps(
            {
                "maximum_canonical_violation_pu": float(
                    max(np.max(canonical_violation), 0.0) / case.base_mva
                ),
                "maximum_native_violation": float(max(np.max(native_violation), 0.0)),
                "top_native_rows": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
