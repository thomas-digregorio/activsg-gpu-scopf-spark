"""Build tracked evidence for the single ACTIVSg2000 DGX Spark 1e-3 run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "experiments"
REPORT_ROOT = ROOT / "reports" / "activsg2000-gpu-1e-3-v1"
CPU_PATH = RESULT_ROOT / "activsg2000-gap-v1-1e-3-laptop.json"
GPU_PATH = RESULT_ROOT / "activsg2000-gpu-gap-v1-1e-3-dgx-spark.json"
VERIFY_PATH = ROOT / "work" / "activsg2000-gpu-1e-3-posthoc-verification.json"
CPU_COMMIT = "173fd0c10b8af5ed431956f2eea9726032df6f9e"
GPU_COMMIT = "c96d73532609c028ac34e8e41132efa28a249859"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload["_bytes"] = path.stat().st_size
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _difference(gpu: list[float], cpu: list[float]) -> dict[str, float | int]:
    values = [float(left) - float(right) for left, right in zip(gpu, cpu, strict=True)]
    absolute = [abs(value) for value in values]
    return {
        "count_above_1e-6": sum(value > 1e-6 for value in absolute),
        "count_above_1e-2": sum(value > 1e-2 for value in absolute),
        "count_above_1": sum(value > 1.0 for value in absolute),
        "sum_absolute": sum(absolute),
        "mean_absolute": sum(absolute) / len(absolute),
        "root_mean_square": math.sqrt(
            sum(value * value for value in values) / len(values)
        ),
        "maximum_absolute": max(absolute, default=0.0),
    }


def _audit_identity(
    cpu: dict[str, Any], gpu: dict[str, Any], verification: dict[str, Any]
) -> None:
    if cpu.get("status") != "optimal_verified":
        raise ValueError("Laptop comparison result is not optimal_verified")
    if cpu.get("pricing", {}).get("status") != "optimal_secure_fixed_commitment_lp":
        raise ValueError("Laptop comparison result lacks accepted pricing")
    if gpu.get("status") != "incomplete_restricted_master_not_optimal":
        raise ValueError("Unexpected GPU result status")
    if gpu.get("pricing") is not None:
        raise ValueError("Incomplete GPU result unexpectedly contains pricing")
    if len(gpu.get("constraint_generation_rounds", [])) != 1:
        raise ValueError("GPU run did not stop after exactly one recorded round")
    gpu_solve = gpu["constraint_generation_rounds"][0]["solve"]
    if gpu_solve.get("status") != "FeasibleFound":
        raise ValueError("GPU round did not preserve FeasibleFound status")
    if not gpu_solve["statistics"].get("reported_gap_meets_request"):
        raise ValueError("GPU reported gap does not meet the configured request")
    if gpu["frozen_identity"]["commit"] != GPU_COMMIT:
        raise ValueError("GPU result differs from the frozen commit")
    if gpu["frozen_identity"]["tag"] != "experiment-2000-gpu-gap-v1":
        raise ValueError("GPU result differs from the frozen tag")
    if cpu["frozen_identity"]["commit"] != CPU_COMMIT:
        raise ValueError("CPU comparison result differs from its frozen commit")
    identities = {
        (
            result["source_manifest"]["source_identity"]["case_sha256"],
            result["source_manifest"]["source_identity"]["contingency_sha256"],
        )
        for result in (cpu, gpu)
    }
    if len(identities) != 1:
        raise ValueError("CPU and GPU source hashes differ")
    if verification.get("passed") is not False:
        raise ValueError("Post-hoc verification must preserve the security failure")
    if int(verification["checked_valid_outages"]) != 2740:
        raise ValueError("Post-hoc verification checked the wrong outage count")


def main() -> None:
    cpu = _load(CPU_PATH)
    gpu = _load(GPU_PATH)
    verification = _load(VERIFY_PATH)
    _audit_identity(cpu, gpu, verification)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    source = gpu["source_manifest"]["generators"]
    cpu_generators = cpu["solution"]["generators"]
    gpu_generators = gpu["solution"]["generators"]
    cpu_pricing = cpu["pricing"]["generators"]
    cpu_prices = cpu["pricing"]["bus_prices"]
    if not len(source) == len(cpu_generators) == len(gpu_generators) == 544:
        raise ValueError("Expected 544 generator records")
    if len(cpu_pricing) != 544 or len(cpu_prices) != 2000:
        raise ValueError("Laptop comparison evidence has the wrong dimensions")
    base_mva = float(cpu["pricing"]["base_mva"])

    audit = {
        "source_limit_mismatches": 0,
        "conditional_limit_violations_above_1e-6_mw": 0,
        "source_offline_availability_violations": 0,
        "dispatch_pu_conversion_mismatches": 0,
    }
    generator_rows: list[dict[str, Any]] = []
    cpu_commitments: list[int] = []
    gpu_commitments: list[int] = []
    for source_row, cpu_row, gpu_row, cpu_price in zip(
        source, cpu_generators, gpu_generators, cpu_pricing, strict=True
    ):
        if not (
            source_row["source_id"]
            == cpu_row["source_id"]
            == gpu_row["source_id"]
            == cpu_price["source_id"]
        ):
            raise ValueError("Generator source-row identity changed")
        for record in (cpu_row, gpu_row):
            if any(
                float(record[field]) != float(source_row[field])
                for field in ("pmin_mw", "pmax_mw")
            ):
                audit["source_limit_mismatches"] += 1
        gpu_commitment = round(float(gpu_row["commitment"]))
        cpu_commitment = round(float(cpu_row["commitment"]))
        gpu_dispatch = float(gpu_row["dispatch_mw"])
        if not (
            float(gpu_row["pmin_mw"]) * gpu_commitment - 1e-6
            <= gpu_dispatch
            <= float(gpu_row["pmax_mw"]) * gpu_commitment + 1e-6
        ):
            audit["conditional_limit_violations_above_1e-6_mw"] += 1
        if int(gpu_row["source_status"]) <= 0 and (
            gpu_commitment != 0 or abs(gpu_dispatch) > 1e-6
        ):
            audit["source_offline_availability_violations"] += 1
        if abs(float(gpu_row["dispatch_pu"]) - gpu_dispatch / base_mva) > 1e-10:
            audit["dispatch_pu_conversion_mismatches"] += 1
        cpu_commitments.append(cpu_commitment)
        gpu_commitments.append(gpu_commitment)
        generator_rows.append(
            {
                "source_id": source_row["source_id"],
                "source_row": source_row["source_row"],
                "bus": source_row["bus"],
                "source_status": source_row["source_status"],
                "pmin_mw": source_row["pmin_mw"],
                "pmin_pu": float(source_row["pmin_mw"]) / base_mva,
                "pmax_mw": source_row["pmax_mw"],
                "pmax_pu": float(source_row["pmax_mw"]) / base_mva,
                "cpu_accepted_commitment": cpu_commitment,
                "cpu_accepted_mip_dispatch_mw": cpu_row["dispatch_mw"],
                "cpu_accepted_mip_dispatch_pu": cpu_row["dispatch_pu"],
                "cpu_accepted_pricing_dispatch_mw": cpu_price[
                    "pricing_dispatch_mw"
                ],
                "cpu_accepted_pricing_dispatch_pu": cpu_price[
                    "pricing_dispatch_pu"
                ],
                "cpu_accepted_price_per_mwh": cpu_price["nodal_price_per_mwh"],
                "cpu_accepted_price_per_pu_hour": cpu_price[
                    "nodal_price_per_pu_hour"
                ],
                "gpu_provisional_commitment": gpu_commitment,
                "gpu_provisional_mip_dispatch_mw": gpu_row["dispatch_mw"],
                "gpu_provisional_mip_dispatch_pu": gpu_row["dispatch_pu"],
                "gpu_pricing_status": "not_reached",
                "gpu_pricing_dispatch_mw": None,
                "gpu_pricing_dispatch_pu": None,
                "gpu_price_per_mwh": None,
                "gpu_price_per_pu_hour": None,
            }
        )
    if any(audit.values()):
        raise ValueError(f"Exact-PMIN or unit audit failed: {audit}")
    _write_csv(REPORT_ROOT / "generator-detail.csv", generator_rows)

    _write_csv(
        REPORT_ROOT / "bus-prices.csv",
        [
            {
                "bus": record["bus"],
                "cpu_accepted_price_per_mwh": record["price_per_mwh"],
                "cpu_accepted_price_per_pu_hour": record["price_per_pu_hour"],
                "gpu_price_status": "not_reached",
                "gpu_price_per_mwh": None,
                "gpu_price_per_pu_hour": None,
            }
            for record in cpu_prices
        ],
    )

    summary_rows = [
        {
            "platform": "laptop_cpu",
            "result_status": cpu["status"],
            "objective": cpu["objective"],
            "bound": cpu["bound"],
            "mip_gap": cpu["mip_gap"],
            "commitment_count": cpu["commitment_count"],
            "constraint_generation_rounds": cpu["constraint_generation_round_count"],
            "added_security_pairs": cpu["added_security_pairs"],
            "last_screen_new_violations": 0,
            "final_security_violation_pu": cpu["final_exhaustive_violation_pu"],
            "pricing_status": cpu["pricing"]["status"],
            "total_wall_seconds": cpu["total_wall_time_seconds"],
            "raw_result_sha256": cpu["_sha256"],
        },
        {
            "platform": "dgx_spark",
            "result_status": gpu["status"],
            "objective": gpu["objective"],
            "bound": gpu["bound"],
            "mip_gap": gpu["mip_gap"],
            "commitment_count": sum(gpu_commitments),
            "constraint_generation_rounds": gpu["constraint_generation_round_count"],
            "added_security_pairs": gpu["added_security_pairs"],
            "last_screen_new_violations": gpu["last_completed_screen"][
                "new_violated_pairs"
            ],
            "final_security_violation_pu": verification[
                "maximum_security_violation_pu"
            ],
            "pricing_status": "not_reached",
            "total_wall_seconds": gpu["total_wall_time_seconds"],
            "raw_result_sha256": gpu["_sha256"],
        },
    ]
    _write_csv(REPORT_ROOT / "summary.csv", summary_rows)

    round_rows: list[dict[str, Any]] = []
    for platform_name, result in (("laptop_cpu", cpu), ("dgx_spark", gpu)):
        for record in result["constraint_generation_rounds"]:
            solve = record["solve"]
            statistics = solve["statistics"]
            screen = record["screen"]
            round_rows.append(
                {
                    "platform": platform_name,
                    "round": record["round"],
                    "solver": solve["solver"],
                    "status": solve["status"],
                    "native_status": statistics.get("native_status"),
                    "accepted_optimal": solve["optimal"],
                    "objective": solve["objective"],
                    "bound": solve["bound"],
                    "mip_gap": solve["mip_gap"],
                    "requested_mip_gap": statistics.get(
                        "requested_mip_relative_gap"
                    ),
                    "reported_gap_meets_request": statistics.get(
                        "reported_gap_meets_request"
                    ),
                    "partial_integer_mip_start_columns": statistics.get(
                        "partial_integer_mip_start_columns"
                    ),
                    "adapter_wall_seconds": record["adapter_wall_time_seconds"],
                    "native_solve_seconds": solve["solve_time_seconds"],
                    "screen_wall_seconds": screen["wall_time_seconds"],
                    "screen_new_violated_pairs": screen["new_violated_pairs"],
                    "screen_maximum_violation_pu": screen[
                        "maximum_violation_pu"
                    ],
                }
            )
    _write_csv(REPORT_ROOT / "round-detail.csv", round_rows)

    commitment_pairs = list(zip(cpu_commitments, gpu_commitments, strict=True))
    diagnostic_comparison = {
        "comparison_status": "not_a_final_grid_solution_comparison",
        "reason": (
            "The DGX Spark stopped after an unaccepted base restricted-master "
            "round and retained 165 unresolved contingency violations."
        ),
        "commitment_flip_count": sum(left != right for left, right in commitment_pairs),
        "cpu_off_gpu_on_count": sum(
            left == 0 and right == 1 for left, right in commitment_pairs
        ),
        "cpu_on_gpu_off_count": sum(
            left == 1 and right == 0 for left, right in commitment_pairs
        ),
        "mip_dispatch_mw": _difference(
            [float(record["dispatch_mw"]) for record in gpu_generators],
            [float(record["dispatch_mw"]) for record in cpu_generators],
        ),
        "pricing_comparison_available": False,
    }
    comparison = {
        "schema_version": "1.0.0",
        "campaign": "activsg2000-gpu-gap-sensitivity-v1",
        "campaign_status": "stopped_after_single_incomplete_run",
        "retry_performed": False,
        "source_identity": gpu["source_manifest"]["source_identity"],
        "source_dimensions": gpu["source_manifest"]["dimensions"],
        "source_online_capacity_mw": gpu["source_manifest"][
            "source_online_capacity_mw"
        ],
        "cpu_frozen_identity": cpu["frozen_identity"],
        "gpu_frozen_identity": gpu["frozen_identity"],
        "raw_evidence": {
            "cpu": {"sha256": cpu["_sha256"], "bytes": cpu["_bytes"]},
            "gpu": {"sha256": gpu["_sha256"], "bytes": gpu["_bytes"]},
            "gpu_posthoc_verification": {
                "sha256": verification["_sha256"],
                "bytes": verification["_bytes"],
            },
        },
        "summary": summary_rows,
        "gpu_round_1": round_rows[-1],
        "gpu_posthoc_verification": {
            key: value
            for key, value in verification.items()
            if not key.startswith("_")
        },
        "record_audit": audit,
        "cpu_accepted_vs_gpu_provisional": diagnostic_comparison,
    }
    (REPORT_ROOT / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cpu_table_row = (
        f"| Laptop CPU | `optimal_verified` | {cpu['objective']:,.6f} | "
        f"{cpu['bound']:,.6f} | {cpu['mip_gap']:.6e} | "
        f"{cpu['commitment_count']} | "
        f"{cpu['constraint_generation_round_count']} | "
        f"{cpu['total_wall_time_seconds']:.3f} | accepted |"
    )
    gpu_table_row = (
        "| DGX Spark | `incomplete_restricted_master_not_optimal` | "
        f"{gpu['objective']:,.6f} | {gpu['bound']:,.6f} | "
        f"{gpu['mip_gap']:.6e} | {sum(gpu_commitments)} provisional | 1 | "
        f"{gpu['total_wall_time_seconds']:.3f} | not reached |"
    )
    readme = f"""# ACTIVSg2000 DGX Spark 1e-3 result

The single authorized run completed in {gpu['total_wall_time_seconds']:.3f}
seconds but did **not** produce an accepted SCOPF solution. cuOpt returned native
status `FeasibleFound` for the base restricted master. Its reported gap was
{gpu['mip_gap']:.6e}, below the requested `1e-3`, but the frozen fail-closed
adapter requires native `Optimal` before promoting contingency rows.

The controller still exhaustively screened the incumbent. It found
{gpu['last_completed_screen']['new_violated_pairs']} violated contingency pairs,
with a maximum violation of
{gpu['last_completed_screen']['maximum_violation_pu']:.6f} p.u., then stopped.
No security rows were added, independent verification was not reached inside the
run, and fixed-commitment pricing was not run.

| Platform | Status | Objective | Bound | Gap | Committed | Rounds | Wall (s) | Pricing |
|---|---|---:|---:|---:|---:|---:|---:|---|
{cpu_table_row}
{gpu_table_row}

## Independent post-hoc check

The raw-input checker was run once on the saved incumbent without another MIP
solve. Base-model residuals pass: the maximum is
{verification['maximum_model_residual_pu']:.3e} p.u., and the conditional
PMIN/PMAX residual is
{verification['details']['conditional_pmin_pmax_violation_pu']:.3e} p.u. The
checker fails N-1 security across
{verification['checked_security_sides']:,} sides with a maximum violation of
{verification['maximum_security_violation_pu']:.6f} p.u.

Thus the provisional commitment and dispatch are not comparable to the accepted
laptop grid solution. The lower GPU objective reflects a base restricted master,
not a better secure solution. GPU nodal prices do not exist for this run.

## Timing attribution

- Model and factor build: {gpu['timings_seconds']['model_and_factor_build']:.3f} s
- cuOpt round 1: {gpu['timings_seconds']['solver_rounds']:.3f} s
- CuPy exhaustive screen: {gpu['timings_seconds']['screening']:.3f} s
- End to end: {gpu['total_wall_time_seconds']:.3f} s
- Fresh suite-specific CUDA/CuPy cache; no full-case warmup

NVIDIA's [cuOpt MIP settings documentation][cuopt-mip-settings] describes
`mip_relative_gap` as a termination tolerance and notes that cuOpt MIP
optimality proofs remain under active development. Reclassifying
`FeasibleFound` when its reported gap passes would change the adapter acceptance
policy and requires a new frozen run identity; this run was not retried.

## Evidence files

- `generator-detail.csv`: all 544 exact source PMIN/PMAX rows, the accepted CPU
  solution, and the clearly labeled provisional GPU commitment/dispatch. GPU
  pricing fields are blank.
- `bus-prices.csv`: all 2,000 accepted CPU prices; GPU price fields are blank.
- `round-detail.csv`: CPU and GPU solver/screening attribution.
- `summary.csv` and `comparison.json`: frozen identities, raw hashes, post-hoc
  verification, record audits, and the explicit no-retry status.

The source audit found zero PMIN/PMAX substitutions, conditional limit
violations above `1e-6` MW, source-offline availability violations, or MW/p.u.
conversion mismatches.

[cuopt-mip-settings]: https://docs.nvidia.com/cuopt/user-guide/latest/mip-settings.html
"""
    (REPORT_ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
