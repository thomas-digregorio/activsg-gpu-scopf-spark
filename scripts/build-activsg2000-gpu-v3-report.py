"""Build tracked evidence for the bounded ACTIVSg2000 DGX Spark v3 run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "experiments"
REPORT_ROOT = ROOT / "reports" / "activsg2000-gpu-1e-3-v3"
CPU_PATH = RESULT_ROOT / "activsg2000-gap-v1-1e-3-laptop.json"
GPU_PATH = RESULT_ROOT / "activsg2000-gpu-gap-v3-1e-3-dgx-spark.json"
VERIFY_PATH = ROOT / "work" / "activsg2000-gpu-v3-round1-posthoc-verification.json"
CPU_COMMIT = "173fd0c10b8af5ed431956f2eea9726032df6f9e"
GPU_COMMIT = "921ab8e59ce71866ba004127d73278d7918c1d9c"


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


def _audit_identity(
    cpu: dict[str, Any], gpu: dict[str, Any], verification: dict[str, Any]
) -> None:
    if cpu.get("status") != "optimal_verified":
        raise ValueError("Laptop comparison result is not optimal_verified")
    if cpu.get("pricing", {}).get("status") != "optimal_secure_fixed_commitment_lp":
        raise ValueError("Laptop comparison result lacks accepted pricing")
    if gpu.get("status") != "incomplete_no_incumbent":
        raise ValueError("Unexpected GPU v3 result status")
    if gpu.get("pricing") is not None or gpu.get("verification") is not None:
        raise ValueError("Incomplete GPU v3 result unexpectedly reached a final stage")
    rounds = gpu.get("constraint_generation_rounds", [])
    if len(rounds) != 2:
        raise ValueError("GPU v3 result must preserve exactly two rounds")
    first, second = rounds
    if first["solve"].get("status") != "FeasibleFound":
        raise ValueError("GPU v3 round 1 did not preserve FeasibleFound")
    if first["solve"].get("requested_gap_certified") is not True:
        raise ValueError("GPU v3 round 1 gap was not certified")
    if first.get("screen", {}).get("new_violated_pairs") != 173:
        raise ValueError("GPU v3 round 1 did not preserve all screened rows")
    if len(first.get("added_pair_ids", [])) != 173:
        raise ValueError("GPU v3 round 1 did not add every violated pair")
    if second["solve"].get("status") != "Infeasible":
        raise ValueError("GPU v3 round 2 did not preserve native Infeasible")
    if second["solve"].get("has_incumbent") is not False:
        raise ValueError("GPU v3 round 2 unexpectedly has an incumbent")
    if second.get("screen") is not None:
        raise ValueError("GPU v3 round 2 must not have a screen")
    if second["solve"]["statistics"].get(
        "partial_integer_mip_start_columns"
    ) != 432:
        raise ValueError("GPU v3 round 2 did not receive the prior commitment")
    if gpu["frozen_identity"]["commit"] != GPU_COMMIT:
        raise ValueError("GPU v3 result differs from the frozen commit")
    if gpu["frozen_identity"]["tag"] != "experiment-2000-gpu-gap-v3":
        raise ValueError("GPU v3 result differs from the frozen tag")
    if cpu["frozen_identity"]["commit"] != CPU_COMMIT:
        raise ValueError("CPU comparison result differs from its frozen commit")
    source_hashes = {
        (
            result["source_manifest"]["source_identity"]["case_sha256"],
            result["source_manifest"]["source_identity"]["contingency_sha256"],
        )
        for result in (cpu, gpu)
    }
    if len(source_hashes) != 1:
        raise ValueError("CPU and GPU source hashes differ")
    if verification.get("passed") is not False:
        raise ValueError("Post-hoc round-1 verification must fail N-1 security")
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
    gpu_commitment_count = 0
    for source_row, cpu_row, gpu_row, cpu_price in zip(
        source, cpu_generators, gpu_generators, cpu_pricing, strict=True
    ):
        identities = {
            source_row["source_id"],
            cpu_row["source_id"],
            gpu_row["source_id"],
            cpu_price["source_id"],
        }
        if len(identities) != 1:
            raise ValueError("Generator source-row identity changed")
        for record in (cpu_row, gpu_row):
            if any(
                float(record[field]) != float(source_row[field])
                for field in ("pmin_mw", "pmax_mw")
            ):
                audit["source_limit_mismatches"] += 1
        commitment = round(float(gpu_row["commitment"]))
        dispatch = float(gpu_row["dispatch_mw"])
        gpu_commitment_count += commitment
        if not (
            float(gpu_row["pmin_mw"]) * commitment - 1e-6
            <= dispatch
            <= float(gpu_row["pmax_mw"]) * commitment + 1e-6
        ):
            audit["conditional_limit_violations_above_1e-6_mw"] += 1
        if int(gpu_row["source_status"]) <= 0 and (
            commitment != 0 or abs(dispatch) > 1e-6
        ):
            audit["source_offline_availability_violations"] += 1
        if abs(float(gpu_row["dispatch_pu"]) - dispatch / base_mva) > 1e-10:
            audit["dispatch_pu_conversion_mismatches"] += 1
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
                "cpu_accepted_commitment": round(float(cpu_row["commitment"])),
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
                "gpu_round_1_provisional_commitment": commitment,
                "gpu_round_1_provisional_dispatch_mw": gpu_row["dispatch_mw"],
                "gpu_round_1_provisional_dispatch_pu": gpu_row["dispatch_pu"],
                "gpu_final_solution_status": "unavailable_round_2_no_incumbent",
                "gpu_price_status": "not_reached",
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
                "bus": row["bus"],
                "cpu_accepted_price_per_mwh": row["price_per_mwh"],
                "cpu_accepted_price_per_pu_hour": row["price_per_pu_hour"],
                "gpu_price_status": "not_reached",
                "gpu_price_per_mwh": None,
                "gpu_price_per_pu_hour": None,
            }
            for row in cpu_prices
        ],
    )

    first, second = gpu["constraint_generation_rounds"]
    summary_rows = [
        {
            "platform": "laptop_cpu",
            "result_status": cpu["status"],
            "objective": cpu["objective"],
            "bound": cpu["bound"],
            "mip_gap": cpu["mip_gap"],
            "commitment_count": cpu["commitment_count"],
            "rounds": cpu["constraint_generation_round_count"],
            "added_pairs": cpu["added_security_pairs"],
            "last_completed_screen_new_violations": 0,
            "pricing_status": cpu["pricing"]["status"],
            "wall_seconds": cpu["total_wall_time_seconds"],
        },
        {
            "platform": "dgx_spark",
            "result_status": gpu["status"],
            "objective": None,
            "bound": None,
            "mip_gap": None,
            "commitment_count": None,
            "rounds": gpu["constraint_generation_round_count"],
            "added_pairs": gpu["added_security_pairs"],
            "last_completed_screen_new_violations": gpu[
                "last_completed_screen"
            ]["new_violated_pairs"],
            "pricing_status": "not_reached",
            "wall_seconds": gpu["total_wall_time_seconds"],
        },
    ]
    _write_csv(REPORT_ROOT / "summary.csv", summary_rows)

    round_rows: list[dict[str, Any]] = []
    for record in gpu["constraint_generation_rounds"]:
        solve = record["solve"]
        statistics = solve["statistics"]
        screen = record.get("screen", {})
        round_rows.append(
            {
                "round": record["round"],
                "rows_before_solve": record["rows_before_solve"],
                "status": solve["status"],
                "native_status": statistics.get("native_status"),
                "has_incumbent": solve["has_incumbent"],
                "requested_gap_certified": solve["requested_gap_certified"],
                "objective": solve["objective"],
                "bound": solve["bound"],
                "mip_gap": solve["mip_gap"] if solve["has_incumbent"] else None,
                "partial_integer_mip_start_columns": statistics.get(
                    "partial_integer_mip_start_columns"
                ),
                "adapter_wall_seconds": record["adapter_wall_time_seconds"],
                "native_solve_seconds": solve["solve_time_seconds"],
                "screen_completed": bool(screen),
                "screen_wall_seconds": screen.get("wall_time_seconds"),
                "screen_new_violated_pairs": screen.get("new_violated_pairs"),
                "screen_maximum_violation_pu": screen.get(
                    "maximum_violation_pu"
                ),
            }
        )
    _write_csv(REPORT_ROOT / "round-detail.csv", round_rows)

    evidence = {
        "schema_version": "1.0.0",
        "campaign": "activsg2000-gpu-gap-sensitivity-v3",
        "campaign_status": "incomplete_round_2_no_incumbent",
        "retry_performed_under_v3_identity": False,
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
            "gpu_round_1_posthoc_verification": {
                "sha256": verification["_sha256"],
                "bytes": verification["_bytes"],
            },
        },
        "summary": summary_rows,
        "gpu_rounds": round_rows,
        "gpu_round_1_provisional_commitment_count": gpu_commitment_count,
        "gpu_peak_memory": gpu["peak_memory"],
        "gpu_timings_seconds": gpu["timings_seconds"],
        "gpu_round_1_posthoc_verification": {
            key: value
            for key, value in verification.items()
            if not key.startswith("_")
        },
        "record_audit": audit,
        "feasibility_interpretation": {
            "cuopt_round_2_native_status": second["solve"]["status"],
            "mathematical_case_infeasibility_established": False,
            "reason": (
                "The accepted laptop solution uses the same immutable inputs and "
                "passes the complete contingency set, so it is a feasible witness "
                "for every 173-row subset added by GPU round 1."
            ),
        },
    }
    (REPORT_ROOT / "comparison.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readme = f"""# ACTIVSg2000 DGX Spark 1e-3 v3 result

The one authorized v3 run ended `incomplete_no_incumbent` in
{gpu['total_wall_time_seconds']:.3f} seconds. It is not a successful SCOPF result.

Round 1 returned cuOpt `FeasibleFound` with objective
{first['solve']['objective']:,.6f}, bound {first['solve']['bound']:,.6f}, and
gap {first['solve']['mip_gap']:.6e}. The finite-bound certificate passed. The
mandatory exhaustive screen then checked
{first['screen']['evaluated_sides']:,} sides, found
{first['screen']['new_violated_pairs']} violations, and added all 173 rows.
The provisional round-1 solution committed {gpu_commitment_count} generators.

Round 2 received the prior values for all
{second['solve']['statistics']['partial_integer_mip_start_columns']} integer
commitment columns. After {second['adapter_wall_time_seconds']:.3f} seconds,
cuOpt returned native `Infeasible` with no incumbent or finite bound. Therefore
there was no round-2 screen, independent verification, final commitment or
dispatch, or fixed-commitment pricing. The native `mip_gap=0` attached to that
no-incumbent status is not an accepted gap and is omitted from the summary.

The native status does **not** establish that the mathematical SCOPF is
infeasible. The accepted laptop result uses the same immutable inputs and model
contract and passes the complete N-1 set; it is therefore a feasible witness
for any subset of the 173 rows added after GPU round 1. This result instead
isolates a cuOpt/adapter numerical or solver-behavior issue in the rebuilt
secured master.

## Provisional round-1 audit

The raw-input checker was run without another MIP solve. It passed the basic
model with maximum residual
{verification['maximum_model_residual_pu']:.3e} p.u. and exact conditional
PMIN/PMAX residual
{verification['details']['conditional_pmin_pmax_violation_pu']:.3e} p.u. It
failed N-1 security, as expected, with maximum violation
{verification['maximum_security_violation_pu']:.6f} p.u. across
{verification['checked_security_sides']:,} sides.

`generator-detail.csv` preserves all 544 exact source PMIN/PMAX rows and labels
the GPU round-1 commitment/dispatch as provisional. GPU pricing columns are
blank. `bus-prices.csv` contains the accepted laptop prices solely as a
reference; every GPU price field is blank. `round-detail.csv`, `summary.csv`,
and `comparison.json` preserve timing, status, hashes, and acceptance gates.

## Timing and peak memory

- Model and factor build: {gpu['timings_seconds']['model_and_factor_build']:.3f} s
- Round 1 cuOpt solve: {first['adapter_wall_time_seconds']:.3f} s
- Round 1 CuPy exhaustive screen: {first['screen']['wall_time_seconds']:.3f} s
- Round 2 cuOpt solve: {second['adapter_wall_time_seconds']:.3f} s
- End to end: {gpu['total_wall_time_seconds']:.3f} s
- Peak process RSS: {gpu['peak_memory']['process_rss_bytes']:,} bytes
- Peak CUDA device-memory delta: {gpu['peak_memory']['cuda_device_memory_delta_bytes']:,} bytes
- Peak CuPy pool use: {gpu['peak_memory']['cupy_pool_used_bytes']:,} bytes

No v3 retry was performed.
"""
    (REPORT_ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
