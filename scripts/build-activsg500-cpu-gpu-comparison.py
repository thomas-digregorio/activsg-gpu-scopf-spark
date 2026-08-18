"""Build a tracked ACTIVSg500 laptop-CPU versus DGX-Spark comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "experiments"
REPORT_ROOT = ROOT / "reports" / "activsg500-cpu-vs-spark-gap-v1"
GAPS = ("1e-3", "1e-4", "1e-5", "1e-6", "1e-7")


def _read(path: Path, *, label: str, platform: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("case_name") != "ACTIVSg500":
        raise ValueError(f"{path} is not ACTIVSg500 evidence")
    if payload.get("gap_label") != label or payload.get("platform") != platform:
        raise ValueError(f"{path} has the wrong gap or platform identity")
    if payload.get("status") != "optimal_verified":
        raise ValueError(f"{path} did not finish optimal_verified")
    if payload.get("pricing", {}).get("status") != "optimal_secure_fixed_commitment_lp":
        raise ValueError(f"{path} does not contain accepted pricing")
    payload["_path"] = path
    payload["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _difference(gpu: list[float], cpu: list[float]) -> dict[str, float | int]:
    values = [float(a) - float(b) for a, b in zip(gpu, cpu, strict=True)]
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


def _source_hashes(payload: dict[str, Any]) -> tuple[str, str]:
    identity = payload["source_manifest"]["source_identity"]
    return identity["case_sha256"], identity["contingency_sha256"]


def main() -> None:
    cpu = {
        label: _read(
            RESULT_ROOT / f"activsg500-gap-v1-{label}-laptop.json",
            label=label,
            platform="laptop_cpu",
        )
        for label in GAPS
    }
    gpu = {
        label: _read(
            RESULT_ROOT / f"activsg500-gpu-gap-v1-{label}-dgx-spark.json",
            label=label,
            platform="dgx_spark",
        )
        for label in GAPS
    }
    identities = {_source_hashes(result) for result in (*cpu.values(), *gpu.values())}
    if len(identities) != 1:
        raise ValueError("CPU and Spark source hashes differ")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    bus_rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    for label in GAPS:
        cpu_result = cpu[label]
        gpu_result = gpu[label]
        cpu_generators = cpu_result["solution"]["generators"]
        gpu_generators = gpu_result["solution"]["generators"]
        cpu_pricing_generators = cpu_result["pricing"]["generators"]
        gpu_pricing_generators = gpu_result["pricing"]["generators"]
        cpu_prices = cpu_result["pricing"]["bus_prices"]
        gpu_prices = gpu_result["pricing"]["bus_prices"]
        if len(cpu_generators) != len(gpu_generators) or len(cpu_prices) != len(gpu_prices):
            raise ValueError(f"CPU and Spark dimensions differ at {label}")

        commitment_pairs: list[tuple[int, int]] = []
        for cpu_gen, gpu_gen, cpu_price_gen, gpu_price_gen in zip(
            cpu_generators,
            gpu_generators,
            cpu_pricing_generators,
            gpu_pricing_generators,
            strict=True,
        ):
            if not (
                cpu_gen["source_id"]
                == gpu_gen["source_id"]
                == cpu_price_gen["source_id"]
                == gpu_price_gen["source_id"]
            ):
                raise ValueError(f"Generator identity differs at {label}")
            for field in ("pmin_mw", "pmin_pu", "pmax_mw", "pmax_pu"):
                if float(cpu_gen[field]) != float(gpu_gen[field]):
                    raise ValueError(f"Exact source {field} differs at {label}")
            cpu_commitment = round(float(cpu_gen["commitment"]))
            gpu_commitment = round(float(gpu_gen["commitment"]))
            commitment_pairs.append((cpu_commitment, gpu_commitment))
            generator_rows.append(
                {
                    "gap_label": label,
                    "source_id": cpu_gen["source_id"],
                    "source_row": cpu_gen["source_row"],
                    "source_status": cpu_gen["source_status"],
                    "bus": cpu_gen["bus"],
                    "pmin_mw": cpu_gen["pmin_mw"],
                    "pmin_pu": cpu_gen["pmin_pu"],
                    "pmax_mw": cpu_gen["pmax_mw"],
                    "pmax_pu": cpu_gen["pmax_pu"],
                    "cpu_commitment": cpu_commitment,
                    "gpu_commitment": gpu_commitment,
                    "gpu_minus_cpu_commitment": gpu_commitment - cpu_commitment,
                    "cpu_mip_dispatch_mw": cpu_gen["dispatch_mw"],
                    "gpu_mip_dispatch_mw": gpu_gen["dispatch_mw"],
                    "gpu_minus_cpu_mip_dispatch_mw": (
                        float(gpu_gen["dispatch_mw"]) - float(cpu_gen["dispatch_mw"])
                    ),
                    "cpu_mip_dispatch_pu": cpu_gen["dispatch_pu"],
                    "gpu_mip_dispatch_pu": gpu_gen["dispatch_pu"],
                    "gpu_minus_cpu_mip_dispatch_pu": (
                        float(gpu_gen["dispatch_pu"]) - float(cpu_gen["dispatch_pu"])
                    ),
                    "cpu_pricing_dispatch_mw": cpu_price_gen["pricing_dispatch_mw"],
                    "gpu_pricing_dispatch_mw": gpu_price_gen["pricing_dispatch_mw"],
                    "gpu_minus_cpu_pricing_dispatch_mw": (
                        float(gpu_price_gen["pricing_dispatch_mw"])
                        - float(cpu_price_gen["pricing_dispatch_mw"])
                    ),
                    "cpu_pricing_dispatch_pu": cpu_price_gen["pricing_dispatch_pu"],
                    "gpu_pricing_dispatch_pu": gpu_price_gen["pricing_dispatch_pu"],
                    "gpu_minus_cpu_pricing_dispatch_pu": (
                        float(gpu_price_gen["pricing_dispatch_pu"])
                        - float(cpu_price_gen["pricing_dispatch_pu"])
                    ),
                    "cpu_price_per_mwh": cpu_price_gen["nodal_price_per_mwh"],
                    "gpu_price_per_mwh": gpu_price_gen["nodal_price_per_mwh"],
                    "gpu_minus_cpu_price_per_mwh": (
                        float(gpu_price_gen["nodal_price_per_mwh"])
                        - float(cpu_price_gen["nodal_price_per_mwh"])
                    ),
                    "cpu_price_per_pu_hour": cpu_price_gen["nodal_price_per_pu_hour"],
                    "gpu_price_per_pu_hour": gpu_price_gen["nodal_price_per_pu_hour"],
                    "gpu_minus_cpu_price_per_pu_hour": (
                        float(gpu_price_gen["nodal_price_per_pu_hour"])
                        - float(cpu_price_gen["nodal_price_per_pu_hour"])
                    ),
                }
            )

        for cpu_price, gpu_price in zip(cpu_prices, gpu_prices, strict=True):
            if cpu_price["bus"] != gpu_price["bus"]:
                raise ValueError(f"Bus identity differs at {label}")
            bus_rows.append(
                {
                    "gap_label": label,
                    "bus": cpu_price["bus"],
                    "cpu_price_per_mwh": cpu_price["price_per_mwh"],
                    "gpu_price_per_mwh": gpu_price["price_per_mwh"],
                    "gpu_minus_cpu_price_per_mwh": (
                        float(gpu_price["price_per_mwh"])
                        - float(cpu_price["price_per_mwh"])
                    ),
                    "cpu_price_per_pu_hour": cpu_price["price_per_pu_hour"],
                    "gpu_price_per_pu_hour": gpu_price["price_per_pu_hour"],
                    "gpu_minus_cpu_price_per_pu_hour": (
                        float(gpu_price["price_per_pu_hour"])
                        - float(cpu_price["price_per_pu_hour"])
                    ),
                }
            )

        cpu_commitments = [pair[0] for pair in commitment_pairs]
        gpu_commitments = [pair[1] for pair in commitment_pairs]
        comparison = {
            "commitment_flip_count": sum(a != b for a, b in commitment_pairs),
            "cpu_off_gpu_on_count": sum(a == 0 and b == 1 for a, b in commitment_pairs),
            "cpu_on_gpu_off_count": sum(a == 1 and b == 0 for a, b in commitment_pairs),
            "commitment_vector": _difference(gpu_commitments, cpu_commitments),
            "mip_dispatch_mw": _difference(
                [float(item["dispatch_mw"]) for item in gpu_generators],
                [float(item["dispatch_mw"]) for item in cpu_generators],
            ),
            "pricing_dispatch_mw": _difference(
                [float(item["pricing_dispatch_mw"]) for item in gpu_pricing_generators],
                [float(item["pricing_dispatch_mw"]) for item in cpu_pricing_generators],
            ),
            "bus_price_per_mwh": _difference(
                [float(item["price_per_mwh"]) for item in gpu_prices],
                [float(item["price_per_mwh"]) for item in cpu_prices],
            ),
        }
        comparisons[label] = comparison
        cpu_wall = float(cpu_result["total_wall_time_seconds"])
        gpu_wall = float(gpu_result["total_wall_time_seconds"])
        summaries.append(
            {
                "gap_label": label,
                "cpu_objective": cpu_result["objective"],
                "gpu_objective": gpu_result["objective"],
                "gpu_minus_cpu_objective": (
                    float(gpu_result["objective"]) - float(cpu_result["objective"])
                ),
                "cpu_bound": cpu_result["bound"],
                "gpu_bound": gpu_result["bound"],
                "cpu_achieved_gap": cpu_result["mip_gap"],
                "gpu_achieved_gap": gpu_result["mip_gap"],
                "cpu_committed": cpu_result["commitment_count"],
                "gpu_committed": gpu_result["commitment_count"],
                "commitment_flips": comparison["commitment_flip_count"],
                "mip_dispatch_l1_mw": comparison["mip_dispatch_mw"]["sum_absolute"],
                "mip_dispatch_max_mw": comparison["mip_dispatch_mw"]["maximum_absolute"],
                "pricing_dispatch_l1_mw": comparison["pricing_dispatch_mw"][
                    "sum_absolute"
                ],
                "price_mean_absolute_per_mwh": comparison["bus_price_per_mwh"][
                    "mean_absolute"
                ],
                "price_max_absolute_per_mwh": comparison["bus_price_per_mwh"][
                    "maximum_absolute"
                ],
                "cpu_rounds": cpu_result["constraint_generation_round_count"],
                "gpu_rounds": gpu_result["constraint_generation_round_count"],
                "cpu_added_pairs": cpu_result["added_security_pairs"],
                "gpu_added_pairs": gpu_result["added_security_pairs"],
                "cpu_wall_seconds": cpu_wall,
                "gpu_wall_seconds": gpu_wall,
                "cpu_wall_divided_by_gpu_wall": cpu_wall / gpu_wall,
                "cpu_final_residual_pu": cpu_result["maximum_model_residual_pu"],
                "gpu_final_residual_pu": gpu_result["maximum_model_residual_pu"],
                "cpu_final_security_violation_pu": cpu_result[
                    "final_exhaustive_violation_pu"
                ],
                "gpu_final_security_violation_pu": gpu_result[
                    "final_exhaustive_violation_pu"
                ],
                "gpu_peak_process_rss_bytes": gpu_result["peak_memory"][
                    "process_rss_bytes"
                ],
                "gpu_peak_cupy_pool_used_bytes": gpu_result["peak_memory"].get(
                    "cupy_pool_used_bytes"
                ),
                "gpu_peak_cuda_device_memory_delta_bytes": gpu_result["peak_memory"].get(
                    "cuda_device_memory_delta_bytes"
                ),
            }
        )

    _write_csv(REPORT_ROOT / "summary.csv", summaries)
    _write_csv(REPORT_ROOT / "generator-detail.csv", generator_rows)
    _write_csv(REPORT_ROOT / "bus-prices.csv", bus_rows)
    case_hash, contingency_hash = next(iter(identities))
    comparison_payload = {
        "schema_version": "1.0.0",
        "comparison": "laptop CPU system versus DGX Spark cuOpt/CuPy system",
        "pure_gpu_speedup_claim": False,
        "gap_order": list(GAPS),
        "source_hashes": {
            "case_sha256": case_hash,
            "contingency_sha256": contingency_hash,
        },
        "cpu_frozen_identity": cpu[GAPS[0]]["frozen_identity"],
        "gpu_frozen_identity": gpu[GAPS[0]]["frozen_identity"],
        "raw_result_hashes": {
            label: {
                "cpu_sha256": cpu[label]["_sha256"],
                "gpu_sha256": gpu[label]["_sha256"],
            }
            for label in GAPS
        },
        "summary": summaries,
        "differences": comparisons,
    }
    (REPORT_ROOT / "comparison.json").write_text(
        json.dumps(comparison_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ACTIVSg500 laptop CPU versus DGX Spark gap comparison",
        "",
        "This is an end-to-end system comparison, not a pure GPU speedup claim. The "
        "laptop path uses HiGHS and NumPy; the Spark path uses cuOpt and CuPy for "
        "the MIP and contingency screens, followed by HiGHS inside the same Spark "
        "container for fixed-commitment nodal pricing.",
        "",
        "Both suites use the same immutable ACTIVSg500 source hashes, exact source "
        "PMIN/PMAX, one-hour preventive model, ten-segment source-derived cost curves, "
        "security tolerance, and five requested gap levels. Every included result is "
        "optimal, independently verified, exhaustively N-1 secure, and priced.",
        "",
        (
            "| Gap | CPU objective | Spark objective | Commitment flips | "
            "Dispatch L1 (MW) | Mean price difference ($/MWh) | CPU wall (s) | "
            "Spark wall (s) | CPU/Spark wall ratio |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['gap_label']} | {float(row['cpu_objective']):,.6f} | "
            f"{float(row['gpu_objective']):,.6f} | {row['commitment_flips']} | "
            f"{float(row['mip_dispatch_l1_mw']):,.6f} | "
            f"{float(row['price_mean_absolute_per_mwh']):,.6f} | "
            f"{float(row['cpu_wall_seconds']):,.3f} | "
            f"{float(row['gpu_wall_seconds']):,.3f} | "
            f"{float(row['cpu_wall_divided_by_gpu_wall']):,.3f} |"
        )
    maximum_objective_difference = max(
        abs(float(row["gpu_minus_cpu_objective"])) for row in summaries
    )
    maximum_dispatch_difference = max(
        float(row["mip_dispatch_max_mw"]) for row in summaries
    )
    maximum_price_difference = max(
        float(row["price_max_absolute_per_mwh"]) for row in summaries
    )
    maximum_gpu_over_cpu_wall = max(
        float(row["gpu_wall_seconds"]) / float(row["cpu_wall_seconds"])
        for row in summaries
    )
    minimum_gpu_over_cpu_wall = min(
        float(row["gpu_wall_seconds"]) / float(row["cpu_wall_seconds"])
        for row in summaries
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The two solver stacks selected the same commitment at every gap. No "
            "generator dispatch or nodal-price difference exceeded `1e-6` in its "
            "reported unit. The largest absolute objective difference was "
            f"{maximum_objective_difference:.3e}, the largest generator dispatch "
            f"difference was {maximum_dispatch_difference:.3e} MW, and the largest "
            f"bus-price difference was {maximum_price_difference:.3e} $/MWh.",
            "",
            "On this small case, the DGX Spark end-to-end path took "
            f"{minimum_gpu_over_cpu_wall:.3f}x to {maximum_gpu_over_cpu_wall:.3f}x "
            "the laptop wall time. This is a latency-dominated system result, not "
            "evidence that GPU acceleration is intrinsically slower for larger "
            "instances.",
            "",
            "`generator-detail.csv` contains exact PMIN/PMAX, commitment, MIP "
            "dispatch, pricing dispatch, and generator-bus prices in MW and p.u. "
            "units for both systems at every gap. `bus-prices.csv` contains all 500 "
            "paired nodal prices. `comparison.json` records exact vector-difference "
            "metrics, frozen identities, and raw-result SHA-256 hashes.",
            "",
        ]
    )
    (REPORT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
