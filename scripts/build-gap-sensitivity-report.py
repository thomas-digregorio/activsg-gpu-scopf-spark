"""Build compact tracked comparisons from the five ignored one-shot results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GAPS = ("1e-3", "1e-4", "1e-5", "1e-6", "1e-7")
RESULT_DIR = ROOT / "results" / "experiments"
REPORT_DIR = ROOT / "reports" / "activsg10k-gap-sensitivity-v1"


def _read(label: str) -> dict[str, Any]:
    path = RESULT_DIR / f"activsg10k-gap-{label}-laptop.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gap_label") != label:
        raise ValueError(f"{path} has the wrong gap label")
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


def _vector_difference(left: list[float], right: list[float]) -> dict[str, float | int]:
    differences = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    absolute = [abs(value) for value in differences]
    return {
        "count_above_1e-6": sum(value > 1e-6 for value in absolute),
        "count_above_1e-2": sum(value > 1e-2 for value in absolute),
        "count_above_1": sum(value > 1.0 for value in absolute),
        "sum_absolute": sum(absolute),
        "mean_absolute": sum(absolute) / len(absolute),
        "root_mean_square": math.sqrt(
            sum(value * value for value in differences) / len(differences)
        ),
        "maximum_absolute": max(absolute, default=0.0),
    }


def _comparison(
    left_label: str,
    right_label: str,
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    left = results[left_label]
    right = results[right_label]
    left_mip = left["solution"]["generators"]
    right_mip = right["solution"]["generators"]
    left_pricing = left["pricing"]["generators"]
    right_pricing = right["pricing"]["generators"]
    left_commitment = [round(float(record["commitment"])) for record in left_mip]
    right_commitment = [round(float(record["commitment"])) for record in right_mip]
    paired_commitments = zip(left_commitment, right_commitment, strict=True)
    commitment_pairs = list(paired_commitments)
    off_to_on = sum(a == 0 and b == 1 for a, b in commitment_pairs)
    on_to_off = sum(a == 1 and b == 0 for a, b in commitment_pairs)
    left_prices = [float(record["price_per_mwh"]) for record in left["pricing"]["bus_prices"]]
    right_prices = [
        float(record["price_per_mwh"]) for record in right["pricing"]["bus_prices"]
    ]
    objective_difference = float(left["objective"] - right["objective"])
    return {
        "from_gap": left_label,
        "to_gap": right_label,
        "objective_difference_from_minus_to": objective_difference,
        "objective_relative_difference_from_minus_to": (
            objective_difference / float(right["objective"])
        ),
        "bound_difference_from_minus_to": float(left["bound"] - right["bound"]),
        "commitment_flip_count": off_to_on + on_to_off,
        "off_to_on_count": off_to_on,
        "on_to_off_count": on_to_off,
        "mip_dispatch_mw": _vector_difference(
            [float(record["dispatch_mw"]) for record in left_mip],
            [float(record["dispatch_mw"]) for record in right_mip],
        ),
        "pricing_dispatch_mw": _vector_difference(
            [float(record["pricing_dispatch_mw"]) for record in left_pricing],
            [float(record["pricing_dispatch_mw"]) for record in right_pricing],
        ),
        "bus_price_per_mwh": _vector_difference(left_prices, right_prices),
    }


def main() -> None:
    results = {label: _read(label) for label in GAPS}
    commits = {result["frozen_identity"]["commit"] for result in results.values()}
    tags = {result["frozen_identity"]["tag"] for result in results.values()}
    source_hashes = {
        (
            result["source_manifest"]["source_identity"]["case_sha256"],
            result["source_manifest"]["source_identity"]["contingency_sha256"],
        )
        for result in results.values()
    }
    if len(commits) != 1 or len(tags) != 1 or len(source_hashes) != 1:
        raise ValueError("Experiment identity differs across gap levels")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for label in GAPS:
        result = results[label]
        pricing = result["pricing"]
        summary_rows.append(
            {
                "gap_label": label,
                "requested_mip_gap": float(label),
                "achieved_mip_gap": result["mip_gap"],
                "objective": result["objective"],
                "bound": result["bound"],
                "commitment_count": result["commitment_count"],
                "constraint_generation_rounds": result[
                    "constraint_generation_round_count"
                ],
                "mip_added_security_pairs": result["added_security_pairs"],
                "final_model_residual_pu": result["maximum_model_residual_pu"],
                "final_security_violation_pu": result[
                    "final_exhaustive_violation_pu"
                ],
                "mip_solver_seconds": result["timings_seconds"]["solver_rounds"],
                "mip_screen_seconds": result["timings_seconds"]["screening"],
                "pricing_seconds": result["timings_seconds"][
                    "fixed_commitment_pricing"
                ],
                "total_wall_seconds": result["total_wall_time_seconds"],
                "pricing_lp_objective": pricing["pricing_lp_objective"],
                "mip_minus_pricing_lp_objective": pricing[
                    "mip_minus_pricing_lp_objective"
                ],
                "pricing_rounds": len(pricing["pricing_constraint_generation_rounds"]),
                "pricing_added_security_pairs": len(
                    pricing["pricing_added_security_pair_ids"]
                ),
                "pricing_final_security_violation_pu": pricing[
                    "final_exhaustive_violation_pu"
                ],
                "price_min_per_mwh": pricing["bus_price_summary_per_mwh"]["minimum"],
                "price_mean_per_mwh": pricing["bus_price_summary_per_mwh"]["mean"],
                "price_median_per_mwh": pricing["bus_price_summary_per_mwh"]["median"],
                "price_max_per_mwh": pricing["bus_price_summary_per_mwh"]["maximum"],
                "peak_process_rss_bytes": result["peak_memory"]["process_rss_bytes"],
                "raw_result_sha256": result["_sha256"],
            }
        )
    _write_csv(REPORT_DIR / "summary.csv", summary_rows)

    first_generators = results[GAPS[0]]["solution"]["generators"]
    generator_rows: list[dict[str, Any]] = []
    for index, source in enumerate(first_generators):
        row: dict[str, Any] = {
            "source_id": source["source_id"],
            "source_row": source["source_row"],
            "source_status": source["source_status"],
            "bus": source["bus"],
            "pmin_mw": source["pmin_mw"],
            "pmin_pu": source["pmin_pu"],
            "pmax_mw": source["pmax_mw"],
            "pmax_pu": source["pmax_pu"],
        }
        for label in GAPS:
            mip = results[label]["solution"]["generators"][index]
            pricing = results[label]["pricing"]["generators"][index]
            if mip["source_id"] != source["source_id"]:
                raise ValueError("Generator source-row identity changed between results")
            prefix = f"gap_{label}_"
            row.update(
                {
                    f"{prefix}commitment": round(float(mip["commitment"])),
                    f"{prefix}mip_dispatch_mw": mip["dispatch_mw"],
                    f"{prefix}mip_dispatch_pu": mip["dispatch_pu"],
                    f"{prefix}pricing_dispatch_mw": pricing["pricing_dispatch_mw"],
                    f"{prefix}pricing_dispatch_pu": pricing["pricing_dispatch_pu"],
                    f"{prefix}price_per_mwh": pricing["nodal_price_per_mwh"],
                    f"{prefix}price_per_pu_hour": pricing[
                        "nodal_price_per_pu_hour"
                    ],
                }
            )
        generator_rows.append(row)
    _write_csv(REPORT_DIR / "generator-detail.csv", generator_rows)

    first_buses = results[GAPS[0]]["pricing"]["bus_prices"]
    bus_rows: list[dict[str, Any]] = []
    for index, source in enumerate(first_buses):
        row = {"bus": source["bus"]}
        for label in GAPS:
            price = results[label]["pricing"]["bus_prices"][index]
            if price["bus"] != source["bus"]:
                raise ValueError("Bus identity changed between price results")
            row[f"gap_{label}_price_per_mwh"] = price["price_per_mwh"]
            row[f"gap_{label}_price_per_pu_hour"] = price["price_per_pu_hour"]
        bus_rows.append(row)
    _write_csv(REPORT_DIR / "bus-prices.csv", bus_rows)

    adjacent = [
        _comparison(left, right, results)
        for left, right in zip(GAPS[:-1], GAPS[1:], strict=True)
    ]
    versus_strictest = [
        _comparison(label, GAPS[-1], results) for label in GAPS[:-1]
    ]
    comparison = {
        "schema_version": "1.0.0",
        "experiment_suite_id": "activsg10k-gap-sensitivity-v1",
        "commit": next(iter(commits)),
        "tag": next(iter(tags)),
        "source_hashes": {
            "case_sha256": next(iter(source_hashes))[0],
            "contingency_sha256": next(iter(source_hashes))[1],
        },
        "base_mva": results[GAPS[0]]["pricing"]["base_mva"],
        "gap_order": list(GAPS),
        "summary": summary_rows,
        "adjacent_comparisons": adjacent,
        "comparisons_to_1e-7": versus_strictest,
    }
    (REPORT_DIR / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ACTIVSg10k MIP-gap sensitivity",
        "",
        "Each gap level is one independent, unbounded laptop MIP run from the base "
        "master. Within each run, dynamic N-1 constraint generation retains the "
        "previous commitment as a partial MIP start. Gap levels are not seeded from "
        "one another. Prices are from a separately identified fixed-commitment, "
        "N-1-secure LP and are not MILP duals.",
        "",
        (
            "| Requested gap | Achieved gap | Objective | Bound | Committed | "
            "MIP rounds | Wall (s) | Mean price ($/MWh) |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['gap_label']} | {float(row['achieved_mip_gap']):.3e} | "
            f"{float(row['objective']):,.6f} | {float(row['bound']):,.6f} | "
            f"{row['commitment_count']} | {row['constraint_generation_rounds']} | "
            f"{float(row['total_wall_seconds']):,.3f} | "
            f"{float(row['price_mean_per_mwh']):,.6f} |"
        )
    lines.extend(
        [
            "",
            "## Differences from the 1e-7 result",
            "",
            (
                "| Gap | Commitment flips | MIP dispatch L1 (MW) | Maximum dispatch "
                "change (MW) | Mean absolute bus-price change ($/MWh) | Maximum "
                "bus-price change ($/MWh) |"
            ),
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in versus_strictest:
        lines.append(
            f"| {item['from_gap']} | {item['commitment_flip_count']} | "
            f"{item['mip_dispatch_mw']['sum_absolute']:,.6f} | "
            f"{item['mip_dispatch_mw']['maximum_absolute']:,.6f} | "
            f"{item['bus_price_per_mwh']['mean_absolute']:,.6f} | "
            f"{item['bus_price_per_mwh']['maximum_absolute']:,.6f} |"
        )
    lines.extend(
        [
            "",
            "Full source-row generator data are in `generator-detail.csv`; all 10,000 "
            "bus prices are in `bus-prices.csv`; exact metrics and evidence hashes are "
            "in `comparison.json`.",
            "",
        ]
    )
    (REPORT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
