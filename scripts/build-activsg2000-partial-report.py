"""Build tracked evidence for the stopped ACTIVSg2000 bounded gap campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "experiments"
REPORT_ROOT = ROOT / "reports" / "activsg2000-gap-sensitivity-v1"
REGISTRY_PATH = RESULT_ROOT / "activsg2000-gap-sensitivity-v1-run-registry.json"
GAPS = ("1e-3", "1e-4", "1e-5", "1e-6", "1e-7")
EXPECTED_COMMIT = "173fd0c10b8af5ed431956f2eea9726032df6f9e"
EXPECTED_TAG = "experiment-2000-gap-v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result(label: str) -> dict[str, Any]:
    path = RESULT_ROOT / f"activsg2000-gap-v1-{label}-laptop.json"
    payload = _load_json(path)
    payload["_path"] = path
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


def _difference(left: list[float], right: list[float]) -> dict[str, float | int]:
    values = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    absolute = [abs(value) for value in values]
    return {
        "count_above_1e-6": sum(value > 1e-6 for value in absolute),
        "count_above_1e-2": sum(value > 1e-2 for value in absolute),
        "count_above_1_mw": sum(value > 1.0 for value in absolute),
        "sum_absolute": sum(absolute),
        "mean_absolute": sum(absolute) / len(absolute),
        "root_mean_square": math.sqrt(
            sum(value * value for value in values) / len(values)
        ),
        "maximum_absolute": max(absolute, default=0.0),
    }


def _identity_checks(
    accepted: dict[str, Any], incomplete: dict[str, Any], registry: dict[str, Any]
) -> None:
    if accepted.get("status") != "optimal_verified":
        raise ValueError("The 1e-3 result is not optimal_verified")
    if accepted.get("pricing", {}).get("status") != (
        "optimal_secure_fixed_commitment_lp"
    ):
        raise ValueError("The 1e-3 result does not have accepted pricing")
    if incomplete.get("status") != "incomplete_restricted_master_not_optimal":
        raise ValueError("The 1e-4 result is not the registered incomplete result")
    if incomplete.get("solver_status") != "TimeLimit":
        raise ValueError("The 1e-4 result did not preserve the expected TimeLimit")
    identities = (accepted["frozen_identity"], incomplete["frozen_identity"])
    if {item["commit"] for item in identities} != {EXPECTED_COMMIT}:
        raise ValueError("Result commit differs from the frozen experiment commit")
    if {item["tag"] for item in identities} != {EXPECTED_TAG}:
        raise ValueError("Result tag differs from the frozen experiment tag")
    source_hashes = {
        (
            item["source_manifest"]["source_identity"]["case_sha256"],
            item["source_manifest"]["source_identity"]["contingency_sha256"],
        )
        for item in (accepted, incomplete)
    }
    if len(source_hashes) != 1:
        raise ValueError("Source hashes changed between runs")
    if set(registry.get("runs", {})) != {"1e-3", "1e-4"}:
        raise ValueError("The run registry does not show the required stop after 1e-4")
    for label in GAPS[2:]:
        if (RESULT_ROOT / f"activsg2000-gap-v1-{label}-laptop.json").exists():
            raise ValueError(f"Unexpected later-gap result exists for {label}")


def _audit_records(
    source: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    incomplete: list[dict[str, Any]],
    prices: list[dict[str, Any]],
    *,
    base_mva: float,
) -> dict[str, int]:
    counters = {
        "accepted_source_limit_mismatches": 0,
        "incomplete_source_limit_mismatches": 0,
        "accepted_dispatch_pu_mismatches": 0,
        "incomplete_dispatch_pu_mismatches": 0,
        "accepted_conditional_limit_violations_above_1e-6_mw": 0,
        "incomplete_conditional_limit_violations_above_1e-6_mw": 0,
        "accepted_source_offline_availability_violations": 0,
        "incomplete_source_offline_availability_violations": 0,
        "price_pu_conversion_mismatches": 0,
    }
    for source_row, accepted_row, incomplete_row in zip(
        source, accepted, incomplete, strict=True
    ):
        if not (
            source_row["source_id"]
            == accepted_row["source_id"]
            == incomplete_row["source_id"]
        ):
            raise ValueError("Generator source-row identity changed")
        for label, record in (
            ("accepted", accepted_row),
            ("incomplete", incomplete_row),
        ):
            if (
                abs(float(record["pmin_mw"]) - float(source_row["pmin_mw"]))
                > 1e-12
                or abs(float(record["pmax_mw"]) - float(source_row["pmax_mw"]))
                > 1e-12
            ):
                counters[f"{label}_source_limit_mismatches"] += 1
            if (
                abs(
                    float(record["dispatch_pu"])
                    - float(record["dispatch_mw"]) / base_mva
                )
                > 1e-12
            ):
                counters[f"{label}_dispatch_pu_mismatches"] += 1
            commitment = round(float(record["commitment"]))
            dispatch = float(record["dispatch_mw"])
            lower = float(record["pmin_mw"]) * commitment
            upper = float(record["pmax_mw"]) * commitment
            if dispatch < lower - 1e-6 or dispatch > upper + 1e-6:
                counters[
                    f"{label}_conditional_limit_violations_above_1e-6_mw"
                ] += 1
            if int(record["source_status"]) <= 0 and (
                commitment != 0 or abs(dispatch) > 1e-7
            ):
                counters[f"{label}_source_offline_availability_violations"] += 1
    for record in prices:
        if (
            abs(
                float(record["price_per_pu_hour"])
                - base_mva * float(record["price_per_mwh"])
            )
            > 1e-8
        ):
            counters["price_pu_conversion_mismatches"] += 1
    return counters


def main() -> None:
    accepted = _result("1e-3")
    incomplete = _result("1e-4")
    registry = _load_json(REGISTRY_PATH)
    _identity_checks(accepted, incomplete, registry)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    source = accepted["source_manifest"]["generators"]
    accepted_generators = accepted["solution"]["generators"]
    incomplete_generators = incomplete["solution"]["generators"]
    pricing_generators = accepted["pricing"]["generators"]
    bus_prices = accepted["pricing"]["bus_prices"]
    base_mva = float(accepted["pricing"]["base_mva"])
    if not (
        len(source)
        == len(accepted_generators)
        == len(incomplete_generators)
        == len(pricing_generators)
        == 544
    ):
        raise ValueError("Expected 544 source-row generator records")
    if len(bus_prices) != 2000:
        raise ValueError("Expected 2,000 accepted bus-price records")
    audits = _audit_records(
        source,
        accepted_generators,
        incomplete_generators,
        bus_prices,
        base_mva=base_mva,
    )
    if any(audits.values()):
        raise ValueError(f"Generator or unit-conversion audit failed: {audits}")

    accepted_commitment = [
        round(float(record["commitment"])) for record in accepted_generators
    ]
    incomplete_commitment = [
        round(float(record["commitment"])) for record in incomplete_generators
    ]
    commitment_pairs = list(
        zip(accepted_commitment, incomplete_commitment, strict=True)
    )
    provisional_comparison = {
        "comparison_status": "not_an_accepted_gap_comparison",
        "reason": (
            "The 1e-4 incumbent hit TimeLimit on a restricted master, then its "
            "provisional screen found 18 additional violated security pairs."
        ),
        "commitment_flip_count": sum(a != b for a, b in commitment_pairs),
        "off_to_on_count": sum(a == 0 and b == 1 for a, b in commitment_pairs),
        "on_to_off_count": sum(a == 1 and b == 0 for a, b in commitment_pairs),
        "accepted_1e-3_commitment_count": sum(accepted_commitment),
        "provisional_1e-4_commitment_count": sum(incomplete_commitment),
        "mip_dispatch_mw": _difference(
            [float(record["dispatch_mw"]) for record in accepted_generators],
            [float(record["dispatch_mw"]) for record in incomplete_generators],
        ),
        "pricing_comparison_available": False,
    }

    last_accepted_screen = accepted["constraint_generation_rounds"][-1]["screen"]
    last_incomplete_screen = incomplete["last_completed_screen"]
    summary_rows = [
        {
            "gap_label": "1e-3",
            "requested_mip_gap": 1e-3,
            "status": accepted["status"],
            "objective": accepted["objective"],
            "bound": accepted["bound"],
            "achieved_mip_gap": accepted["mip_gap"],
            "commitment_count": accepted["commitment_count"],
            "recorded_rounds": len(accepted["constraint_generation_rounds"]),
            "accepted_added_security_pairs": accepted["added_security_pairs"],
            "last_screen_new_violated_pairs": last_accepted_screen[
                "new_violated_pairs"
            ],
            "last_screen_maximum_violation_pu": last_accepted_screen[
                "maximum_violation_pu"
            ],
            "independent_verification_passed": accepted["verification"]["passed"],
            "pricing_status": accepted["pricing"]["status"],
            "total_wall_seconds": accepted["total_wall_time_seconds"],
            "raw_result_sha256": accepted["_sha256"],
        },
        {
            "gap_label": "1e-4",
            "requested_mip_gap": 1e-4,
            "status": incomplete["status"],
            "objective": incomplete["objective"],
            "bound": incomplete["bound"],
            "achieved_mip_gap": incomplete["mip_gap"],
            "commitment_count": sum(incomplete_commitment),
            "recorded_rounds": len(incomplete["constraint_generation_rounds"]),
            "accepted_added_security_pairs": incomplete["added_security_pairs"],
            "last_screen_new_violated_pairs": last_incomplete_screen[
                "new_violated_pairs"
            ],
            "last_screen_maximum_violation_pu": last_incomplete_screen[
                "maximum_violation_pu"
            ],
            "independent_verification_passed": None,
            "pricing_status": None,
            "total_wall_seconds": incomplete["total_wall_time_seconds"],
            "raw_result_sha256": incomplete["_sha256"],
        },
    ]
    for label in GAPS[2:]:
        summary_rows.append(
            {
                "gap_label": label,
                "requested_mip_gap": float(label),
                "status": "not_started_after_1e-4_failure",
                "objective": None,
                "bound": None,
                "achieved_mip_gap": None,
                "commitment_count": None,
                "recorded_rounds": None,
                "accepted_added_security_pairs": None,
                "last_screen_new_violated_pairs": None,
                "last_screen_maximum_violation_pu": None,
                "independent_verification_passed": None,
                "pricing_status": None,
                "total_wall_seconds": None,
                "raw_result_sha256": None,
            }
        )
    _write_csv(REPORT_ROOT / "summary.csv", summary_rows)

    round_rows: list[dict[str, Any]] = []
    for label, result in (("1e-3", accepted), ("1e-4", incomplete)):
        for record in result["constraint_generation_rounds"]:
            solve = record["solve"]
            screen = record["screen"]
            added = record.get("added_pair_ids", [])
            if screen["new_violated_pairs"] == 0:
                disposition = "no_new_violations"
            elif len(added) == screen["new_violated_pairs"]:
                disposition = "accepted_and_added"
            else:
                disposition = "provisional_not_added"
            round_rows.append(
                {
                    "gap_label": label,
                    "round": record["round"],
                    "rows_before_solve": record["rows_before_solve"],
                    "solver_status": solve["status"],
                    "solver_optimal": solve["optimal"],
                    "objective": solve["objective"],
                    "bound": solve["bound"],
                    "mip_gap": solve["mip_gap"],
                    "mip_nodes": solve["statistics"]["mip_node_count"],
                    "simplex_iterations": solve["statistics"][
                        "simplex_iteration_count"
                    ],
                    "partial_mip_start_status": solve["statistics"][
                        "partial_integer_mip_start_return_status"
                    ],
                    "solve_seconds": solve["solve_time_seconds"],
                    "screen_evaluated_sides": screen["evaluated_sides"],
                    "screen_new_violated_pairs": screen["new_violated_pairs"],
                    "screen_maximum_violation_pu": screen["maximum_violation_pu"],
                    "screen_maximum_pair_id": screen["maximum_pair_id"],
                    "screen_seconds": screen["wall_time_seconds"],
                    "screen_disposition": disposition,
                }
            )
    _write_csv(REPORT_ROOT / "round-detail.csv", round_rows)

    generator_rows: list[dict[str, Any]] = []
    for source_row, accepted_row, incomplete_row, pricing_row in zip(
        source,
        accepted_generators,
        incomplete_generators,
        pricing_generators,
        strict=True,
    ):
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
                "gap_1e-3_commitment": round(float(accepted_row["commitment"])),
                "gap_1e-3_mip_dispatch_mw": accepted_row["dispatch_mw"],
                "gap_1e-3_mip_dispatch_pu": accepted_row["dispatch_pu"],
                "gap_1e-3_pricing_dispatch_mw": pricing_row[
                    "pricing_dispatch_mw"
                ],
                "gap_1e-3_pricing_dispatch_pu": pricing_row[
                    "pricing_dispatch_pu"
                ],
                "gap_1e-3_price_per_mwh": pricing_row["nodal_price_per_mwh"],
                "gap_1e-3_price_per_pu_hour": pricing_row[
                    "nodal_price_per_pu_hour"
                ],
                "gap_1e-4_incumbent_status": "provisional_time_limit",
                "gap_1e-4_commitment": round(float(incomplete_row["commitment"])),
                "gap_1e-4_mip_dispatch_mw": incomplete_row["dispatch_mw"],
                "gap_1e-4_mip_dispatch_pu": incomplete_row["dispatch_pu"],
                "gap_1e-4_pricing_dispatch_mw": None,
                "gap_1e-4_pricing_dispatch_pu": None,
                "gap_1e-4_price_per_mwh": None,
                "gap_1e-4_price_per_pu_hour": None,
            }
        )
    _write_csv(REPORT_ROOT / "generator-detail.csv", generator_rows)

    bus_rows = [
        {
            "bus": record["bus"],
            "gap_1e-3_price_per_mwh": record["price_per_mwh"],
            "gap_1e-3_price_per_pu_hour": record["price_per_pu_hour"],
            "gap_1e-4_price_per_mwh": None,
            "gap_1e-4_price_per_pu_hour": None,
        }
        for record in bus_prices
    ]
    _write_csv(REPORT_ROOT / "bus-prices.csv", bus_rows)

    comparison = {
        "schema_version": "1.0.0",
        "experiment_suite_id": "activsg2000-gap-sensitivity-v1",
        "campaign_status": "stopped_after_1e-4_incomplete",
        "stop_rule": (
            "Do not start a later gap after timeout, nonoptimal termination, "
            "verification failure, or missing pricing."
        ),
        "not_started_gap_labels": list(GAPS[2:]),
        "frozen_identity": {
            "commit": EXPECTED_COMMIT,
            "tag": EXPECTED_TAG,
            "config_sha256_by_gap": {
                "1e-3": accepted["frozen_identity"]["config_sha256"],
                "1e-4": incomplete["frozen_identity"]["config_sha256"],
            },
        },
        "source_identity": accepted["source_manifest"]["source_identity"],
        "source_dimensions": accepted["source_manifest"]["dimensions"],
        "base_mva": base_mva,
        "raw_result_evidence": {
            "1e-3": {
                "path": str(accepted["_path"].relative_to(ROOT)),
                "sha256": accepted["_sha256"],
                "bytes": accepted["_bytes"],
            },
            "1e-4": {
                "path": str(incomplete["_path"].relative_to(ROOT)),
                "sha256": incomplete["_sha256"],
                "bytes": incomplete["_bytes"],
            },
        },
        "summary": summary_rows,
        "accepted_1e-3": {
            "maximum_model_residual_pu": accepted["maximum_model_residual_pu"],
            "final_exhaustive_violation_pu": accepted[
                "final_exhaustive_violation_pu"
            ],
            "verification": accepted["verification"],
            "pricing": {
                "status": accepted["pricing"]["status"],
                "pricing_lp_objective": accepted["pricing"][
                    "pricing_lp_objective"
                ],
                "mip_minus_pricing_lp_objective": accepted["pricing"][
                    "mip_minus_pricing_lp_objective"
                ],
                "pricing_dispatch_difference": accepted["pricing"][
                    "pricing_dispatch_difference"
                ],
                "bus_price_summary_per_mwh": accepted["pricing"][
                    "bus_price_summary_per_mwh"
                ],
                "bus_price_summary_per_pu_hour": accepted["pricing"][
                    "bus_price_summary_per_pu_hour"
                ],
            },
        },
        "incomplete_1e-4": {
            "solver_status": incomplete["solver_status"],
            "active_stage": incomplete["active_stage"],
            "last_completed_screen": last_incomplete_screen,
            "independent_verification_reached": False,
            "pricing_reached": False,
        },
        "accepted_1e-3_vs_provisional_1e-4": provisional_comparison,
        "record_audits": audits,
    }
    (REPORT_ROOT / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    accepted_dispatch = sum(
        float(record["dispatch_mw"]) for record in accepted_generators
    )
    accepted_dispatch_pu = accepted_dispatch / base_mva
    price_mw = accepted["pricing"]["bus_price_summary_per_mwh"]
    price_pu = accepted["pricing"]["bus_price_summary_per_pu_hour"]
    dispatch_difference = accepted["pricing"]["pricing_dispatch_difference"]
    accepted_table_row = (
        f"| 1e-3 | `optimal_verified` | {accepted['objective']:,.6f} | "
        f"{accepted['bound']:,.6f} | {accepted['mip_gap']:.6e} | "
        f"{accepted['commitment_count']} | "
        f"{accepted['total_wall_time_seconds']:.3f} | accepted |"
    )
    incomplete_table_row = (
        "| 1e-4 | `incomplete_restricted_master_not_optimal` | "
        f"{incomplete['objective']:,.6f} | {incomplete['bound']:,.6f} | "
        f"{incomplete['mip_gap']:.6e} | {sum(incomplete_commitment)} provisional | "
        f"{incomplete['total_wall_time_seconds']:.3f} | not reached |"
    )
    incomplete_nodes = incomplete["constraint_generation_rounds"][-1]["solve"][
        "statistics"
    ]["mip_node_count"]
    readme = f"""# ACTIVSg2000 MIP-gap sensitivity v1

The bounded campaign stopped at `1e-4`, exactly as registered. The `1e-3` run
completed `optimal_verified` with accepted fixed-commitment pricing in
{accepted['total_wall_time_seconds']:.3f} seconds. The `1e-4` run reached its
solver budget in round 2 after {incomplete['total_wall_time_seconds']:.3f}
seconds with an incumbent gap of {incomplete['mip_gap']:.6e}, above its requested
`1e-4`. Therefore `1e-5`, `1e-6`, and `1e-7` were not started.

| Gap | Status | Objective | Bound | Gap | Committed | Wall (s) | Pricing |
|---:|---|---:|---:|---:|---:|---:|---|
{accepted_table_row}
{incomplete_table_row}
| 1e-5 | not started | N/A | N/A | N/A | N/A | N/A | N/A |
| 1e-6 | not started | N/A | N/A | N/A | N/A | N/A | N/A |
| 1e-7 | not started | N/A | N/A | N/A | N/A | N/A | N/A |

## Accepted 1e-3 grid result

The accepted result commits {accepted['commitment_count']} of the
{accepted['source_manifest']['dimensions']['source_online_generators']}
source-online generators and dispatches {accepted_dispatch:,.6f} MW
({accepted_dispatch_pu:,.6f} p.u.). Its objective is
{accepted['objective']:,.6f}, bound is {accepted['bound']:,.6f}, and achieved
gap is {accepted['mip_gap']:.6e}. Three dynamic constraint-generation rounds
added {accepted['added_security_pairs']} pairs. The final exhaustive violation
is {accepted['final_exhaustive_violation_pu']:.3e} p.u., the maximum independently
checked model residual is {accepted['maximum_model_residual_pu']:.3e} p.u., and
all {accepted['verification']['checked_security_sides']:,} contingency sides
passed the independent checker.

The fixed-commitment pricing LP is independently N-1 secure. Its dispatch differs
from the MIP dispatch by at most
{dispatch_difference['maximum_absolute_mw']:.6f} MW and
{dispatch_difference['sum_absolute_mw']:.6f} MW in aggregate. Prices range from
${price_mw['minimum']:,.6f}/MWh to ${price_mw['maximum']:,.6f}/MWh, with mean
${price_mw['mean']:,.6f}/MWh. On the {base_mva:g} MVA base, the range is
${price_pu['minimum']:,.6f} to ${price_pu['maximum']:,.6f} per p.u.-hour.

## Why 1e-4 is not comparable as a final grid solution

Round 1 added 87 violated pairs. Round 2 used the prior commitment as a partial
MIP start, processed {incomplete_nodes:,}
nodes, and stopped with HiGHS `TimeLimit`. A provisional exhaustive screen of
that incumbent then found 18 more violated pairs, with maximum violation
{last_incomplete_screen['maximum_violation_pu']:.6f} p.u. Those pairs were not
promoted into a new accepted master because round 2 had not met `1e-4`.
Independent verification and pricing were therefore not run.

For diagnostic context only, the provisional incumbent has
{sum(incomplete_commitment)} committed units, differs from the accepted `1e-3`
commitment at {provisional_comparison['commitment_flip_count']} source rows, and
has {provisional_comparison['mip_dispatch_mw']['sum_absolute']:,.6f} MW of
absolute dispatch difference. These are not accepted `1e-4` modeling results.

## Evidence files

- `generator-detail.csv`: all 544 exact source PMIN/PMAX rows, accepted `1e-3`
  commitment/MIP dispatch/pricing dispatch/price, and the clearly labeled
  provisional `1e-4` commitment/dispatch. The `1e-4` pricing fields are blank
  because no valid pricing solve occurred.
- `bus-prices.csv`: all 2,000 accepted `1e-3` nodal prices in $/MWh and
  $/p.u.-hour. The `1e-4` price fields are blank.
- `round-detail.csv`: solve, MIP-start, screening, and disposition evidence for
  every recorded internal round.
- `summary.csv` and `comparison.json`: campaign status, raw-result hashes,
  numerical checks, and the explicit stop gate.

The source-record audit found zero PMIN/PMAX substitutions, p.u. conversion
mismatches, conditional limit violations above `1e-6` MW, unavailable-generator
violations, or price conversion mismatches. ACTIVSg2000 is synthetic; this result
does not establish a suitable gap for a different case or formulation.
"""
    (REPORT_ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
