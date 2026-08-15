"""Build tracked evidence for the corrected-start ACTIVSg2000 Spark v9 run."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v9.json"
EXPERIMENTS = ROOT / "results" / "experiments"
DIAGNOSTICS = ROOT / "results" / "diagnostics"
CHECKPOINTS = ROOT / "results" / "checkpoints"
RESULT = EXPERIMENTS / "activsg2000-gpu-gap-v9-1e-3-dgx-spark.json"
REGISTRY = EXPERIMENTS / "activsg2000-gpu-gap-sensitivity-v9-run-registry.json"
LAUNCHER_CONSOLE = EXPERIMENTS / "activsg2000-gpu-gap-v9-launcher.log"
CUOPT_CONSOLE = EXPERIMENTS / "activsg2000-gpu-gap-v9-1e-3-cuopt-console.log"
EVENTS = DIAGNOSTICS / "activsg2000-gpu-gap-v9-1e-3-dgx_spark-events.jsonl"
WORKER_CONSOLE = (
    DIAGNOSTICS
    / "activsg2000-gpu-gap-v9-1e-3-dgx_spark-worker-console.log"
)
VERIFY = (
    DIAGNOSTICS / "activsg2000-gpu-gap-v9-1e-3-independent-verification.json"
)
CHECKPOINT = (
    CHECKPOINTS / "activsg2000-gpu-gap-v9-1e-3-dgx_spark.json"
)
INVALID_START_LOG = (
    DIAGNOSTICS
    / "activsg2000-v8-round2-prechecked-prior-gpu-start-901be5d.log"
)
VALID_START_LOG = (
    DIAGNOSTICS
    / "activsg2000-v8-round2-secure-full-start-c3bbc80.log"
)
REPORT = ROOT / "reports" / "activsg2000-gpu-1e-3-v9"
FROZEN_COMMIT = "c3bbc80db65aff27287ed479c48dcc2cb8d20f99"
FROZEN_TAG = "experiment-2000-gpu-gap-v9"
CONFIG_SHA256 = "2ec50634b9c361da8d5801486410a44d9b0ae99ebb5f921f807be68e5bfa9386"
EXPECTED_ADDITIONS = [173, 13, 2]
REQUESTED_GAP = 1e-3
SECURITY_TOLERANCE_PU = 1e-5


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _structured_log(path: Path) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    marker = text.rfind("\n{\n")
    if marker < 0:
        raise ValueError(f"No trailing structured payload in {path}")
    return text, json.loads(text[marker + 1 :])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _audit(
    config: dict[str, Any],
    result: dict[str, Any],
    verification: dict[str, Any],
    invalid_start: dict[str, Any],
    valid_start: dict[str, Any],
    valid_start_text: str,
) -> None:
    if _digest(CONFIG)["sha256"] != CONFIG_SHA256:
        raise ValueError("Frozen v9 configuration changed")
    if result["status"] != "deadline_budget_exhausted":
        raise ValueError("Unexpected v9 run status")
    if result["frozen_identity"] != {
        "commit": FROZEN_COMMIT,
        "config_sha256": CONFIG_SHA256,
        "tag": FROZEN_TAG,
    }:
        raise ValueError("Frozen v9 identity changed")
    if config["benchmark"]["initialization"]["external_cpu_mip_start"]:
        raise ValueError("V9 unexpectedly enabled an external CPU start")
    rounds = result["constraint_generation_rounds"]
    if len(rounds) != 3:
        raise ValueError("V9 must preserve exactly three completed rounds")
    additions = [len(record["added_pair_ids"]) for record in rounds]
    if additions != EXPECTED_ADDITIONS:
        raise ValueError(f"V9 pair additions changed: {additions}")
    for record in rounds:
        statistics = record["solve"]["statistics"]
        if statistics["native_log_audit"]["mip_start_rejection_count"]:
            raise ValueError("V9 contains a native MIP-start rejection")
        if statistics["native_log_audit"]["barrier_numerical_warning_count"]:
            raise ValueError("V9 contains a native barrier numerical warning")
    for record in rounds[1:]:
        statistics = record["solve"]["statistics"]
        precheck = statistics["mip_start_feasibility_precheck"]
        contract = statistics["mip_start_native_contract"]
        if precheck["prior_commitment_extendable"]:
            raise ValueError("A later v9 prior commitment unexpectedly extended")
        if precheck["decision"] != "solve_cold":
            raise ValueError("An unextendable v9 commitment did not solve cold")
        if contract["submitted"] or statistics["mip_start_columns"]:
            raise ValueError("V9 submitted an unextendable MIP start")
    gates = result["acceptance_gates"]
    if gates["requested_mip_gap_certified"]:
        raise ValueError("V9 unexpectedly certified the requested gap")
    if gates["final_exhaustive_screen_zero_above_tolerance"]:
        raise ValueError("V9 unexpectedly passed the security gate")
    if result.get("pricing") is not None:
        raise ValueError("Incomplete v9 result must not contain pricing")
    if verification["passed"]:
        raise ValueError("Independent v9 verification unexpectedly passed")
    if verification["checked_security_sides"] != 17_563_400:
        raise ValueError("Independent v9 verification is not exhaustive")
    if invalid_start["passed"] is not True:
        raise ValueError("Invalid-start diagnostic did not pass")
    invalid_precheck = invalid_start["solve"]["mip_start_feasibility_precheck"]
    if invalid_precheck["decision"] != "solve_cold":
        raise ValueError("Invalid-start diagnostic did not solve cold")
    valid_contract = valid_start["solve"]["mip_start_native_contract"]
    if not valid_start["passed"] or not valid_contract["contract_passed"]:
        raise ValueError("Valid full-start diagnostic did not pass")
    if valid_contract["submitted_canonical_columns"] != 9220:
        raise ValueError("Valid diagnostic canonical start length changed")
    if valid_contract["submitted_native_columns"] != 11214:
        raise ValueError("Valid diagnostic native start length changed")
    if valid_contract["explicit_free_split_columns"] != 1999:
        raise ValueError("Valid diagnostic free split changed")
    if valid_contract["explicit_eliminated_columns"] != 5:
        raise ValueError("Valid diagnostic elimination count changed")
    if "Adding initial solution success!" not in valid_start_text:
        raise ValueError("Native valid-start acceptance message is absent")
    if "Error cannot add the provided initial solution!" in valid_start_text:
        raise ValueError("Valid-start diagnostic contains a native rejection")


def _generator_rows(
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = result["source_manifest"]["generators"]
    solution = result["solution"]["generators"]
    if len(source) != 544 or len(solution) != 544:
        raise ValueError("Expected all 544 source generator rows")
    rows: list[dict[str, Any]] = []
    committed = 0
    dispatch_mw = 0.0
    for source_row, solution_row in zip(source, solution, strict=True):
        if source_row["source_id"] != solution_row["source_id"]:
            raise ValueError("Generator source identity changed")
        for field in ("pmin_mw", "pmax_mw"):
            if float(source_row[field]) != float(solution_row[field]):
                raise ValueError(f"Exact source {field} changed")
        commitment = round(float(solution_row["commitment"]))
        dispatch = float(solution_row["dispatch_mw"])
        pmin = float(source_row["pmin_mw"])
        pmax = float(source_row["pmax_mw"])
        if not pmin * commitment - 1e-6 <= dispatch <= pmax * commitment + 1e-6:
            raise ValueError("Conditional exact PMIN/PMAX violation")
        if abs(float(solution_row["dispatch_pu"]) - dispatch / 100.0) > 1e-10:
            raise ValueError("Generator p.u. conversion changed")
        committed += commitment
        dispatch_mw += dispatch
        rows.append(
            {
                "source_id": source_row["source_id"],
                "source_row": source_row["source_row"],
                "bus": source_row["bus"],
                "source_status": source_row["source_status"],
                "pmin_mw": pmin,
                "pmin_pu": pmin / 100.0,
                "pmax_mw": pmax,
                "pmax_pu": pmax / 100.0,
                "v9_provisional_commitment": commitment,
                "v9_provisional_dispatch_mw": dispatch,
                "v9_provisional_dispatch_pu": solution_row["dispatch_pu"],
                "result_status": result["status"],
                "mip_gap": result["mip_gap"],
                "requested_mip_gap": REQUESTED_GAP,
                "security_status": "failed_two_unresolved_pairs",
                "price_status": "withheld_gap_and_security_not_certified",
                "price_per_mwh": None,
                "price_per_pu_hour": None,
            }
        )
    return rows, {
        "commitment_count": committed,
        "dispatch_total_mw": dispatch_mw,
        "dispatch_total_pu": dispatch_mw / 100.0,
    }


def _round_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in result["constraint_generation_rounds"]:
        solve = record["solve"]
        statistics = solve["statistics"]
        screen = record["screen"]
        precheck = statistics.get("mip_start_feasibility_precheck") or {}
        contract = statistics["mip_start_native_contract"]
        rows.append(
            {
                "round": record["round"],
                "rows_before_solve": record["rows_before_solve"],
                "solver_budget_seconds": record["solver_budget_seconds"],
                "solver_status": solve["status"],
                "objective": solve["objective"],
                "bound": solve["bound"],
                "mip_gap": solve["mip_gap"],
                "requested_gap_certified": solve["requested_gap_certified"],
                "precheck_status": precheck.get("solver_status"),
                "precheck_extendable": precheck.get("prior_commitment_extendable"),
                "precheck_decision": precheck.get("decision"),
                "precheck_wall_seconds": precheck.get("wall_time_seconds"),
                "native_start_submitted": contract["submitted"],
                "native_start_columns": contract["submitted_native_columns"],
                "native_start_rejections": statistics["native_log_audit"][
                    "mip_start_rejection_count"
                ],
                "native_barrier_warnings": statistics["native_log_audit"][
                    "barrier_numerical_warning_count"
                ],
                "native_solve_seconds": solve["solve_time_seconds"],
                "adapter_wall_seconds": record["adapter_wall_time_seconds"],
                "nodes": statistics["num_nodes"],
                "simplex_iterations": statistics["num_simplex_iterations"],
                "screen_seconds": screen["wall_time_seconds"],
                "screen_evaluated_sides": screen["evaluated_sides"],
                "screen_new_violated_pairs": screen["new_violated_pairs"],
                "screen_maximum_violation_pu": screen["maximum_violation_pu"],
                "added_pair_count": len(record["added_pair_ids"]),
            }
        )
    return rows


def main() -> None:
    config = _load(CONFIG)
    result = _load(RESULT)
    verification = _load(VERIFY)
    invalid_text, invalid_start = _structured_log(INVALID_START_LOG)
    valid_text, valid_start = _structured_log(VALID_START_LOG)
    _audit(
        config,
        result,
        verification,
        invalid_start,
        valid_start,
        valid_text,
    )
    REPORT.mkdir(parents=True, exist_ok=True)
    generators, generator_summary = _generator_rows(result)
    rounds = _round_rows(result)
    _write_csv(REPORT / "generator-detail.csv", generators)
    _write_csv(REPORT / "round-detail.csv", rounds)
    _write_csv(
        REPORT / "bus-prices.csv",
        [
            {
                "bus": row["bus"],
                "price_status": "withheld_gap_and_security_not_certified",
                "price_per_mwh": None,
                "price_per_pu_hour": None,
            }
            for row in result["solution"]["bus_angles_rad"]
        ],
    )
    solver_rounds_seconds = sum(row["adapter_wall_seconds"] for row in rounds)
    screen_seconds = sum(row["screen_seconds"] for row in rounds)
    unresolved_pairs = result["constraint_generation_rounds"][-1][
        "added_pair_ids"
    ]
    summary_row = {
        "status": result["status"],
        "objective": result["objective"],
        "bound": result["bound"],
        "absolute_gap": float(result["objective"] - result["bound"]),
        "mip_gap": result["mip_gap"],
        "requested_mip_gap": REQUESTED_GAP,
        "gap_certified": False,
        "final_screen_zero_violations": False,
        "final_screen_new_violations": 2,
        "final_screen_maximum_violation_pu": result["last_completed_screen"][
            "maximum_violation_pu"
        ],
        "independent_verification_passed": False,
        "maximum_model_residual_pu": verification["maximum_model_residual_pu"],
        "commitment_count": generator_summary["commitment_count"],
        "dispatch_total_mw": generator_summary["dispatch_total_mw"],
        "pricing_status": "withheld_gap_and_security_not_certified",
        "constraint_generation_rounds": len(rounds),
        "added_security_pairs": len(result["added_security_pair_ids"]),
        "solver_rounds_seconds": solver_rounds_seconds,
        "screen_seconds": screen_seconds,
        "wall_seconds": result["total_wall_time_seconds"],
    }
    _write_csv(REPORT / "summary.csv", [summary_row])
    evidence = {
        "schema_version": "1.0.0",
        "campaign": "activsg2000-gpu-gap-sensitivity-v9",
        "success": False,
        "status": result["status"],
        "failure_gates": [
            "requested_mip_gap_not_certified",
            "final_exhaustive_screen_has_two_violations",
            "independent_security_verification_failed",
        ],
        "frozen_identity": result["frozen_identity"],
        "source_identity": result["source_manifest"]["source_identity"],
        "initialization": config["benchmark"]["initialization"],
        "mip_start_behavior": {
            "round_1": "cold",
            "round_2": rounds[1]["precheck_decision"],
            "round_2_precheck_status": rounds[1]["precheck_status"],
            "round_3": rounds[2]["precheck_decision"],
            "round_3_precheck_status": rounds[2]["precheck_status"],
            "native_starts_submitted_in_official_run": 0,
            "native_start_rejections_in_official_run": 0,
            "valid_full_start_translation_diagnostic": valid_start,
            "unextendable_prior_start_diagnostic": invalid_start,
        },
        "postrun_safety_fix": {
            "package_version": "0.18.1",
            "change": (
                "HiGHS has_incumbent now requires explicit feasible primal "
                "solution status; Unknown vectors cannot be reused"
            ),
            "official_v9_math_affected": False,
            "official_v9_start_decision_affected": False,
        },
        "objective": result["objective"],
        "bound": result["bound"],
        "absolute_gap": float(result["objective"] - result["bound"]),
        "mip_gap": result["mip_gap"],
        "requested_mip_gap": REQUESTED_GAP,
        **generator_summary,
        "constraint_generation_rounds": rounds,
        "added_security_pairs": len(result["added_security_pair_ids"]),
        "unresolved_pair_ids": unresolved_pairs,
        "final_model_after_unresolved_rows": {
            "rows": 8976,
            "columns": 9220,
            "nonzeros": 27150,
            "last_solved_rows": 8974,
        },
        "controller_acceptance_gates": result["acceptance_gates"],
        "postrun_independent_verification": verification,
        "pricing_status": "withheld_gap_and_security_not_certified",
        "solver_rounds_seconds": solver_rounds_seconds,
        "screen_seconds": screen_seconds,
        "total_wall_time_seconds": result["total_wall_time_seconds"],
        "peak_memory": result["peak_memory"],
        "raw_evidence": {
            "config": _digest(CONFIG),
            "result": _digest(RESULT),
            "registry": _digest(REGISTRY),
            "checkpoint": _digest(CHECKPOINT),
            "events": _digest(EVENTS),
            "worker_console": _digest(WORKER_CONSOLE),
            "launcher_console": _digest(LAUNCHER_CONSOLE),
            "cuopt_console": _digest(CUOPT_CONSOLE),
            "independent_verification": _digest(VERIFY),
            "invalid_start_diagnostic": _digest(INVALID_START_LOG),
            "valid_start_diagnostic": _digest(VALID_START_LOG),
        },
    }
    (REPORT / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for source, name in (
        (EVENTS, "events.jsonl"),
        (WORKER_CONSOLE, "worker-console.log"),
        (LAUNCHER_CONSOLE, "launcher-console.log"),
        (CUOPT_CONSOLE, "cuopt-console.log"),
        (VERIFY, "independent-verification.json"),
        (INVALID_START_LOG, "unextendable-start-diagnostic.log"),
        (VALID_START_LOG, "valid-full-start-diagnostic.log"),
    ):
        shutil.copyfile(source, REPORT / name)
    first, second, third = rounds
    readme = f"""# ACTIVSg2000 DGX Spark corrected-start v9 result

The single authorized v9 run completed cleanly as
`deadline_budget_exhausted` after {result['total_wall_time_seconds']:.3f}
end-to-end seconds. It is not infeasible, but it is also not a successful
N-1 SCOPF result. The final solved master has objective
{result['objective']:,.6f}, lower bound {result['bound']:,.6f}, and relative
gap {result['mip_gap']:.7f}, above the requested `0.001`.

V9 fixed the rejected-start defect. Before each later solve, a bounded HiGHS
LP fixed the prior GPU commitment and tested whether continuous redispatch
could satisfy the expanded master. Round 2's prior commitment was proven
infeasible under the 173 new rows, so round 2 solved cold. Round 3's precheck
did not produce an acceptable feasible completion, so round 3 also solved
cold. No start reached cuOpt in the official run, and the native audit records
zero MIP-start rejections and zero barrier warnings. There was no external CPU
initialization.

The separately approved exact translation diagnostic proves the usable-start
path itself: all 9,220 canonical values were translated into exactly 11,214
native values using 1,999 free-variable splits and five fixed/unused-column
eliminations. Presolve was disabled, native readback passed, and cuOpt printed
`Adding initial solution success!`. The unextendable prior-round diagnostic
separately proved the round-1 commitment infeasible for the round-2 master and
correctly solved that diagnostic master cold.

Round 1 solved in {first['adapter_wall_seconds']:.3f} adapter seconds and its
exhaustive screen added {first['added_pair_count']} pairs. Round 2 solved in
{second['adapter_wall_seconds']:.3f} seconds, certified a
{second['mip_gap']:.7f} restricted-master gap, then exposed and added
{second['added_pair_count']} more pairs. Round 3 used the remaining
{third['adapter_wall_seconds']:.3f} seconds, ending at gap
{third['mip_gap']:.7f}. Its mandatory exhaustive screen checked
{third['screen_evaluated_sides']:,} sides and exposed two more rows:
`{unresolved_pairs[0]}` and `{unresolved_pairs[1]}`. The maximum violation was
{third['screen_maximum_violation_pu']:.9f} p.u. The rows were added
deterministically, but no solve budget remained for round 4.

The post-run independent raw-input checker repeated all 17,563,400 security
sides in {verification['elapsed_seconds']:.3f} seconds. Base limits, exact
conditional source PMIN/PMAX, balance, DC equations, angles, integrality, and
the objective passed; maximum model residual was
{verification['maximum_model_residual_pu']:.3e} p.u. It independently found
the same maximum N-1 violation, so overall verification failed exactly at the
security gate.

The provisional incumbent commits {generator_summary['commitment_count']}
units and dispatches {generator_summary['dispatch_total_mw']:,.6f} MW.
`generator-detail.csv` preserves all 544 source rows with exact PMIN/PMAX and
provisional commitment/dispatch in MW and p.u. Pricing is intentionally
withheld because neither the requested gap nor N-1 security was certified;
`bus-prices.csv` preserves all 2,000 bus identities with blank price fields.

Post-run inspection found one conservative telemetry defect in the HiGHS
precheck: an `Unknown` status could label a finite but grossly infeasible
vector as `has_incumbent`, although the independent residual gate still
rejected it and solved cold. Version 0.18.1 now also requires HiGHS' explicit
feasible-primal status before exposing or reusing a vector. This did not alter
the frozen v9 solve or its safe cold-start decision. No retry was performed.
"""
    (REPORT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
