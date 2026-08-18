"""Build tracked evidence for the 30-minute ACTIVSg2000 Spark v7 run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "results" / "experiments"
DIAGNOSTIC_ROOT = ROOT / "results" / "diagnostics"
CONFIG_PATH = ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v7.json"
GPU_PATH = EXPERIMENT_ROOT / "activsg2000-gpu-gap-v7-1e-3-dgx-spark.json"
REGISTRY_PATH = (
    EXPERIMENT_ROOT / "activsg2000-gpu-gap-sensitivity-v7-run-registry.json"
)
LAUNCHER_CONSOLE_PATH = (
    EXPERIMENT_ROOT / "activsg2000-gpu-gap-v7-1e-3-cuopt-console.log"
)
EVENT_PATH = (
    DIAGNOSTIC_ROOT / "activsg2000-gpu-gap-v7-1e-3-dgx_spark-events.jsonl"
)
WORKER_CONSOLE_PATH = (
    DIAGNOSTIC_ROOT
    / "activsg2000-gpu-gap-v7-1e-3-dgx_spark-worker-console.log"
)
VERIFY_PATH = (
    DIAGNOSTIC_ROOT
    / "activsg2000-gpu-gap-v7-1e-3-independent-verification.json"
)
V6_EVIDENCE_PATH = ROOT / "reports" / "activsg2000-gpu-1e-3-v6" / "evidence.json"
V6_GENERATOR_PATH = (
    ROOT / "reports" / "activsg2000-gpu-1e-3-v6" / "generator-detail.csv"
)
REPORT_ROOT = ROOT / "reports" / "activsg2000-gpu-1e-3-v7"
GPU_COMMIT = "6a387441d688333c425df8efd8cc6c71a2eb48d8"
SOLVER_ALLOWANCE_SECONDS = 1800.0
OUTER_BOUNDARY_SECONDS = 1935.0
PDLP_PROFILE = {
    "method": "pdlp",
    "solver_mode": "stable3",
    "precision": "fp64",
    "batch_strong_branching": True,
    "batch_reliability_branching": True,
    "reliability_branching_factor": 1,
}
PDLP_PARAMETERS = {
    "method": 1,
    "pdlp_solver_mode": 4,
    "pdlp_precision": 1,
    "mip_batch_pdlp_strong_branching": 1,
    "mip_batch_pdlp_reliability_branching": 1,
    "mip_reliability_branching": 1,
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload["_bytes"] = path.stat().st_size
    return payload


def _digest(path: Path) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _audit(
    config: dict[str, Any], gpu: dict[str, Any], verification: dict[str, Any]
) -> None:
    expected_status = "incomplete_restricted_master_gap_not_certified"
    if gpu.get("status") != expected_status:
        raise ValueError("Unexpected GPU v7 status")
    identity = gpu["frozen_identity"]
    if identity["commit"] != GPU_COMMIT:
        raise ValueError("GPU v7 commit changed")
    if identity["tag"] != "experiment-2000-gpu-gap-v7":
        raise ValueError("GPU v7 tag changed")
    if identity["config_sha256"] != config["_sha256"]:
        raise ValueError("GPU v7 config digest changed")
    platform = config["platforms"]["dgx_spark"]
    if platform["cuopt_pdlp_profile"] != PDLP_PROFILE:
        raise ValueError("GPU v7 PDLP profile changed")
    if config["benchmark"]["initialization"] != {
        "external_cpu_mip_start": False,
        "round_1": "cold",
        "later_rounds": "prior_gpu_integer_commitment_only",
    }:
        raise ValueError("GPU v7 initialization contract changed")
    runtime = config["runtime"]
    available = (
        float(runtime["deadline_seconds"])
        - float(runtime["verification_reserve_seconds"])
        - float(runtime["serialization_reserve_seconds"])
    )
    if available != SOLVER_ALLOWANCE_SECONDS:
        raise ValueError("GPU v7 no longer preserves its 1,800-second allowance")
    if float(runtime["deadline_seconds"]) != OUTER_BOUNDARY_SECONDS:
        raise ValueError("GPU v7 outer deadline changed")

    rounds = gpu.get("constraint_generation_rounds", [])
    if len(rounds) != 3:
        raise ValueError("GPU v7 must preserve exactly three rounds")
    additions = [len(record.get("added_pair_ids") or []) for record in rounds]
    if additions != [173, 14, 0]:
        raise ValueError(f"GPU v7 security-pair additions changed: {additions}")
    starts = [
        record["solve"]["statistics"]["partial_integer_mip_start_columns"]
        for record in rounds
    ]
    if starts != [0, 432, 432]:
        raise ValueError(f"GPU v7 round starts changed: {starts}")
    for record in rounds:
        statistics = record["solve"]["statistics"]
        if statistics["mip_start_mode"] != "integer_only":
            raise ValueError("GPU v7 used a non-integer MIP start")
        if statistics["cuopt_pdlp_profile"] != PDLP_PROFILE:
            raise ValueError("A v7 round changed its descriptive PDLP profile")
        if statistics["cuopt_pdlp_parameters_requested"] != PDLP_PARAMETERS:
            raise ValueError("A v7 round changed its requested PDLP parameters")
        if statistics["cuopt_pdlp_parameters_readback"] != PDLP_PARAMETERS:
            raise ValueError("A v7 round failed exact PDLP parameter readback")
    final_round = rounds[-1]
    if final_round["solve"].get("requested_gap_certified") is not False:
        raise ValueError("GPU v7 unexpectedly certifies the requested gap")
    if final_round["screen"].get("new_violated_pairs") != 0:
        raise ValueError("GPU v7 final exhaustive screen is not clean")
    if final_round["screen"].get("evaluated_sides") != 17_563_400:
        raise ValueError("GPU v7 final screen is not exhaustive")
    if float(gpu["total_wall_time_seconds"]) > OUTER_BOUNDARY_SECONDS:
        raise ValueError("GPU v7 exceeded its guarded end-to-end boundary")
    if verification.get("passed") is not True:
        raise ValueError("Independent verification of the final incumbent failed")
    if verification.get("checked_security_sides") != 17_563_400:
        raise ValueError("Independent verification is not exhaustive")
    if gpu.get("pricing") is not None:
        raise ValueError("Gap-uncertified GPU result must not contain pricing")

    console = WORKER_CONSOLE_PATH.read_text(encoding="utf-8")
    for name, value in PDLP_PARAMETERS.items():
        evidence = f"Setting parameter {name} to {value}"
        if console.count(evidence) != 3:
            raise ValueError(f"Native console did not record {evidence!r} three times")
    cooperative = "Cooperative batch PDLP and Dual Simplex for strong branching"
    if console.count(cooperative) != 3:
        raise ValueError("Native console did not confirm cooperative batch PDLP")
    rejection = "Error cannot add the provided initial solution!"
    if console.count(rejection) != 2:
        raise ValueError("GPU v7 native MIP-start rejection evidence changed")


def _generator_rows(
    gpu: dict[str, Any], v6_generators: dict[str, dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = gpu["source_manifest"]["generators"]
    generators = gpu["solution"]["generators"]
    if len(source) != 544 or len(generators) != 544:
        raise ValueError("Expected all 544 generator source rows")

    rows: list[dict[str, Any]] = []
    commitment_count = 0
    dispatch_total_mw = 0.0
    commitment_flips = 0
    v6_off_v7_on = 0
    v6_on_v7_off = 0
    dispatch_l1_difference_mw = 0.0
    max_dispatch_difference_mw = 0.0
    max_dispatch_difference_source_id = ""
    for source_row, result_row in zip(source, generators, strict=True):
        source_id = source_row["source_id"]
        if source_id != result_row["source_id"]:
            raise ValueError("Generator source identity changed")
        if any(
            float(source_row[field]) != float(result_row[field])
            for field in ("pmin_mw", "pmax_mw")
        ):
            raise ValueError("Generator source PMIN/PMAX changed")
        commitment = round(float(result_row["commitment"]))
        dispatch = float(result_row["dispatch_mw"])
        if not (
            float(source_row["pmin_mw"]) * commitment - 1e-6
            <= dispatch
            <= float(source_row["pmax_mw"]) * commitment + 1e-6
        ):
            raise ValueError("Final incumbent violates exact conditional PMIN/PMAX")
        if abs(float(result_row["dispatch_pu"]) - dispatch / 100.0) > 1e-10:
            raise ValueError("Final incumbent dispatch p.u. conversion changed")

        v6 = v6_generators[source_id]
        v6_commitment = int(v6["v6_gap_uncertified_secure_commitment"])
        v6_dispatch = float(v6["v6_gap_uncertified_secure_dispatch_mw"])
        dispatch_difference = dispatch - v6_dispatch
        if abs(dispatch_difference) > abs(max_dispatch_difference_mw):
            max_dispatch_difference_mw = dispatch_difference
            max_dispatch_difference_source_id = source_id
        commitment_flips += int(v6_commitment != commitment)
        v6_off_v7_on += int(v6_commitment == 0 and commitment == 1)
        v6_on_v7_off += int(v6_commitment == 1 and commitment == 0)
        dispatch_l1_difference_mw += abs(dispatch_difference)
        commitment_count += commitment
        dispatch_total_mw += dispatch
        rows.append(
            {
                "source_id": source_id,
                "source_row": source_row["source_row"],
                "bus": source_row["bus"],
                "source_status": source_row["source_status"],
                "pmin_mw": source_row["pmin_mw"],
                "pmin_pu": float(source_row["pmin_mw"]) / 100.0,
                "pmax_mw": source_row["pmax_mw"],
                "pmax_pu": float(source_row["pmax_mw"]) / 100.0,
                "v6_gap_uncertified_secure_commitment": v6_commitment,
                "v6_gap_uncertified_secure_dispatch_mw": v6_dispatch,
                "v6_gap_uncertified_secure_dispatch_pu": float(
                    v6["v6_gap_uncertified_secure_dispatch_pu"]
                ),
                "v7_gap_uncertified_secure_commitment": commitment,
                "v7_gap_uncertified_secure_dispatch_mw": dispatch,
                "v7_gap_uncertified_secure_dispatch_pu": result_row["dispatch_pu"],
                "commitment_changed_from_v6": v6_commitment != commitment,
                "dispatch_change_from_v6_mw": dispatch_difference,
                "mip_gap": gpu["mip_gap"],
                "requested_mip_gap": 1e-3,
                "independent_security_verification": "passed",
                "price_status": "withheld_mip_gap_not_certified",
                "price_per_mwh": None,
                "price_per_pu_hour": None,
            }
        )
    return rows, {
        "commitment_count": commitment_count,
        "dispatch_total_mw": dispatch_total_mw,
        "commitment_flips_from_v6": commitment_flips,
        "v6_off_v7_on": v6_off_v7_on,
        "v6_on_v7_off": v6_on_v7_off,
        "dispatch_l1_difference_from_v6_mw": dispatch_l1_difference_mw,
        "maximum_dispatch_difference_from_v6_mw": max_dispatch_difference_mw,
        "maximum_dispatch_difference_source_id": max_dispatch_difference_source_id,
    }


def _round_rows(gpu: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in gpu["constraint_generation_rounds"]:
        solve = record["solve"]
        statistics = solve["statistics"]
        screen = record["screen"]
        requested = statistics["cuopt_pdlp_parameters_requested"]
        readback = statistics["cuopt_pdlp_parameters_readback"]
        rows.append(
            {
                "round": record["round"],
                "rows_before_solve": record["rows_before_solve"],
                "solver_budget_seconds": record["solver_budget_seconds"],
                "native_status": statistics["native_status"],
                "adapter_status": solve["status"],
                "requested_gap_certified": solve["requested_gap_certified"],
                "objective": solve["objective"],
                "bound": solve["bound"],
                "mip_gap": solve["mip_gap"],
                "partial_integer_mip_start_columns": statistics[
                    "partial_integer_mip_start_columns"
                ],
                "native_solve_seconds": solve["solve_time_seconds"],
                "adapter_wall_seconds": record["adapter_wall_time_seconds"],
                "num_nodes": statistics.get("num_nodes"),
                "num_simplex_iterations": statistics.get("num_simplex_iterations"),
                "screen_seconds": screen["wall_time_seconds"],
                "screen_evaluated_sides": screen["evaluated_sides"],
                "screen_new_violated_pairs": screen["new_violated_pairs"],
                "screen_maximum_violation_pu": screen["maximum_violation_pu"],
                "pdlp_method_requested": requested["method"],
                "pdlp_method_readback": readback["method"],
                "pdlp_mode_requested": requested["pdlp_solver_mode"],
                "pdlp_mode_readback": readback["pdlp_solver_mode"],
                "pdlp_precision_requested": requested["pdlp_precision"],
                "pdlp_precision_readback": readback["pdlp_precision"],
                "batch_pdlp_strong_branching_readback": readback[
                    "mip_batch_pdlp_strong_branching"
                ],
                "batch_pdlp_reliability_branching_readback": readback[
                    "mip_batch_pdlp_reliability_branching"
                ],
                "mip_reliability_branching_readback": readback[
                    "mip_reliability_branching"
                ],
            }
        )
    return rows


def main() -> None:
    config = _load(CONFIG_PATH)
    gpu = _load(GPU_PATH)
    verification = _load(VERIFY_PATH)
    v6 = _load(V6_EVIDENCE_PATH)
    with V6_GENERATOR_PATH.open(newline="", encoding="utf-8") as stream:
        v6_generators = {row["source_id"]: row for row in csv.DictReader(stream)}
    _audit(config, gpu, verification)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    generator_rows, generator_summary = _generator_rows(gpu, v6_generators)
    _write_csv(REPORT_ROOT / "generator-detail.csv", generator_rows)
    _write_csv(
        REPORT_ROOT / "bus-prices.csv",
        [
            {
                "bus": row["bus"],
                "price_status": "withheld_mip_gap_not_certified",
                "price_per_mwh": None,
                "price_per_pu_hour": None,
            }
            for row in gpu["solution"]["bus_angles_rad"]
        ],
    )
    round_rows = _round_rows(gpu)
    _write_csv(REPORT_ROOT / "round-detail.csv", round_rows)

    solver_overrun = float(gpu["timings_seconds"]["solver_rounds"]) - (
        SOLVER_ALLOWANCE_SECONDS
    )
    comparison = {
        "v6_objective": v6["objective"],
        "v7_objective": gpu["objective"],
        "v7_minus_v6_objective": float(gpu["objective"] - v6["objective"]),
        "v6_bound": v6["bound"],
        "v7_bound": gpu["bound"],
        "v7_minus_v6_bound": float(gpu["bound"] - v6["bound"]),
        "v6_mip_gap": v6["mip_gap"],
        "v7_mip_gap": gpu["mip_gap"],
        "v7_minus_v6_mip_gap": float(gpu["mip_gap"] - v6["mip_gap"]),
        "v6_solver_rounds_seconds": v6["timings_seconds"]["solver_rounds"],
        "v7_solver_rounds_seconds": gpu["timings_seconds"]["solver_rounds"],
        "v7_solver_allowance_overrun_seconds": solver_overrun,
        "v6_wall_seconds": v6["total_wall_time_seconds"],
        "v7_wall_seconds": gpu["total_wall_time_seconds"],
        "v6_added_security_pairs": v6["added_security_pairs"],
        "v7_added_security_pairs": len(gpu["added_security_pair_ids"]),
        **generator_summary,
    }
    _write_csv(REPORT_ROOT / "comparison-v6.csv", [comparison])

    raw_evidence = {
        "config": _digest(CONFIG_PATH),
        "gpu_result": {"sha256": gpu["_sha256"], "bytes": gpu["_bytes"]},
        "independent_verification": {
            "sha256": verification["_sha256"],
            "bytes": verification["_bytes"],
        },
        "event_log": _digest(EVENT_PATH),
        "worker_console": _digest(WORKER_CONSOLE_PATH),
        "launcher_console": _digest(LAUNCHER_CONSOLE_PATH),
        "run_registry": _digest(REGISTRY_PATH),
        "v6_evidence": _digest(V6_EVIDENCE_PATH),
    }
    summary = {
        "schema_version": "1.0.0",
        "campaign": "activsg2000-gpu-gap-sensitivity-v7",
        "status": gpu["status"],
        "success": False,
        "failure_gate": "requested_mip_gap_not_certified",
        "initialization": config["benchmark"]["initialization"],
        "observed_mip_start_status": (
            "submitted_by_adapter_but_rejected_by_cuopt_in_rounds_2_and_3"
        ),
        "solver_allowance_seconds": SOLVER_ALLOWANCE_SECONDS,
        "solver_allowance_overrun_seconds": solver_overrun,
        "outer_boundary_seconds": OUTER_BOUNDARY_SECONDS,
        "outer_boundary_passed": (
            float(gpu["total_wall_time_seconds"]) <= OUTER_BOUNDARY_SECONDS
        ),
        "pdlp_profile": PDLP_PROFILE,
        "pdlp_parameters_requested_and_read_back": PDLP_PARAMETERS,
        "native_log_confirmation": (
            "Cooperative batch PDLP and Dual Simplex for strong branching"
        ),
        "frozen_identity": gpu["frozen_identity"],
        "source_identity": gpu["source_manifest"]["source_identity"],
        "objective": gpu["objective"],
        "bound": gpu["bound"],
        "absolute_gap": float(gpu["objective"] - gpu["bound"]),
        "mip_gap": gpu["mip_gap"],
        "requested_mip_gap": 1e-3,
        **generator_summary,
        "dispatch_total_pu": generator_summary["dispatch_total_mw"] / 100.0,
        "constraint_generation_rounds": round_rows,
        "added_security_pairs": len(gpu["added_security_pair_ids"]),
        "final_model_dimensions": gpu["final_model_dimensions"],
        "controller_acceptance_gates": gpu["acceptance_gates"],
        "post_run_independent_verification": {
            key: value for key, value in verification.items() if not key.startswith("_")
        },
        "pricing_status": "withheld_mip_gap_not_certified",
        "timings_seconds": gpu["timings_seconds"],
        "total_wall_time_seconds": gpu["total_wall_time_seconds"],
        "peak_memory": gpu["peak_memory"],
        "comparison_to_v6": comparison,
        "raw_evidence": raw_evidence,
    }
    (REPORT_ROOT / "evidence.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (REPORT_ROOT / "events.jsonl").write_bytes(EVENT_PATH.read_bytes())
    (REPORT_ROOT / "independent-verification.json").write_bytes(
        VERIFY_PATH.read_bytes()
    )
    (REPORT_ROOT / "worker-console.log").write_bytes(WORKER_CONSOLE_PATH.read_bytes())
    (REPORT_ROOT / "launcher-console.log").write_bytes(
        LAUNCHER_CONSOLE_PATH.read_bytes()
    )
    _write_csv(
        REPORT_ROOT / "summary.csv",
        [
            {
                "status": gpu["status"],
                "objective": gpu["objective"],
                "bound": gpu["bound"],
                "absolute_gap": float(gpu["objective"] - gpu["bound"]),
                "mip_gap": gpu["mip_gap"],
                "requested_mip_gap": 1e-3,
                "gap_certified": False,
                "final_screen_zero_violations": True,
                "independent_verification_passed": True,
                "commitment_count": generator_summary["commitment_count"],
                "dispatch_total_mw": generator_summary["dispatch_total_mw"],
                "pricing_status": "withheld_mip_gap_not_certified",
                "solver_allowance_seconds": SOLVER_ALLOWANCE_SECONDS,
                "solver_rounds_seconds": gpu["timings_seconds"]["solver_rounds"],
                "solver_allowance_overrun_seconds": solver_overrun,
                "outer_boundary_seconds": OUTER_BOUNDARY_SECONDS,
                "wall_seconds": gpu["total_wall_time_seconds"],
            }
        ],
    )

    first, second, third = round_rows
    readme = f"""# ACTIVSg2000 DGX Spark 30-minute PDLP v7 result

The one authorized v7 run ended
`incomplete_restricted_master_gap_not_certified` after
{gpu['total_wall_time_seconds']:.3f} seconds. It is not infeasible. Its final
incumbent is independently N-1 secure, but the requested `1e-3` MIP gap was
not certified, so the overall result remains incomplete and pricing is
intentionally withheld.

Round 1 started cold. The adapter submitted the prior GPU solution's
{second['partial_integer_mip_start_columns']} integer commitment columns in
rounds 2 and 3, but retrospective native-log inspection found that cuOpt
rejected both starts after internal model expansion. No CPU commitment, CPU
dispatch, or CPU bound initialized the run. V7 preserves the v6 exact-PMIN
model, inputs, tolerances, dynamic add-resolve-screen loop, and cuOpt PDLP
policy. Its substantive runtime change was increasing the cumulative cuOpt
allowance from 900 to 1,800 seconds.

Every round requested and read back method 1 (PDLP), Stable3 mode 4, FP64
precision 1, batched PDLP strong branching, batched PDLP reliability
branching, and reliability factor 1. The native log confirmed
`Cooperative batch PDLP and Dual Simplex for strong branching` in all three
rounds. This means PDLP assisted MIP branching; it does not mean every
branch-and-bound relaxation used PDLP alone.

Round 1 took {first['adapter_wall_seconds']:.3f} adapter seconds and added
{first['screen_new_violated_pairs']} contingency pairs. Round 2 took
{second['adapter_wall_seconds']:.3f} seconds and added
{second['screen_new_violated_pairs']} more. Round 3 took
{third['adapter_wall_seconds']:.3f} seconds and stopped on cuOpt's time limit
after exploring {third['num_nodes']:,} nodes. Its exhaustive screen checked
{third['screen_evaluated_sides']:,} sides and found zero violations above
`1e-5` p.u. The post-run independent raw-input checker also passed all
{verification['checked_security_sides']:,} sides, with maximum security
violation {verification['maximum_security_violation_pu']:.3e} p.u. and maximum
model residual {verification['maximum_model_residual_pu']:.3e} p.u.

The final objective is {gpu['objective']:,.6f}, the finite lower bound is
{gpu['bound']:,.6f}, and the reported relative gap is {gpu['mip_gap']:.7f}.
That gap is above `0.001`. The recorded restricted-master total was
{gpu['timings_seconds']['solver_rounds']:.3f} seconds, which is
{solver_overrun:.3f} seconds above the nominal 1,800-second allowance because
cuOpt returned slightly after its final native time limit. The full controller
still completed inside the separate 1,935-second outer guard. This overrun is
retained as evidence rather than normalized away.

Compared with v6, v7's incumbent objective is
{comparison['v7_minus_v6_objective']:+.6f}, its lower bound is
{comparison['v7_minus_v6_bound']:+.6f}, and its reported gap changes by
{comparison['v7_minus_v6_mip_gap']:+.7f}. V7 commits
{generator_summary['commitment_count']} units and dispatches
{generator_summary['dispatch_total_mw']:,.6f} MW. The fresh parallel MIP path
has {generator_summary['commitment_flips_from_v6']} commitment flips versus v6
({generator_summary['v6_off_v7_on']} off-to-on and
{generator_summary['v6_on_v7_off']} on-to-off) and an L1 dispatch difference
of {generator_summary['dispatch_l1_difference_from_v6_mw']:,.6f} MW. This is
not a fixed-master timing comparison: the dynamic pair paths differ.

`generator-detail.csv` preserves all 544 exact source PMIN/PMAX rows, v6 and
v7 commitment/dispatch values in MW and p.u., and blank price fields with the
withholding reason. `bus-prices.csv` preserves all 2,000 bus identities with
blank prices. The other files retain round timing, requested/read-back PDLP
parameters, event and native logs, hashes, and independent verification. No
v7 retry was performed.
"""
    (REPORT_ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
