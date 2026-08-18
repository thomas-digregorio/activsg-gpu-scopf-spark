"""Build tracked evidence for the ACTIVSg2000 Spark PDLP-policy v6 run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "results" / "experiments"
DIAGNOSTIC_ROOT = ROOT / "results" / "diagnostics"
CONFIG_PATH = ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v6.json"
GPU_PATH = EXPERIMENT_ROOT / "activsg2000-gpu-gap-v6-1e-3-dgx-spark.json"
REGISTRY_PATH = (
    EXPERIMENT_ROOT / "activsg2000-gpu-gap-sensitivity-v6-run-registry.json"
)
LAUNCHER_CONSOLE_PATH = (
    EXPERIMENT_ROOT / "activsg2000-gpu-gap-v6-1e-3-cuopt-console.log"
)
EVENT_PATH = (
    DIAGNOSTIC_ROOT
    / "activsg2000-gpu-gap-v6-1e-3-dgx_spark-events.jsonl"
)
WORKER_CONSOLE_PATH = (
    DIAGNOSTIC_ROOT
    / "activsg2000-gpu-gap-v6-1e-3-dgx_spark-worker-console.log"
)
VERIFY_PATH = (
    DIAGNOSTIC_ROOT
    / "activsg2000-gpu-gap-v6-1e-3-independent-verification.json"
)
V5_EVIDENCE_PATH = ROOT / "reports" / "activsg2000-gpu-1e-3-v5" / "evidence.json"
V5_GENERATOR_PATH = (
    ROOT / "reports" / "activsg2000-gpu-1e-3-v5" / "generator-detail.csv"
)
REPORT_ROOT = ROOT / "reports" / "activsg2000-gpu-1e-3-v6"
GPU_COMMIT = "b209112c51e373d65180bf21d75ae52440fd9df3"
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
    if gpu.get("status") != "incomplete_restricted_master_gap_not_certified":
        raise ValueError("Unexpected GPU v6 status")
    identity = gpu["frozen_identity"]
    if identity["commit"] != GPU_COMMIT:
        raise ValueError("GPU v6 commit changed")
    if identity["tag"] != "experiment-2000-gpu-gap-v6":
        raise ValueError("GPU v6 tag changed")
    if identity["config_sha256"] != config["_sha256"]:
        raise ValueError("GPU v6 config digest changed")
    if config["platforms"]["dgx_spark"]["cuopt_pdlp_profile"] != PDLP_PROFILE:
        raise ValueError("GPU v6 PDLP profile changed")
    if config["benchmark"]["initialization"] != {
        "external_cpu_mip_start": False,
        "round_1": "cold",
        "later_rounds": "prior_gpu_integer_commitment_only",
    }:
        raise ValueError("GPU v6 initialization contract changed")
    runtime = config["runtime"]
    available = (
        float(runtime["deadline_seconds"])
        - float(runtime["verification_reserve_seconds"])
        - float(runtime["serialization_reserve_seconds"])
    )
    if available != 900.0:
        raise ValueError("GPU v6 no longer preserves a 900-second solver allowance")
    rounds = gpu.get("constraint_generation_rounds", [])
    if len(rounds) != 3:
        raise ValueError("GPU v6 must preserve exactly three rounds")
    additions = [len(record.get("added_pair_ids") or []) for record in rounds]
    if additions != [173, 18, 0]:
        raise ValueError(f"GPU v6 security-pair additions changed: {additions}")
    starts = [
        record["solve"]["statistics"]["partial_integer_mip_start_columns"]
        for record in rounds
    ]
    if starts != [0, 432, 432]:
        raise ValueError(f"GPU v6 round starts changed: {starts}")
    for record in rounds:
        statistics = record["solve"]["statistics"]
        if statistics["mip_start_mode"] != "integer_only":
            raise ValueError("GPU v6 used a non-integer MIP start")
        if statistics["cuopt_pdlp_profile"] != PDLP_PROFILE:
            raise ValueError("A GPU v6 round changed its descriptive PDLP profile")
        if statistics["cuopt_pdlp_parameters_requested"] != PDLP_PARAMETERS:
            raise ValueError("A GPU v6 round changed its requested PDLP parameters")
        if statistics["cuopt_pdlp_parameters_readback"] != PDLP_PARAMETERS:
            raise ValueError("A GPU v6 round failed exact PDLP parameter readback")
    if rounds[2]["solve"].get("requested_gap_certified") is not False:
        raise ValueError("GPU v6 round 3 unexpectedly certifies the requested gap")
    if rounds[2]["screen"].get("new_violated_pairs") != 0:
        raise ValueError("GPU v6 round 3 did not preserve the zero-violation screen")
    if float(gpu["timings_seconds"]["solver_rounds"]) > 900.0:
        raise ValueError("GPU v6 exceeded the cumulative 900-second solver allowance")
    if float(gpu["total_wall_time_seconds"]) > 1035.0:
        raise ValueError("GPU v6 exceeded the end-to-end boundary")
    if verification.get("passed") is not True:
        raise ValueError("Independent verification of the final incumbent did not pass")
    if gpu.get("pricing") is not None:
        raise ValueError("Gap-uncertified GPU result must not contain accepted pricing")
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
        raise ValueError("GPU v6 native MIP-start rejection evidence changed")


def _generator_rows(
    gpu: dict[str, Any], v5_generators: dict[str, dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = gpu["source_manifest"]["generators"]
    generators = gpu["solution"]["generators"]
    if len(source) != 544 or len(generators) != 544:
        raise ValueError("Expected all 544 generator source rows")
    rows: list[dict[str, Any]] = []
    commitment_count = 0
    dispatch_total_mw = 0.0
    commitment_flips = 0
    v5_off_v6_on = 0
    v5_on_v6_off = 0
    dispatch_l1_difference_mw = 0.0
    for source_row, result_row in zip(source, generators, strict=True):
        if source_row["source_id"] != result_row["source_id"]:
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
        v5 = v5_generators[source_row["source_id"]]
        v5_commitment = int(v5["gap_uncertified_secure_commitment"])
        v5_dispatch = float(v5["gap_uncertified_secure_dispatch_mw"])
        commitment_flips += int(v5_commitment != commitment)
        v5_off_v6_on += int(v5_commitment == 0 and commitment == 1)
        v5_on_v6_off += int(v5_commitment == 1 and commitment == 0)
        dispatch_l1_difference_mw += abs(dispatch - v5_dispatch)
        commitment_count += commitment
        dispatch_total_mw += dispatch
        rows.append(
            {
                "source_id": source_row["source_id"],
                "source_row": source_row["source_row"],
                "bus": source_row["bus"],
                "source_status": source_row["source_status"],
                "pmin_mw": source_row["pmin_mw"],
                "pmin_pu": float(source_row["pmin_mw"]) / 100.0,
                "pmax_mw": source_row["pmax_mw"],
                "pmax_pu": float(source_row["pmax_mw"]) / 100.0,
                "v5_gap_uncertified_secure_commitment": v5_commitment,
                "v5_gap_uncertified_secure_dispatch_mw": v5_dispatch,
                "v6_gap_uncertified_secure_commitment": commitment,
                "v6_gap_uncertified_secure_dispatch_mw": dispatch,
                "v6_gap_uncertified_secure_dispatch_pu": result_row["dispatch_pu"],
                "commitment_changed_from_v5": v5_commitment != commitment,
                "dispatch_change_from_v5_mw": dispatch - v5_dispatch,
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
        "commitment_flips_from_v5": commitment_flips,
        "v5_off_v6_on": v5_off_v6_on,
        "v5_on_v6_off": v5_on_v6_off,
        "dispatch_l1_difference_from_v5_mw": dispatch_l1_difference_mw,
    }


def main() -> None:
    config = _load(CONFIG_PATH)
    gpu = _load(GPU_PATH)
    verification = _load(VERIFY_PATH)
    v5 = _load(V5_EVIDENCE_PATH)
    with V5_GENERATOR_PATH.open(newline="", encoding="utf-8") as stream:
        v5_generators = {row["source_id"]: row for row in csv.DictReader(stream)}
    _audit(config, gpu, verification)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    generator_rows, generator_summary = _generator_rows(gpu, v5_generators)
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

    round_rows: list[dict[str, Any]] = []
    for record in gpu["constraint_generation_rounds"]:
        solve = record["solve"]
        statistics = solve["statistics"]
        screen = record["screen"]
        round_rows.append(
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
                "screen_new_violated_pairs": screen["new_violated_pairs"],
                "screen_maximum_violation_pu": screen["maximum_violation_pu"],
                "pdlp_method_requested": statistics[
                    "cuopt_pdlp_parameters_requested"
                ]["method"],
                "pdlp_method_readback": statistics[
                    "cuopt_pdlp_parameters_readback"
                ]["method"],
                "pdlp_mode_requested": statistics[
                    "cuopt_pdlp_parameters_requested"
                ]["pdlp_solver_mode"],
                "pdlp_mode_readback": statistics[
                    "cuopt_pdlp_parameters_readback"
                ]["pdlp_solver_mode"],
                "pdlp_precision_requested": statistics[
                    "cuopt_pdlp_parameters_requested"
                ]["pdlp_precision"],
                "pdlp_precision_readback": statistics[
                    "cuopt_pdlp_parameters_readback"
                ]["pdlp_precision"],
                "batch_pdlp_strong_branching_readback": statistics[
                    "cuopt_pdlp_parameters_readback"
                ]["mip_batch_pdlp_strong_branching"],
                "batch_pdlp_reliability_branching_readback": statistics[
                    "cuopt_pdlp_parameters_readback"
                ]["mip_batch_pdlp_reliability_branching"],
                "mip_reliability_branching_readback": statistics[
                    "cuopt_pdlp_parameters_readback"
                ]["mip_reliability_branching"],
            }
        )
    _write_csv(REPORT_ROOT / "round-detail.csv", round_rows)

    comparison = {
        "v5_objective": v5["objective"],
        "v6_objective": gpu["objective"],
        "v6_minus_v5_objective": float(gpu["objective"] - v5["objective"]),
        "v5_bound": v5["bound"],
        "v6_bound": gpu["bound"],
        "v6_minus_v5_bound": float(gpu["bound"] - v5["bound"]),
        "v5_mip_gap": v5["mip_gap"],
        "v6_mip_gap": gpu["mip_gap"],
        "v6_minus_v5_mip_gap": float(gpu["mip_gap"] - v5["mip_gap"]),
        "v5_solver_rounds_seconds": v5["timings_seconds"]["solver_rounds"],
        "v6_solver_rounds_seconds": gpu["timings_seconds"]["solver_rounds"],
        "v5_wall_seconds": v5["total_wall_time_seconds"],
        "v6_wall_seconds": gpu["total_wall_time_seconds"],
        "v5_added_security_pairs": v5["added_security_pairs"],
        "v6_added_security_pairs": len(gpu["added_security_pair_ids"]),
        **generator_summary,
    }
    _write_csv(REPORT_ROOT / "comparison-v5.csv", [comparison])

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
        "v5_evidence": _digest(V5_EVIDENCE_PATH),
    }
    summary = {
        "schema_version": "1.0.0",
        "campaign": "activsg2000-gpu-gap-sensitivity-v6",
        "status": gpu["status"],
        "success": False,
        "failure_gate": "requested_mip_gap_not_certified",
        "initialization": config["benchmark"]["initialization"],
        "observed_mip_start_status": (
            "submitted_by_adapter_but_rejected_by_cuopt_in_rounds_2_and_3"
        ),
        "solver_allowance_seconds": 900.0,
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
        "comparison_to_v5": comparison,
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
                "solver_rounds_seconds": gpu["timings_seconds"]["solver_rounds"],
                "wall_seconds": gpu["total_wall_time_seconds"],
            }
        ],
    )

    first, second, third = round_rows
    readme = f"""# ACTIVSg2000 DGX Spark PDLP-policy v6 result

The one authorized v6 run ended
`incomplete_restricted_master_gap_not_certified` after
{gpu['total_wall_time_seconds']:.3f} seconds. It is not a successful SCOPF
result, but it is not infeasible. The final incumbent is independently secure.
Round 1 started cold; the adapter submitted the prior GPU solution's
{second['partial_integer_mip_start_columns']} commitment columns in rounds 2
and 3. Retrospective native-log inspection found that cuOpt rejected both
starts after internal model expansion. No CPU commitment, dispatch, or lower
bound initialized the run, and the rejected starts are not counted as reuse.

Relative to v5, v6 changed only the registered cuOpt policy and frozen identity.
Every round selected method 1 (PDLP), Stable3 mode 4, FP64 precision 1, batched
PDLP strong branching, batched PDLP reliability branching, and reliability
factor 1. Requested and read-back values matched exactly. The native log
confirmed `Cooperative batch PDLP and Dual Simplex for strong branching` in all
three rounds. This evidence means PDLP assisted the MIP branching work; it does
not mean every branch-and-bound relaxation used PDLP alone.

The dynamic loop added {first['screen_new_violated_pairs']} contingency pairs
after round 1 and {second['screen_new_violated_pairs']} after round 2. Round 3's
exhaustive screen found zero violations above `1e-5` p.u. The independent raw-
input checker passed all {verification['checked_security_sides']:,} sides with
maximum security violation {verification['maximum_security_violation_pu']:.3e}
p.u. and maximum model residual
{verification['maximum_model_residual_pu']:.3e} p.u.

The sole failed acceptance gate is the requested MIP gap. Round 3 returned
objective {gpu['objective']:,.6f}, finite bound {gpu['bound']:,.6f}, and gap
{gpu['mip_gap']:.7f}, above `0.001`. It reached its time limit after
{third['native_solve_seconds']:.3f} native seconds. Total restricted-master
time was {gpu['timings_seconds']['solver_rounds']:.3f} seconds inside the
900-second cumulative allowance; end-to-end wall time stayed below 1,035
seconds. Fixed-commitment pricing is therefore intentionally withheld.

Compared with v5, v6 improved the final objective by
{-comparison['v6_minus_v5_objective']:.6f}, improved the bound by
{comparison['v6_minus_v5_bound']:.6f}, and reduced the reported relative gap
from {comparison['v5_mip_gap']:.7f} to {comparison['v6_mip_gap']:.7f}. That is
still insufficient for the requested certificate. Both solutions commit
{generator_summary['commitment_count']} units and serve
{generator_summary['dispatch_total_mw']:,.6f} MW, but
{generator_summary['commitment_flips_from_v5']} commitment decisions differ
({generator_summary['v5_off_v6_on']} off-to-on and
{generator_summary['v5_on_v6_off']} on-to-off).

`generator-detail.csv` preserves all 544 exact source PMIN/PMAX rows, both v5
and v6 secure-but-gap-uncertified commitments/dispatches, MW and p.u. values,
and blank price fields with the withholding reason. `bus-prices.csv` preserves
all 2,000 bus identities with blank prices. The other files retain parameter
readbacks, round timing, event/native logs, hashes, verification, and the v5
comparison. No v6 retry was performed.
"""
    (REPORT_ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
