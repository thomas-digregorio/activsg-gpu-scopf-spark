"""Build tracked evidence for the 15-minute ACTIVSg2000 Spark v5 run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "results" / "experiments"
DIAGNOSTIC_ROOT = ROOT / "results" / "diagnostics"
CONFIG_PATH = ROOT / "configs" / "activsg2000-gpu-gap-1e-3-v5.json"
GPU_PATH = EXPERIMENT_ROOT / "activsg2000-gpu-gap-v5-1e-3-dgx-spark.json"
REGISTRY_PATH = (
    EXPERIMENT_ROOT / "activsg2000-gpu-gap-sensitivity-v5-run-registry.json"
)
LAUNCHER_CONSOLE_PATH = (
    EXPERIMENT_ROOT / "activsg2000-gpu-gap-v5-1e-3-cuopt-console.log"
)
EVENT_PATH = (
    DIAGNOSTIC_ROOT
    / "activsg2000-gpu-gap-v5-1e-3-dgx_spark-events.jsonl"
)
WORKER_CONSOLE_PATH = (
    DIAGNOSTIC_ROOT
    / "activsg2000-gpu-gap-v5-1e-3-dgx_spark-worker-console.log"
)
VERIFY_PATH = (
    DIAGNOSTIC_ROOT
    / "activsg2000-gpu-gap-v5-1e-3-independent-verification.json"
)
REPORT_ROOT = ROOT / "reports" / "activsg2000-gpu-1e-3-v5"
GPU_COMMIT = "c3ee4300330c614c3a30eb47180a267c28988129"


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
        raise ValueError("Unexpected GPU v5 status")
    identity = gpu["frozen_identity"]
    if identity["commit"] != GPU_COMMIT:
        raise ValueError("GPU v5 commit changed")
    if identity["tag"] != "experiment-2000-gpu-gap-v5":
        raise ValueError("GPU v5 tag changed")
    if identity["config_sha256"] != config["_sha256"]:
        raise ValueError("GPU v5 config digest changed")
    initialization = config["benchmark"]["initialization"]
    if initialization != {
        "external_cpu_mip_start": False,
        "round_1": "cold",
        "later_rounds": "prior_gpu_integer_commitment_only",
    }:
        raise ValueError("GPU v5 initialization contract changed")
    runtime = config["runtime"]
    available = (
        float(runtime["deadline_seconds"])
        - float(runtime["verification_reserve_seconds"])
        - float(runtime["serialization_reserve_seconds"])
    )
    if available != 900.0:
        raise ValueError("GPU v5 no longer preserves a 900-second solver allowance")
    rounds = gpu.get("constraint_generation_rounds", [])
    if len(rounds) != 3:
        raise ValueError("GPU v5 must preserve exactly three rounds")
    additions = [len(record.get("added_pair_ids") or []) for record in rounds]
    if additions != [349, 14, 0]:
        raise ValueError(f"GPU v5 security-pair additions changed: {additions}")
    starts = [
        record["solve"]["statistics"]["partial_integer_mip_start_columns"]
        for record in rounds
    ]
    if starts != [0, 432, 432]:
        raise ValueError(f"GPU v5 round starts changed: {starts}")
    if any(
        record["solve"]["statistics"]["mip_start_mode"] != "integer_only"
        for record in rounds
    ):
        raise ValueError("GPU v5 used a non-integer MIP start")
    if rounds[2]["solve"].get("requested_gap_certified") is not False:
        raise ValueError("GPU v5 round 3 unexpectedly certifies the requested gap")
    if rounds[2]["screen"].get("new_violated_pairs") != 0:
        raise ValueError("GPU v5 round 3 did not preserve the zero-violation screen")
    if float(gpu["timings_seconds"]["solver_rounds"]) > 900.0:
        raise ValueError("GPU v5 exceeded the cumulative 900-second solver allowance")
    if verification.get("passed") is not True:
        raise ValueError("Independent verification of the final incumbent did not pass")
    if gpu.get("pricing") is not None:
        raise ValueError("Gap-uncertified GPU result must not contain accepted pricing")


def main() -> None:
    config = _load(CONFIG_PATH)
    gpu = _load(GPU_PATH)
    verification = _load(VERIFY_PATH)
    _audit(config, gpu, verification)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    source = gpu["source_manifest"]["generators"]
    generators = gpu["solution"]["generators"]
    if len(source) != 544 or len(generators) != 544:
        raise ValueError("Expected all 544 generator source rows")
    base_mva = 100.0
    generator_rows: list[dict[str, Any]] = []
    commitment_count = 0
    dispatch_total_mw = 0.0
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
        commitment_count += commitment
        dispatch_total_mw += dispatch
        if not (
            float(source_row["pmin_mw"]) * commitment - 1e-6
            <= dispatch
            <= float(source_row["pmax_mw"]) * commitment + 1e-6
        ):
            raise ValueError("Final incumbent violates exact conditional PMIN/PMAX")
        if abs(float(result_row["dispatch_pu"]) - dispatch / base_mva) > 1e-10:
            raise ValueError("Final incumbent dispatch p.u. conversion changed")
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
                "gap_uncertified_secure_commitment": commitment,
                "gap_uncertified_secure_dispatch_mw": result_row["dispatch_mw"],
                "gap_uncertified_secure_dispatch_pu": result_row["dispatch_pu"],
                "mip_gap": gpu["mip_gap"],
                "requested_mip_gap": 1e-3,
                "independent_security_verification": "passed",
                "price_status": "withheld_mip_gap_not_certified",
                "price_per_mwh": None,
                "price_per_pu_hour": None,
            }
        )
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
        screen = record["screen"]
        round_rows.append(
            {
                "round": record["round"],
                "rows_before_solve": record["rows_before_solve"],
                "solver_budget_seconds": record["solver_budget_seconds"],
                "native_status": solve["statistics"]["native_status"],
                "adapter_status": solve["status"],
                "requested_gap_certified": solve["requested_gap_certified"],
                "objective": solve["objective"],
                "bound": solve["bound"],
                "mip_gap": solve["mip_gap"],
                "partial_integer_mip_start_columns": solve["statistics"][
                    "partial_integer_mip_start_columns"
                ],
                "native_solve_seconds": solve["solve_time_seconds"],
                "adapter_wall_seconds": record["adapter_wall_time_seconds"],
                "screen_seconds": screen["wall_time_seconds"],
                "screen_new_violated_pairs": screen["new_violated_pairs"],
                "screen_maximum_violation_pu": screen["maximum_violation_pu"],
            }
        )
    _write_csv(REPORT_ROOT / "round-detail.csv", round_rows)

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
    }
    summary = {
        "schema_version": "1.0.0",
        "campaign": "activsg2000-gpu-gap-sensitivity-v5",
        "status": gpu["status"],
        "success": False,
        "failure_gate": "requested_mip_gap_not_certified",
        "initialization": config["benchmark"]["initialization"],
        "solver_allowance_seconds": 900.0,
        "frozen_identity": gpu["frozen_identity"],
        "source_identity": gpu["source_manifest"]["source_identity"],
        "objective": gpu["objective"],
        "bound": gpu["bound"],
        "absolute_gap": float(gpu["objective"] - gpu["bound"]),
        "mip_gap": gpu["mip_gap"],
        "requested_mip_gap": 1e-3,
        "commitment_count": commitment_count,
        "dispatch_total_mw": dispatch_total_mw,
        "dispatch_total_pu": dispatch_total_mw / base_mva,
        "constraint_generation_rounds": round_rows,
        "added_security_pairs": len(gpu["added_security_pair_ids"]),
        "final_model_dimensions": gpu["final_model_dimensions"],
        "final_screen": gpu["acceptance_gates"],
        "post_run_independent_verification": {
            key: value for key, value in verification.items() if not key.startswith("_")
        },
        "pricing_status": "withheld_mip_gap_not_certified",
        "timings_seconds": gpu["timings_seconds"],
        "total_wall_time_seconds": gpu["total_wall_time_seconds"],
        "peak_memory": gpu["peak_memory"],
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
                "commitment_count": commitment_count,
                "dispatch_total_mw": dispatch_total_mw,
                "pricing_status": "withheld_mip_gap_not_certified",
                "solver_rounds_seconds": gpu["timings_seconds"]["solver_rounds"],
                "wall_seconds": gpu["total_wall_time_seconds"],
            }
        ],
    )

    first, second, third = round_rows
    readme = f"""# ACTIVSg2000 DGX Spark 15-minute dynamic v5 result

The one authorized v5 run ended
`incomplete_restricted_master_gap_not_certified` after
{gpu['total_wall_time_seconds']:.3f} seconds. It is not a successful SCOPF
result, but it is not infeasible. Round 1 started cold. Rounds 2 and 3 each
received the prior GPU solution's {second['partial_integer_mip_start_columns']}
commitment columns as a partial integer MIP start. No CPU commitment, dispatch,
or lower bound initialized this run.

The frozen controller reserved 900 seconds for raw loading and cumulative cuOpt
rounds after retaining 120 seconds for verification and 15 seconds for
serialization. Actual restricted-master solve time was
{gpu['timings_seconds']['solver_rounds']:.3f} seconds and end-to-end wall time
was {gpu['total_wall_time_seconds']:.3f} seconds, below the 1,035-second hard
boundary.

The per-unit cuOpt formulation completed three add-resolve-screen rounds. Round
1 added {first['screen_new_violated_pairs']} contingency pairs; round 2 added
{second['screen_new_violated_pairs']}; round 3's exhaustive screen found zero
violations above `1e-5` p.u. The final incumbent independently passed all
{verification['checked_security_sides']:,} sides with maximum security
violation {verification['maximum_security_violation_pu']:.3e} p.u. and maximum
model residual {verification['maximum_model_residual_pu']:.3e} p.u.

The sole failed acceptance gate is the requested MIP gap. Round 3 returned
objective {gpu['objective']:,.6f}, finite bound {gpu['bound']:,.6f}, and gap
{gpu['mip_gap']:.7f}, above the requested `0.001`. cuOpt reached its time limit
after {third['native_solve_seconds']:.3f} native seconds and
{third['adapter_wall_seconds']:.3f} adapter seconds. Therefore the result
remains incomplete and fixed-commitment prices are intentionally withheld.

`generator-detail.csv` preserves all 544 exact source PMIN/PMAX rows and the
gap-uncertified but independently secure round-3 commitment/dispatch in MW and
p.u. It commits {commitment_count} generators and dispatches
{dispatch_total_mw:,.6f} MW. `bus-prices.csv` preserves all 2,000 bus identities
with blank price fields and the reason prices were withheld. `worker-console.log`
contains the complete native cuOpt stream for all three rounds. The remaining
files preserve round timing, hashes, the event timeline, and independent
verification.

No v5 retry was performed.
"""
    (REPORT_ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
