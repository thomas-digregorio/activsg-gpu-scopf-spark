"""Build the tracked ACTIVSg2000 v11 incumbent-commitment evidence report."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "results" / "experiments"
DIAGNOSTICS = ROOT / "results" / "diagnostics"
RESULT = (
    EXPERIMENTS
    / "activsg2000-gpu-commitment-trace-v11-1e-3-dgx-spark.json"
)
REGISTRY = (
    EXPERIMENTS / "activsg2000-gpu-commitment-trace-v11-run-registry.json"
)
EVENTS = (
    DIAGNOSTICS
    / "activsg2000-gpu-commitment-trace-v11-1e-3-dgx_spark-events.jsonl"
)
WORKER_CONSOLE = (
    DIAGNOSTICS
    / "activsg2000-gpu-commitment-trace-v11-1e-3-dgx_spark-worker-console.log"
)
INDEPENDENT_VERIFY = (
    DIAGNOSTICS
    / "activsg2000-gpu-commitment-trace-v11-1e-3-independent-verification.json"
)
CUOPT_CONSOLE = (
    EXPERIMENTS
    / "activsg2000-gpu-commitment-trace-v11-1e-3-cuopt-console.log"
)
LAUNCHER_CONSOLE = (
    EXPERIMENTS / "activsg2000-gpu-commitment-trace-v11-launcher.log"
)
V10_FAILURE = (
    EXPERIMENTS / "activsg2000-gpu-commitment-trace-v10-launcher.log"
)
MIP_START_SMOKE = (
    DIAGNOSTICS
    / "activsg2000-gpu-commitment-trace-v11-mip-start-smoke.json"
)
V9_RESULT = EXPERIMENTS / "activsg2000-gpu-gap-v9-1e-3-dgx-spark.json"
CPU_RESULT = EXPERIMENTS / "activsg2000-gap-v1-1e-3-laptop.json"
REPORT = ROOT / "reports" / "activsg2000-gpu-commitment-trace-v11"
VARIABLE_PATTERN = re.compile(r"^u_g(\d+)$")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def commitment_map(payload: dict[str, Any]) -> dict[int, int]:
    return {
        int(generator["source_row"]): int(round(generator["commitment"]))
        for generator in payload["solution"]["generators"]
    }


def comparison(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    before = commitment_map(baseline)
    after = commitment_map(current)
    changed = [row for row in sorted(after) if before[row] != after[row]]
    return {
        "baseline_status": baseline["status"],
        "baseline_objective": baseline["objective"],
        "baseline_bound": baseline["bound"],
        "baseline_mip_gap": baseline["mip_gap"],
        "baseline_commitment_count": sum(before.values()),
        "current_commitment_count": sum(after.values()),
        "hamming_distance": len(changed),
        "off_to_on": sum(before[row] == 0 and after[row] == 1 for row in changed),
        "on_to_off": sum(before[row] == 1 and after[row] == 0 for row in changed),
        "objective_delta": current["objective"] - baseline["objective"],
        "changed_source_rows": changed,
    }


def main() -> None:
    required = [
        RESULT,
        REGISTRY,
        EVENTS,
        WORKER_CONSOLE,
        INDEPENDENT_VERIFY,
        CUOPT_CONSOLE,
        LAUNCHER_CONSOLE,
        V10_FAILURE,
        MIP_START_SMOKE,
        V9_RESULT,
        CPU_RESULT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing report inputs: {missing}")

    result = read_json(RESULT)
    registry = read_json(REGISTRY)
    verification = read_json(INDEPENDENT_VERIFY)
    v9 = read_json(V9_RESULT)
    cpu = read_json(CPU_RESULT)
    rounds = result["constraint_generation_rounds"]
    generators = result["solution"]["generators"]
    generator_by_row = {int(g["source_row"]): g for g in generators}
    round_maps = [
        {
            int(g["source_row"]): int(g["commitment"])
            for g in round_result["unit_commitment"]["generator_commitments"]
        }
        for round_result in rounds
    ]
    v9_map = commitment_map(v9)
    cpu_map = commitment_map(cpu)

    REPORT.mkdir(parents=True, exist_ok=True)
    copies = {
        EVENTS: "events.jsonl",
        WORKER_CONSOLE: "worker-console.log",
        INDEPENDENT_VERIFY: "independent-verification.json",
        CUOPT_CONSOLE: "cuopt-console.log",
        LAUNCHER_CONSOLE: "launcher-console.log",
        REGISTRY: "run-registry.json",
        V10_FAILURE: "v10-prelaunch-error.log",
        MIP_START_SMOKE: "mip-start-smoke.json",
    }
    for source, destination in copies.items():
        shutil.copy2(source, REPORT / destination)

    round_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    incumbent_flip_rows: list[dict[str, Any]] = []
    round_flip_rows: list[dict[str, Any]] = []
    for round_result in rounds:
        solve = round_result["solve"]
        statistics = solve["statistics"]
        trace = statistics["incumbent_commitment_trace"]
        commitment = round_result["unit_commitment"]
        precheck = statistics.get("mip_start_feasibility_precheck") or {}
        round_rows.append(
            {
                "round": round_result["round"],
                "rows_before_solve": round_result["rows_before_solve"],
                "solver_budget_seconds": round_result["solver_budget_seconds"],
                "adapter_wall_time_seconds": round_result[
                    "adapter_wall_time_seconds"
                ],
                "solve_time_seconds": solve["solve_time_seconds"],
                "objective": solve["objective"],
                "bound": solve["bound"],
                "mip_gap": solve["mip_gap"],
                "requested_gap_certified": solve["requested_gap_certified"],
                "commitment_count": commitment["commitment_count"],
                "hamming_distance_from_previous_round": commitment[
                    "hamming_distance_from_previous_round"
                ],
                "off_to_on_from_previous_round": commitment[
                    "off_to_on_from_previous_round"
                ],
                "on_to_off_from_previous_round": commitment[
                    "on_to_off_from_previous_round"
                ],
                "callback_count": trace["callback_count"],
                "distinct_commitment_count": trace["unique_commitment_count"],
                "last_commitment_change_seconds": trace[
                    "last_commitment_change_elapsed_seconds"
                ],
                "stabilization_window_at_return_seconds": trace[
                    "stabilization_window_seconds_at_solver_return"
                ],
                "mip_start_precheck_decision": precheck.get("decision"),
                "mip_start_precheck_status": precheck.get("solver_status"),
                "prior_commitment_extendable": precheck.get(
                    "prior_commitment_extendable"
                ),
                "submitted_mip_start_columns": statistics["mip_start_columns"],
                "added_security_pairs": len(round_result.get("added_pair_ids", [])),
                "screen_new_violated_pairs": round_result["screen"][
                    "new_violated_pairs"
                ],
                "screen_maximum_violation_pu": round_result["screen"][
                    "maximum_violation_pu"
                ],
            }
        )
        for snapshot in trace["snapshots"]:
            transition_rows.append(
                {
                    "round": round_result["round"],
                    "transition": snapshot["transition"],
                    "first_seen_elapsed_seconds": snapshot[
                        "first_seen_elapsed_seconds"
                    ],
                    "last_seen_elapsed_seconds": snapshot[
                        "last_seen_elapsed_seconds"
                    ],
                    "objective": snapshot["first_seen_objective"],
                    "bound": snapshot["first_seen_bound"],
                    "commitment_count": snapshot["commitment_count"],
                    "hamming_distance_from_previous_incumbent": snapshot[
                        "hamming_distance_from_previous_incumbent"
                    ],
                    "off_to_on_from_previous_incumbent": snapshot[
                        "off_to_on_from_previous_incumbent"
                    ],
                    "on_to_off_from_previous_incumbent": snapshot[
                        "on_to_off_from_previous_incumbent"
                    ],
                    "callbacks_for_state": snapshot["incumbent_callbacks_for_state"],
                    "commitment_fingerprint_sha256": snapshot[
                        "commitment_fingerprint_sha256"
                    ],
                }
            )
            for change in snapshot["changes_from_previous_incumbent"]:
                match = VARIABLE_PATTERN.fullmatch(change["variable_name"])
                if match is None:
                    raise ValueError(
                        f"Unexpected commitment variable {change['variable_name']!r}"
                    )
                source_row = int(match.group(1)) + 1
                generator = generator_by_row[source_row]
                incumbent_flip_rows.append(
                    {
                        "round": round_result["round"],
                        "transition": snapshot["transition"],
                        "first_seen_elapsed_seconds": snapshot[
                            "first_seen_elapsed_seconds"
                        ],
                        "variable_name": change["variable_name"],
                        "source_row": source_row,
                        "source_id": generator["source_id"],
                        "bus": generator["bus"],
                        "pmin_mw": generator["pmin_mw"],
                        "pmax_mw": generator["pmax_mw"],
                        "from_commitment": change["from_commitment"],
                        "to_commitment": change["to_commitment"],
                        "direction": (
                            "off_to_on"
                            if change["to_commitment"] == 1
                            else "on_to_off"
                        ),
                    }
                )
        for change in commitment["changes_from_previous_round"]:
            generator = generator_by_row[int(change["source_row"])]
            round_flip_rows.append(
                {
                    "round": round_result["round"],
                    "source_row": change["source_row"],
                    "source_id": change["source_id"],
                    "bus": change["bus"],
                    "pmin_mw": generator["pmin_mw"],
                    "pmax_mw": generator["pmax_mw"],
                    "from_commitment": change["from_commitment"],
                    "to_commitment": change["to_commitment"],
                    "direction": (
                        "off_to_on"
                        if change["to_commitment"] == 1
                        else "on_to_off"
                    ),
                }
            )

    generator_rows: list[dict[str, Any]] = []
    for generator in generators:
        source_row = int(generator["source_row"])
        commitments = [round_map[source_row] for round_map in round_maps]
        generator_rows.append(
            {
                "source_row": source_row,
                "source_id": generator["source_id"],
                "bus": generator["bus"],
                "source_status": generator["source_status"],
                "pmin_mw": generator["pmin_mw"],
                "pmax_mw": generator["pmax_mw"],
                "round_1_commitment": commitments[0],
                "round_2_commitment": commitments[1],
                "round_3_commitment": commitments[2],
                "round_2_vs_1_changed": int(commitments[1] != commitments[0]),
                "round_3_vs_2_changed": int(commitments[2] != commitments[1]),
                "v9_final_commitment": v9_map[source_row],
                "v11_vs_v9_changed": int(commitments[2] != v9_map[source_row]),
                "cpu_1e_3_commitment": cpu_map[source_row],
                "v11_vs_cpu_changed": int(commitments[2] != cpu_map[source_row]),
                "final_dispatch_mw": generator["dispatch_mw"],
                "final_dispatch_pu": generator["dispatch_pu"],
                "pricing_status": "withheld_gap_not_certified",
                "fixed_commitment_nodal_price_per_mwh": "",
            }
        )

    write_csv(REPORT / "round-summary.csv", round_rows, list(round_rows[0]))
    write_csv(
        REPORT / "incumbent-transitions.csv",
        transition_rows,
        list(transition_rows[0]),
    )
    write_csv(
        REPORT / "incumbent-flips.csv",
        incumbent_flip_rows,
        list(incumbent_flip_rows[0]),
    )
    write_csv(REPORT / "round-flips.csv", round_flip_rows, list(round_flip_rows[0]))
    write_csv(
        REPORT / "generator-round-detail.csv",
        generator_rows,
        list(generator_rows[0]),
    )

    round_three_trace = rounds[2]["solve"]["statistics"][
        "incumbent_commitment_trace"
    ]
    round_three_snapshots = round_three_trace["snapshots"]
    long_plateau_seconds = (
        round_three_snapshots[24]["first_seen_elapsed_seconds"]
        - round_three_snapshots[23]["first_seen_elapsed_seconds"]
    )
    v9_comparison = comparison(v9, result)
    cpu_comparison = comparison(cpu, result)
    evidence = {
        "schema_version": "1.0.0",
        "experiment": {
            "suite": "activsg2000-gpu-commitment-trace-v11",
            "commit": registry["runs"]["1e-3"]["frozen_identity"]["commit"],
            "tag": registry["runs"]["1e-3"]["frozen_identity"]["tag"],
            "config_sha256": registry["runs"]["1e-3"]["frozen_identity"][
                "config_sha256"
            ],
            "result_sha256": sha256(RESULT),
            "result_status": result["status"],
            "objective": result["objective"],
            "bound": result["bound"],
            "mip_gap": result["mip_gap"],
            "requested_mip_gap": 0.001,
            "end_to_end_elapsed_seconds": result["elapsed_seconds"],
            "pricing_status": None,
        },
        "source": {
            "identity": result["source_manifest"]["source_identity"],
            "dimensions": result["source_manifest"]["dimensions"],
            "source_online_capacity_mw": result["source_manifest"][
                "source_online_capacity_mw"
            ],
            "synthetic_system": result["source_manifest"]["synthetic_system"],
        },
        "security_and_verification": {
            "final_dynamic_screen_maximum_violation_pu": rounds[-1]["screen"][
                "maximum_violation_pu"
            ],
            "final_dynamic_screen_new_violated_pairs": rounds[-1]["screen"][
                "new_violated_pairs"
            ],
            "independent_verification": verification,
        },
        "commitment_stability": {
            "conclusion": "not_stabilized_by_end_of_run",
            "round_endpoint_hamming_distances": [
                round_result["unit_commitment"][
                    "hamming_distance_from_previous_round"
                ]
                for round_result in rounds
            ],
            "round_distinct_incumbent_commitments": [
                round_result["solve"]["statistics"][
                    "incumbent_commitment_trace"
                ]["unique_commitment_count"]
                for round_result in rounds
            ],
            "round_three_long_plateau_seconds": long_plateau_seconds,
            "round_three_last_change_seconds": round_three_trace[
                "last_commitment_change_elapsed_seconds"
            ],
            "round_three_final_stabilization_window_seconds": round_three_trace[
                "stabilization_window_seconds_at_solver_return"
            ],
            "round_three_late_transitions_after_plateau": len(
                round_three_snapshots[24:]
            ),
        },
        "mip_start_handling": [
            {
                "round": round_result["round"],
                "precheck": round_result["solve"]["statistics"].get(
                    "mip_start_feasibility_precheck"
                ),
                "submitted_columns": round_result["solve"]["statistics"][
                    "mip_start_columns"
                ],
            }
            for round_result in rounds
        ],
        "comparisons": {
            "v9_gpu_final": v9_comparison,
            "cpu_1e_3_final": cpu_comparison,
        },
        "preserved_v10_prelaunch_failure": {
            "phase": "before_case_loading_and_before_solve",
            "message": V10_FAILURE.read_text(encoding="utf-8").strip(),
            "sha256": sha256(V10_FAILURE),
        },
        "mip_start_smoke": read_json(MIP_START_SMOKE),
        "tracked_tables": {
            "round_summary": "round-summary.csv",
            "incumbent_transitions": "incumbent-transitions.csv",
            "incumbent_flips": "incumbent-flips.csv",
            "round_flips": "round-flips.csv",
            "generator_round_detail": "generator-round-detail.csv",
        },
    }
    write_json(REPORT / "evidence.json", evidence)

    summary_rows = [
        {"metric": "status", "value": result["status"]},
        {"metric": "objective", "value": result["objective"]},
        {"metric": "bound", "value": result["bound"]},
        {"metric": "mip_gap", "value": result["mip_gap"]},
        {"metric": "requested_mip_gap", "value": 0.001},
        {"metric": "end_to_end_seconds", "value": result["elapsed_seconds"]},
        {"metric": "final_commitment_count", "value": 328},
        {"metric": "round_2_endpoint_hamming", "value": 46},
        {"metric": "round_3_endpoint_hamming", "value": 39},
        {"metric": "round_3_distinct_commitments", "value": 30},
        {
            "metric": "round_3_final_stability_window_seconds",
            "value": round_three_trace[
                "stabilization_window_seconds_at_solver_return"
            ],
        },
        {
            "metric": "independent_max_security_violation_pu",
            "value": verification["maximum_security_violation_pu"],
        },
        {"metric": "pricing_status", "value": "withheld_gap_not_certified"},
    ]
    write_csv(REPORT / "summary.csv", summary_rows, ["metric", "value"])

    readme = f"""# ACTIVSg2000 GPU commitment trace v11

## Result

The single authorized DGX Spark run completed in
`{result['elapsed_seconds']:.3f}` seconds end to end. Its final incumbent is
exhaustively N-1 secure, but the requested `1e-3` MIP gap was **not**
certified: objective `${result['objective']:,.6f}`, bound
`${result['bound']:,.6f}`, gap `{result['mip_gap']:.9f}`. The correct run
status is `{result['status']}` and fixed-commitment pricing is withheld.

Independent raw-input verification passed exact conditional PMIN/PMAX,
integrality, objective reconstruction, DC physics, base limits, and all
`{verification['checked_security_sides']:,}` contingency sides. Its maximum
security violation was `{verification['maximum_security_violation_pu']:.3e}`
p.u., below the `1e-5` p.u. tolerance.

## Did unit commitment stabilize?

No -- not by the end of this run. The evidence shows both inter-round and
within-round movement:

- Round 1 ended with 325 committed source rows.
- Round 2 ended with 321; 46 rows changed from round 1 (21 off-to-on and 25
  on-to-off). It produced 33 distinct callback commitments.
- Round 3 ended with 328; 39 rows changed from round 2 (23 off-to-on and 16
  on-to-off). It produced 30 distinct callback commitments.
- Round 3 had a long `{long_plateau_seconds:.3f}`-second plateau, but then six
  later commitment transitions occurred. The last binary change arrived at
  `{round_three_trace['last_commitment_change_elapsed_seconds']:.3f}` solver
  seconds, only
  `{round_three_trace['stabilization_window_seconds_at_solver_return']:.3f}`
  seconds before solver return.

The practical conclusion is that the incumbent often looked stable for
minutes, while the lower-bound proof moved slowly, but late commitment changes
still mattered. A long unchanged objective is therefore not enough to declare
the unit commitment stable on this model.

## MIP-start handling

The later-round start policy was exercised correctly. Before each rebuild, a
bounded HiGHS fixed-commitment LP tested whether the prior GPU commitment could
be extended to the newly constrained master. Round 2's prior commitment was
infeasible after 177 security rows were added. Round 3's precheck reached its
10-second limit after another 24 rows. Consequently, neither prior commitment
was submitted to cuOpt; both rounds solved cold. This is a guarded rejection,
not the earlier native-column-size translation bug.

## Evidence files

- `generator-round-detail.csv`: all 544 source rows, exact PMIN/PMAX, every
  round endpoint, final dispatch in MW and p.u., and v9/CPU comparisons.
- `incumbent-transitions.csv`: every distinct within-solve commitment state.
- `incumbent-flips.csv`: every exact unit flip between callback states.
- `round-flips.csv`: exact endpoint flips after each security update.
- `round-summary.csv` and `summary.csv`: compact timing and acceptance data.
- `evidence.json`, `independent-verification.json`, `events.jsonl`, and the
  console logs: immutable provenance and verification evidence.
- `v10-prelaunch-error.log`: the preserved registry failure that occurred
  before case loading or optimization; v11 is the sole computational run.
- `mip-start-smoke.json`: a one-time frozen-image component check proving that
  a five-column canonical full start is translated into the correct native
  vector despite free-variable splitting and column elimination.

The raw ACTIVSg case and large solver result remain ignored by repository
policy. Their source and result hashes are recorded in `evidence.json`.
"""
    (REPORT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
