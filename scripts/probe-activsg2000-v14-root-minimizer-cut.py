#!/usr/bin/env python3
"""Test a GPU root-certificate minimizer with exact Phase I and one global cut.

This is a bounded development probe, not a registered experiment.  It reads
only the preserved GPU v12 certificate, rebuilds the mathematical model from
the immutable raw inputs, and never reads a CPU objective, bound, commitment,
or dispatch.  The purpose is to determine whether a replay-certified Phase-I
cut against the current Lagrangian minimizer can materially strengthen the
GPU-resident lower bound before a full replacement run is authorized.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from activsg_scopf.commitment_cuts import (
    CommitmentFeasibilityCut,
    commitment_upper_cut_from_record,
)
from activsg_scopf.config import load_config
from activsg_scopf.deadline import Deadline
from activsg_scopf.lagrangian import (
    RegionMasks,
    evaluate_lagrangian_bound,
    optimize_commitment_cut_duals_coordinate_cupy,
    replay_lagrangian_certificate,
)
from activsg_scopf.lagrangian_experiment import (
    PrimalCandidatePolicy,
    RegionAttemptRejected,
    _certificate_dual_arrays,
    _prepare_region_master,
    _solve_fixed_commitment_feasibility,
    _validate_prepared_region_master,
)
from activsg_scopf.matpower import GEN_STATUS, read_contingency_table, read_matpower_case
from activsg_scopf.network import build_contingency_catalog, build_network
from activsg_scopf.reduced import security_pair_from_record
from activsg_scopf.screening import ContingencyScreener


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _commitment_sha256(commitment: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(commitment, dtype=np.int8).tobytes()).hexdigest()


def _diagnostic_row_capacity_cut(
    *, master: Any, row_name: str, commitment: np.ndarray, base_mva: float
) -> tuple[CommitmentFeasibilityCut, dict[str, Any]]:
    """Create a conservative necessary commitment condition for one row.

    For an upper row ``a @ p <= b``, every feasible dispatch must satisfy
    ``sum(min(a_g PMIN_g, a_g PMAX_g) u_g) <= b``.  The analogous lower-row
    condition is negated into the common upper-inequality form.  This probe
    keeps the implementation local until its measured bound lift justifies a
    production certificate type and independent replay path.
    """

    coupling = next(row for row in master.coupling_rows if row.row_name == row_name)
    row_lower, row_upper = master.canonical.row_bound_arrays()
    lower = float(row_lower[coupling.row_index])
    upper = float(row_upper[coupling.row_index])
    coefficients = np.asarray(coupling.generator_coefficients, dtype=np.float64)
    curves = [master.costs[int(row)] for row in master.index.generator_source_rows]
    pmin = np.asarray([curve.pmin_mw for curve in curves], dtype=np.float64)
    pmax = np.asarray([curve.pmax_mw for curve in curves], dtype=np.float64)
    minimum_activity = np.minimum(coefficients * pmin, coefficients * pmax)
    maximum_activity = np.maximum(coefficients * pmin, coefficients * pmax)
    candidates: list[tuple[str, np.ndarray, float, float]] = []
    if np.isfinite(upper):
        raw_coefficients = minimum_activity / base_mva
        raw_rhs = upper / base_mva
        candidates.append(
            (
                "upper",
                raw_coefficients,
                raw_rhs,
                float(raw_coefficients @ commitment - raw_rhs),
            )
        )
    if np.isfinite(lower):
        raw_coefficients = -maximum_activity / base_mva
        raw_rhs = -lower / base_mva
        candidates.append(
            (
                "lower",
                raw_coefficients,
                raw_rhs,
                float(raw_coefficients @ commitment - raw_rhs),
            )
        )
    side, cut_coefficients, raw_rhs, raw_violation = max(
        candidates, key=lambda item: (item[3], item[0])
    )
    scale = max(1.0, abs(raw_rhs), float(np.sum(np.abs(cut_coefficients))))
    outward_relaxation = 32.0 * np.finfo(np.float64).eps * scale
    rhs = float(raw_rhs + outward_relaxation)
    cut_coefficients = np.where(cut_coefficients == 0.0, 0.0, cut_coefficients)
    source_violation = float(cut_coefficients @ commitment - rhs)
    if source_violation <= 0.0:
        raise RuntimeError("Constant-row contradiction did not yield a violated capacity cut")
    source_sha = _commitment_sha256(commitment)
    identity = (
        row_name.encode("utf-8")
        + side.encode("ascii")
        + source_sha.encode("ascii")
        + np.asarray([rhs], dtype=np.float64).tobytes()
        + np.asarray(cut_coefficients, dtype=np.float64).tobytes()
    )
    cut = CommitmentFeasibilityCut(
        cut_id="diagnostic_rc_" + hashlib.sha256(identity).hexdigest()[:24],
        coefficients=np.asarray(cut_coefficients, dtype=np.float64),
        rhs=rhs,
        source_commitment_sha256=source_sha,
        conservative_source_violation_pu=source_violation,
    )
    cut.validate(commitment.size)
    return cut, {
        "derivation": "direct_exact_pmin_pmax_row_capacity_necessary_condition_probe_v1",
        "source_row_name": row_name,
        "source_row_side": side,
        "raw_source_violation_pu": raw_violation,
        "conservative_source_violation_pu": source_violation,
        "outward_rhs_relaxation_pu": outward_relaxation,
        "nonzero_coefficient_count": int(np.count_nonzero(cut_coefficients)),
        "cut_id": cut.cut_id,
    }


def main() -> None:
    started = time.perf_counter()
    workspace = Path("/workspace")
    config = load_config(workspace / "configs" / "activsg2000-gpu-lagrangian-v12.json")
    prior = _read_json(
        workspace
        / "results"
        / "experiments"
        / "activsg2000-gpu-lagrangian-v12-dgx-spark.json"
    )
    if prior["frozen_identity"]["tag"] != "experiment-2000-gpu-lagrangian-v12":
        raise RuntimeError("The preserved v12 GPU fixture identity changed")

    case = read_matpower_case(
        config.case_path,
        expected_sha256=config.raw["raw_inputs"]["case_sha256"],
    )
    table = read_contingency_table(
        config.contingency_path,
        expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
    )
    network = build_network(case)
    catalog = build_contingency_catalog(
        case,
        network,
        table,
        validation_columns=int(config.model["lodf_validation_columns"]),
        validation_tolerance_pu=float(config.model["lodf_validation_tolerance_pu"]),
        chunk_columns=int(config.model["lodf_build_chunk_columns"]),
    )
    source_rows = np.flatnonzero(case.gen[:, GEN_STATUS] > 0).astype(np.int64)
    root_record = next(
        record
        for record in prior["solved_region_history"]
        if str(record["region_id"]) == "r"
    )
    root_pairs = tuple(
        security_pair_from_record(
            record,
            catalog,
            lodf_absolute_tolerance=float(config.model["serialized_lodf_replay_tolerance"]),
        )
        for record in root_record["security_pairs"]
    )
    existing_cuts = tuple(
        commitment_upper_cut_from_record(record, source_rows + 1)
        for record in root_record["commitment_upper_cuts"]
    )
    masks = RegionMasks.root(source_rows.size)
    master = _prepare_region_master(
        case=case,
        network=network,
        config=config,
        masks=masks,
        initial_pairs=root_pairs,
        commitment_cuts=existing_cuts,
    )
    _validate_prepared_region_master(master, masks, root_pairs, existing_cuts)
    replayed_root = replay_lagrangian_certificate(
        master,
        root_record["lagrangian_certificate"],
        masks,
        commitment_cuts_by_id={cut.cut_id: cut for cut in existing_cuts},
    )
    root_row_dual, root_cut_dual = _certificate_dual_arrays(
        master, replayed_root, existing_cuts
    )
    minimizer = np.asarray(replayed_root.minimizing_commitment, dtype=np.int8)
    screener = ContingencyScreener(
        network,
        catalog,
        backend="cupy",
        chunk_columns=int(config.model["screen_chunk_columns"]),
    )
    policy = PrimalCandidatePolicy(
        total_seconds=180.0,
        maximum_round_seconds=60.0,
        minimum_round_seconds=1.0,
        stagnation_window_rounds=3,
        minimum_relative_residual_improvement=0.001,
        dual_divergence_multiple=1_000_000.0,
        cold_restart_attempts=1,
    )
    output: dict[str, Any] = {
        "passed": False,
        "development_component_only": True,
        "gpu_only_problem_solution_data": True,
        "cpu_problem_solution_data_used": False,
        "exact_source_pmin_pmax": True,
        "root_certificate_replay_bound": replayed_root.conservative_lower_bound,
        "root_certificate_recorded_bound": float(
            root_record["lagrangian_certificate"]["conservative_lower_bound"]
        ),
        "root_certificate_replay_difference_dollars": abs(
            replayed_root.conservative_lower_bound
            - float(root_record["lagrangian_certificate"]["conservative_lower_bound"])
        ),
        "existing_global_cut_count": len(existing_cuts),
        "root_minimizer_commitment_count": int(np.count_nonzero(minimizer)),
        "root_minimizer_commitment_sha256": _commitment_sha256(minimizer),
        "phase_one_policy": policy.as_dict(),
    }
    phase_started = time.perf_counter()
    try:
        feasible = _solve_fixed_commitment_feasibility(
            region_id="v14_root_lagrangian_minimizer",
            commitment=minimizer,
            case=case,
            network=network,
            catalog=catalog,
            config=config,
            deadline=Deadline(240.0, 15.0, 5.0),
            initial_pairs=root_pairs,
            screener=screener,
            checkpoint=lambda: None,
            progress=None,
            policy=policy,
        )
    except RegionAttemptRejected as exc:
        output["root_minimizer_phase_one"] = {
            "status": "rejected",
            "reason": exc.reason,
            "wall_time_seconds": time.perf_counter() - phase_started,
            "rounds": exc.rounds,
            "replay_certified_global_cut_available": (
                exc.commitment_feasibility_cut is not None
            ),
        }
        new_cut = exc.commitment_feasibility_cut
        direct_capacity_audit: dict[str, Any] | None = None
        if (
            new_cut is None
            and exc.reason == "projected_constant_coupling_row_violation"
            and exc.rounds
        ):
            row_name = str(exc.rounds[-1]["projection_precheck"]["row_name"])
            new_cut, direct_capacity_audit = _diagnostic_row_capacity_cut(
                master=exc.master,
                row_name=row_name,
                commitment=minimizer,
                base_mva=float(case.base_mva),
            )
            output["root_minimizer_phase_one"]["direct_row_capacity_cut"] = (
                direct_capacity_audit
            )
        if new_cut is None:
            output["passed"] = True
            output["component_gate"] = "no_replay_certified_global_cut_available"
        else:
            output["root_minimizer_phase_one"]["global_cut"] = new_cut.as_dict(
                source_rows
            )
            combined_cuts = tuple(sorted(existing_cuts + (new_cut,), key=lambda cut: cut.cut_id))
            strengthened_master = _prepare_region_master(
                case=case,
                network=network,
                config=config,
                masks=masks,
                initial_pairs=tuple(sorted(exc.security_pairs)),
                commitment_cuts=combined_cuts,
            )
            _validate_prepared_region_master(
                strengthened_master,
                masks,
                tuple(sorted(exc.security_pairs)),
                combined_cuts,
            )
            inherited_row_dual, inherited_cut_dual = _certificate_dual_arrays(
                strengthened_master, replayed_root, combined_cuts
            )
            inherited = evaluate_lagrangian_bound(
                strengthened_master,
                inherited_row_dual,
                masks,
                safety_margin_dollars=float(replayed_root.safety_margin_dollars),
                commitment_cuts=combined_cuts,
                commitment_cut_dual=inherited_cut_dual,
            )
            coordinate_started = time.perf_counter()
            optimized_cut_dual, audit = optimize_commitment_cut_duals_coordinate_cupy(
                strengthened_master,
                inherited_row_dual,
                masks,
                commitment_cuts=combined_cuts,
                initial_commitment_cut_dual=inherited_cut_dual,
                cycles=64,
            )
            coordinate_wall = time.perf_counter() - coordinate_started
            strengthened = evaluate_lagrangian_bound(
                strengthened_master,
                inherited_row_dual,
                masks,
                safety_margin_dollars=float(replayed_root.safety_margin_dollars),
                commitment_cuts=combined_cuts,
                commitment_cut_dual=optimized_cut_dual,
            )
            replay_difference = abs(
                float(audit["best_raw_lower_bound"]) - strengthened.raw_lower_bound
            )
            if replay_difference > float(
                config.raw["benchmark"]["gpu_cpu_replay_tolerance_dollars"]
            ):
                raise RuntimeError(
                    "GPU cut coordinate certificate failed exact replay"
                ) from exc
            if strengthened.conservative_lower_bound + 1e-6 < inherited.conservative_lower_bound:
                raise RuntimeError("GPU cut coordinate ascent weakened the certificate") from exc
            required_bound = float(prior["objective"]) * (
                1.0 - float(config.model["mip_relative_gap_tolerance"])
            )
            required_lift = max(
                0.0, required_bound - inherited.conservative_lower_bound
            )
            achieved_lift = max(
                0.0,
                strengthened.conservative_lower_bound
                - inherited.conservative_lower_bound,
            )
            output["cut_coordinate_ascent"] = {
                "cycles": 64,
                "wall_time_seconds": coordinate_wall,
                "inherited_bound": inherited.conservative_lower_bound,
                "strengthened_bound": strengthened.conservative_lower_bound,
                "improvement_dollars": achieved_lift,
                "required_bound_for_gpu_incumbent_at_requested_gap": required_bound,
                "required_lift_dollars": required_lift,
                "achieved_fraction_of_required_lift": (
                    1.0 if required_lift == 0.0 else achieved_lift / required_lift
                ),
                "nonzero_cut_dual_count": int(np.count_nonzero(optimized_cut_dual)),
                "new_cut_dual": float(
                    optimized_cut_dual[
                        next(
                            index
                            for index, cut in enumerate(combined_cuts)
                            if cut.cut_id == new_cut.cut_id
                        )
                    ]
                ),
                "new_minimizer_commitment_count": int(
                    np.count_nonzero(strengthened.minimizing_commitment)
                ),
                "new_minimizer_commitment_sha256": _commitment_sha256(
                    strengthened.minimizing_commitment
                ),
                "minimizer_changed": bool(
                    np.any(strengthened.minimizing_commitment != minimizer)
                ),
                "cpu_replay_difference_dollars": replay_difference,
                "host_transfer_during_coordinates": audit[
                    "host_transfer_during_coordinates"
                ],
            }
            output["passed"] = True
            output["component_gate"] = (
                "material" if required_lift == 0.0 or achieved_lift >= 0.01 * required_lift
                else "insufficient_bound_lift"
            )
    else:
        output["root_minimizer_phase_one"] = {
            "status": "secure",
            "wall_time_seconds": time.perf_counter() - phase_started,
            "round_count": len(feasible.rounds),
            "final_screen": feasible.final_screen,
            "canonical_model_residual_pu": (
                feasible.master.canonical.max_row_violation(feasible.source_values)
                / float(case.base_mva)
            ),
        }
        output["passed"] = True
        output["component_gate"] = "root_lagrangian_minimizer_is_secure"

    output["total_wall_time_seconds"] = time.perf_counter() - started
    print("V14_ROOT_MINIMIZER_CUT_PROBE=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
