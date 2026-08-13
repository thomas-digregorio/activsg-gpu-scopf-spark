"""Independent raw-input checker, including every eligible branch outage."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import RunConfig, load_config
from .costs import build_pwl_costs
from .errors import ProvenanceError
from .matpower import (
    ANGMAX,
    ANGMIN,
    GEN_BUS,
    GEN_STATUS,
    GS,
    PD,
    PMAX,
    PMIN,
    read_contingency_table,
    read_matpower_case,
)
from .network import build_contingency_catalog, build_network, solve_dc
from .solution import base_flow_vector


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    elapsed_seconds: float
    maximum_model_residual_pu: float
    maximum_security_violation_pu: float
    maximum_integrality_violation: float
    maximum_angle_limit_violation_rad: float
    objective_recalculated: float
    objective_difference: float
    checked_valid_outages: int
    checked_security_sides: int
    details: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "elapsed_seconds": self.elapsed_seconds,
            "maximum_model_residual_pu": self.maximum_model_residual_pu,
            "maximum_security_violation_pu": self.maximum_security_violation_pu,
            "maximum_integrality_violation": self.maximum_integrality_violation,
            "maximum_angle_limit_violation_rad": self.maximum_angle_limit_violation_rad,
            "objective_recalculated": self.objective_recalculated,
            "objective_difference": self.objective_difference,
            "checked_valid_outages": self.checked_valid_outages,
            "checked_security_sides": self.checked_security_sides,
            "details": self.details,
        }


def verify_serialized_solution(
    config_or_path: RunConfig | str,
    payload: dict[str, Any],
) -> VerificationResult:
    """Reread immutable raw inputs and verify without trusting the master model."""

    started = time.perf_counter()
    config = (
        config_or_path
        if isinstance(config_or_path, RunConfig)
        else load_config(config_or_path)
    )
    case = read_matpower_case(
        config.case_path, expected_sha256=config.raw["raw_inputs"]["case_sha256"]
    )
    contingency_table = read_contingency_table(
        config.contingency_path,
        expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
    )
    network = build_network(case)
    catalog = build_contingency_catalog(
        case,
        network,
        contingency_table,
        validation_columns=int(config.model["lodf_validation_columns"]),
        validation_tolerance_pu=float(config.model["lodf_validation_tolerance_pu"]),
    )
    solution = payload["solution"]
    generator_records = solution["generators"]
    generators = {int(record["source_row"]): record for record in generator_records}
    if len(generators) != case.gen.shape[0] or len(generators) != len(generator_records):
        raise ProvenanceError("Solution must contain each generator source row exactly once")
    curves = build_pwl_costs(case, segments=int(config.model["pwl_segments"]))
    bus_lookup = {int(bus): index for index, bus in enumerate(network.bus_ids)}
    generation_at_bus = np.zeros(len(network.bus_ids), dtype=np.float64)
    bound_violation_mw = 0.0
    dispatch_definition_violation_mw = 0.0
    segment_bound_violation_mw = 0.0
    integrality_violation = 0.0
    objective = 0.0
    for generator_index, source in enumerate(case.gen):
        record = generators[generator_index + 1]
        commitment = float(record["commitment"])
        dispatch = float(record["dispatch_mw"])
        if source[GEN_STATUS] <= 0:
            integrality_violation = max(integrality_violation, abs(commitment))
            bound_violation_mw = max(bound_violation_mw, abs(dispatch))
            if record.get("segment_dispatch_mw"):
                raise ProvenanceError("Source-offline generator contains PWL segment values")
            continue
        integrality_violation = max(
            integrality_violation,
            abs(commitment - round(commitment)),
            -commitment,
            commitment - 1.0,
        )
        bound_violation_mw = max(
            bound_violation_mw,
            float(source[PMIN]) * commitment - dispatch,
            dispatch - float(source[PMAX]) * commitment,
            -dispatch if source[PMIN] >= 0 else 0.0,
            0.0,
        )
        curve = curves[generator_index]
        segments = np.asarray(record["segment_dispatch_mw"], dtype=np.float64)
        if len(segments) != len(curve.segment_widths_mw):
            raise ProvenanceError(
                f"Generator source row {generator_index + 1} does not have 10 segments"
            )
        segment_bound_violation_mw = max(
            segment_bound_violation_mw,
            float(np.max(-segments)),
            float(np.max(segments - curve.segment_widths_mw * commitment)),
            0.0,
        )
        dispatch_definition_violation_mw = max(
            dispatch_definition_violation_mw,
            abs(dispatch - float(source[PMIN]) * commitment - float(np.sum(segments))),
        )
        objective += curve.committed_base_cost * commitment + float(
            curve.segment_slopes_per_mwh @ segments
        )
        generation_at_bus[bus_lookup[int(source[GEN_BUS])]] += dispatch

    demand = case.bus[:, PD] + case.bus[:, GS]
    injection = generation_at_bus - demand
    global_balance_mw = abs(float(np.sum(injection)))
    model_tolerance_pu = float(config.model["model_residual_tolerance_pu"])
    balance_tolerance_mw = model_tolerance_pu * case.base_mva
    angles_records = solution["bus_angles_rad"]
    angle_by_bus = {int(record["bus"]): float(record["angle_rad"]) for record in angles_records}
    if len(angle_by_bus) != len(network.bus_ids) or len(angle_by_bus) != len(angles_records):
        raise ProvenanceError("Solution must contain each bus angle exactly once")
    stored_angles = np.asarray([angle_by_bus[int(bus)] for bus in network.bus_ids])
    stored_flow = base_flow_vector(solution, network)
    nodal_residual_mw = float(
        np.max(np.abs(injection - np.asarray(network.incidence.T @ stored_flow).ravel()))
    )
    dc_flow_from_angles = case.base_mva * network.susceptance_pu * (
        np.asarray(network.incidence @ stored_angles).ravel() - network.phase_shift_rad
    )
    dc_equation_residual_mw = float(np.max(np.abs(stored_flow - dc_flow_from_angles)))
    reference_angle_residual = abs(float(stored_angles[network.reference_bus_index]))
    limited = network.rate_a_mw > 0
    base_limit_violation_mw = float(
        np.max(np.maximum(np.abs(stored_flow[limited]) - network.rate_a_mw[limited], 0.0))
    )
    angle_difference = np.asarray(network.incidence @ stored_angles).ravel()
    active_source = case.branch[network.active_branch_source_rows]
    lower_active = (active_source[:, ANGMIN] != 0) & (active_source[:, ANGMIN] > -360)
    upper_active = (active_source[:, ANGMAX] != 0) & (active_source[:, ANGMAX] < 360)
    angle_violation = 0.0
    if np.any(lower_active):
        angle_violation = max(
            angle_violation,
            float(
                np.max(
                    np.maximum(
                        network.angle_min_rad[lower_active] - angle_difference[lower_active],
                        0,
                    )
                )
            ),
        )
    if np.any(upper_active):
        angle_violation = max(
            angle_violation,
            float(
                np.max(
                    np.maximum(
                        angle_difference[upper_active] - network.angle_max_rad[upper_active],
                        0,
                    )
                )
            ),
        )
    maximum_security_mw = 0.0
    for outage in catalog.valid:
        _, post_flow = solve_dc(
            network,
            injection,
            outage_active_index=outage.active_branch_index,
            balance_tolerance_mw=balance_tolerance_mw,
        )
        monitored = limited.copy()
        monitored[outage.active_branch_index] = False
        maximum_security_mw = max(
            maximum_security_mw,
            float(
                np.max(
                    np.maximum(
                        np.abs(post_flow[monitored]) - network.rate_a_mw[monitored], 0.0
                    )
                )
            ),
        )
    model_residual_mw = max(
        bound_violation_mw,
        segment_bound_violation_mw,
        dispatch_definition_violation_mw,
        global_balance_mw,
        nodal_residual_mw,
        dc_equation_residual_mw,
        base_limit_violation_mw,
    )
    model_residual_pu = model_residual_mw / case.base_mva
    security_violation_pu = maximum_security_mw / case.base_mva
    reported_objective = float(payload["objective"])
    objective_difference = abs(reported_objective - objective)
    objective_tolerance = 1e-7 * max(1.0, abs(objective))
    passed = (
        model_residual_pu <= model_tolerance_pu
        and security_violation_pu <= float(config.model["security_violation_tolerance_pu"])
        and integrality_violation <= model_tolerance_pu
        and angle_violation <= model_tolerance_pu
        and reference_angle_residual <= model_tolerance_pu
        and objective_difference <= objective_tolerance
    )
    checked_sides = sum(
        2 * (int(np.count_nonzero(limited)) - int(limited[outage.active_branch_index]))
        for outage in catalog.valid
    )
    return VerificationResult(
        passed=passed,
        elapsed_seconds=time.perf_counter() - started,
        maximum_model_residual_pu=model_residual_pu,
        maximum_security_violation_pu=security_violation_pu,
        maximum_integrality_violation=integrality_violation,
        maximum_angle_limit_violation_rad=angle_violation,
        objective_recalculated=objective,
        objective_difference=objective_difference,
        checked_valid_outages=len(catalog.valid),
        checked_security_sides=checked_sides,
        details={
            "conditional_pmin_pmax_violation_pu": bound_violation_mw / case.base_mva,
            "segment_bound_violation_pu": segment_bound_violation_mw / case.base_mva,
            "dispatch_definition_violation_pu": (
                dispatch_definition_violation_mw / case.base_mva
            ),
            "global_balance_residual_pu": global_balance_mw / case.base_mva,
            "nodal_balance_residual_pu": nodal_residual_mw / case.base_mva,
            "dc_flow_equation_residual_pu": dc_equation_residual_mw / case.base_mva,
            "base_limit_violation_pu": base_limit_violation_mw / case.base_mva,
            "reference_angle_residual_rad": reference_angle_residual,
        },
    )
