"""Single-hour exact-PMIN preventive DC unit commitment master model."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, pi

import numpy as np
import numpy.typing as npt

from .canonical import CanonicalMILP
from .costs import PwlCost, build_pwl_costs
from .matpower import GEN_BUS, GEN_STATUS, GS, PD, MatpowerCase
from .network import NetworkData

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ModelIndex:
    generator_source_rows: IntArray
    commitment_by_generator: dict[int, int]
    dispatch_by_generator: dict[int, int]
    segments_by_generator: dict[int, tuple[int, ...]]
    theta_by_bus: IntArray
    flow_by_active_branch: IntArray


@dataclass(frozen=True)
class MasterModel:
    canonical: CanonicalMILP
    index: ModelIndex
    costs: dict[int, PwlCost]


def build_master(case: MatpowerCase, network: NetworkData, *, segments: int = 10) -> MasterModel:
    model = CanonicalMILP()
    curves = build_pwl_costs(case, segments=segments)
    online = np.flatnonzero(case.gen[:, GEN_STATUS] > 0).astype(np.int64)
    commitment: dict[int, int] = {}
    dispatch: dict[int, int] = {}
    segment_columns: dict[int, tuple[int, ...]] = {}
    for generator_index in online:
        row = int(generator_index) + 1
        curve = curves[int(generator_index)]
        u = model.add_variable(
            f"u_g{row:04d}",
            objective=curve.committed_base_cost,
            lower=0.0,
            upper=1.0,
            integer=True,
        )
        p = model.add_variable(
            f"pg_g{row:04d}", lower=min(0.0, curve.pmin_mw), upper=max(0.0, curve.pmax_mw)
        )
        commitment[int(generator_index)] = u
        dispatch[int(generator_index)] = p
        y_columns: list[int] = []
        for segment, (width, slope) in enumerate(
            zip(curve.segment_widths_mw, curve.segment_slopes_per_mwh, strict=True),
            start=1,
        ):
            y = model.add_variable(
                f"pseg_g{row:04d}_s{segment:02d}",
                objective=float(slope),
                lower=0.0,
                upper=float(width),
            )
            y_columns.append(y)
            model.add_row(
                f"segment_on_g{row:04d}_s{segment:02d}",
                {y: 1.0, u: -float(width)},
                upper=0.0,
            )
        segment_columns[int(generator_index)] = tuple(y_columns)
        definition = {p: 1.0, u: -curve.pmin_mw}
        definition.update({column: -1.0 for column in y_columns})
        model.add_row(f"exact_pmin_dispatch_g{row:04d}", definition, lower=0.0, upper=0.0)

    theta_columns = np.empty(len(network.bus_ids), dtype=np.int64)
    for bus_index, bus_id in enumerate(network.bus_ids):
        reference = bus_index == network.reference_bus_index
        theta_columns[bus_index] = model.add_variable(
            f"theta_b{int(bus_id):04d}",
            lower=0.0 if reference else -inf,
            upper=0.0 if reference else inf,
        )
    flow_columns = np.empty(len(network.active_branch_source_rows), dtype=np.int64)
    for active_index, source_index in enumerate(network.active_branch_source_rows):
        rate = float(network.rate_a_mw[active_index])
        limit = rate if rate > 0 else inf
        flow_columns[active_index] = model.add_variable(
            f"flow_l{int(source_index) + 1:04d}", lower=-limit, upper=limit
        )

    bus_lookup = {int(bus_id): i for i, bus_id in enumerate(network.bus_ids)}
    generators_at_bus: dict[int, list[int]] = {i: [] for i in range(len(network.bus_ids))}
    for generator_index in online:
        bus_index = bus_lookup[int(case.gen[generator_index, GEN_BUS])]
        generators_at_bus[bus_index].append(dispatch[int(generator_index)])
    demand = case.bus[:, PD] + case.bus[:, GS]
    for bus_index, bus_id in enumerate(network.bus_ids):
        coefficients = {column: 1.0 for column in generators_at_bus[bus_index]}
        for active_index in network.incidence[:, bus_index].nonzero()[0]:
            incidence_value = float(network.incidence[active_index, bus_index])
            coefficients[int(flow_columns[active_index])] = -incidence_value
        rhs = float(demand[bus_index])
        model.add_row(
            f"nodal_balance_b{int(bus_id):04d}", coefficients, lower=rhs, upper=rhs
        )

    for active_index, source_index in enumerate(network.active_branch_source_rows):
        coefficient = network.base_mva * network.susceptance_pu[active_index]
        from_bus = network.from_bus_index[active_index]
        to_bus = network.to_bus_index[active_index]
        rhs = -coefficient * network.phase_shift_rad[active_index]
        model.add_row(
            f"dc_flow_l{int(source_index) + 1:04d}",
            {
                int(flow_columns[active_index]): 1.0,
                int(theta_columns[from_bus]): -coefficient,
                int(theta_columns[to_bus]): coefficient,
            },
            lower=rhs,
            upper=rhs,
        )
        angle_min = network.angle_min_rad[active_index]
        angle_max = network.angle_max_rad[active_index]
        if angle_min != 0 and angle_min > -2 * pi:
            model.add_row(
                f"angle_min_l{int(source_index) + 1:04d}",
                {int(theta_columns[from_bus]): 1.0, int(theta_columns[to_bus]): -1.0},
                lower=float(angle_min),
            )
        if angle_max != 0 and angle_max < 2 * pi:
            model.add_row(
                f"angle_max_l{int(source_index) + 1:04d}",
                {int(theta_columns[from_bus]): 1.0, int(theta_columns[to_bus]): -1.0},
                upper=float(angle_max),
            )

    return MasterModel(
        canonical=model,
        index=ModelIndex(
            generator_source_rows=online,
            commitment_by_generator=commitment,
            dispatch_by_generator=dispatch,
            segments_by_generator=segment_columns,
            theta_by_bus=theta_columns,
            flow_by_active_branch=flow_columns,
        ),
        costs=curves,
    )
