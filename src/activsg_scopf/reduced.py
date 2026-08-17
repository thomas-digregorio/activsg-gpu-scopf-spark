"""Injection-space DC model used by the GPU Lagrangian experiment.

The ordinary benchmark keeps bus angles and branch flows as variables.  This
module eliminates those network variables with the same FP64 DC equations so
that coupling rows contain only generator dispatch.  Generator commitment,
exact PMIN, and the ten source-derived PWL segments remain explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.linalg import splu

from .canonical import CanonicalMILP
from .costs import PwlCost, build_pwl_costs
from .errors import ScopfError
from .matpower import GEN_BUS, GEN_STATUS, GS, PD, MatpowerCase
from .model import MasterModel, build_master
from .network import ContingencyCatalog, NetworkData, solve_dc
from .screening import SecurityPair

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class InjectionOperator:
    """Affine base-flow and angle maps in MW dispatch coordinates."""

    generator_source_rows: IntArray
    generator_bus_indices: IntArray
    demand_mw: FloatArray
    flow_constant_mw: FloatArray
    flow_by_generator: FloatArray
    flow_by_bus_injection: FloatArray
    angle_constant_rad: FloatArray
    angle_by_generator: FloatArray
    angle_by_bus_injection: FloatArray

    @property
    def total_demand_mw(self) -> float:
        return float(np.sum(self.demand_mw))

    def flows(self, dispatch_mw: FloatArray) -> FloatArray:
        dispatch = np.asarray(dispatch_mw, dtype=np.float64)
        expected = (self.generator_source_rows.size,)
        if dispatch.shape != expected:
            raise ScopfError(
                f"Reduced dispatch shape {dispatch.shape} does not match {expected}"
            )
        return self.flow_constant_mw + self.flow_by_generator @ dispatch

    def angles(self, dispatch_mw: FloatArray) -> FloatArray:
        dispatch = np.asarray(dispatch_mw, dtype=np.float64)
        expected = (self.generator_source_rows.size,)
        if dispatch.shape != expected:
            raise ScopfError(
                f"Reduced dispatch shape {dispatch.shape} does not match {expected}"
            )
        return self.angle_constant_rad + self.angle_by_generator @ dispatch


@dataclass(frozen=True)
class CouplingRow:
    row_index: int
    row_name: str
    rhs: float
    generator_coefficients: FloatArray
    bus_coefficients: FloatArray
    kind: str


@dataclass(frozen=True)
class ReducedIndex:
    generator_source_rows: IntArray
    commitment_by_generator: dict[int, int]
    dispatch_by_generator: dict[int, int]
    segments_by_generator: dict[int, tuple[int | None, ...]]


@dataclass
class ReducedMaster:
    canonical: CanonicalMILP
    index: ReducedIndex
    costs: dict[int, PwlCost]
    operator: InjectionOperator
    coupling_rows: list[CouplingRow] = field(default_factory=list)
    security_pair_ids: set[str] = field(default_factory=set)


def build_injection_operator(case: MatpowerCase, network: NetworkData) -> InjectionOperator:
    """Eliminate angles using the same reference-bus equations as ``solve_dc``."""

    online = np.flatnonzero(case.gen[:, GEN_STATUS] > 0).astype(np.int64)
    bus_lookup = {int(bus_id): index for index, bus_id in enumerate(network.bus_ids)}
    generator_buses = np.asarray(
        [bus_lookup[int(case.gen[index, GEN_BUS])] for index in online], dtype=np.int64
    )
    bus_count = len(network.bus_ids)
    keep = np.ones(bus_count, dtype=bool)
    keep[network.reference_bus_index] = False
    factor = splu(network.bbus[keep][:, keep].tocsc())

    demand = np.asarray(case.bus[:, PD] + case.bus[:, GS], dtype=np.float64)
    shift_injection = np.asarray(
        network.incidence.T
        @ (network.susceptance_pu * network.phase_shift_rad)
    ).ravel()
    constant_rhs = -demand / network.base_mva + shift_injection
    theta_constant = np.zeros(bus_count, dtype=np.float64)
    theta_constant[keep] = factor.solve(constant_rhs[keep])

    identity_rhs = np.eye(bus_count, dtype=np.float64) / network.base_mva
    theta_by_bus = np.zeros((bus_count, bus_count), dtype=np.float64)
    theta_by_bus[keep, :] = factor.solve(identity_rhs[keep, :])
    theta_by_generator = theta_by_bus[:, generator_buses]

    branch_angle_constant = np.asarray(network.incidence @ theta_constant).ravel()
    flow_constant = (
        network.base_mva
        * network.susceptance_pu
        * (branch_angle_constant - network.phase_shift_rad)
    )
    branch_angle_by_bus = np.asarray(network.incidence @ theta_by_bus)
    flow_by_bus = (
        network.base_mva * network.susceptance_pu[:, None] * branch_angle_by_bus
    )
    flow_by_generator = flow_by_bus[:, generator_buses]

    return InjectionOperator(
        generator_source_rows=online,
        generator_bus_indices=generator_buses,
        demand_mw=demand,
        flow_constant_mw=np.asarray(flow_constant, dtype=np.float64),
        flow_by_generator=np.asarray(flow_by_generator, dtype=np.float64),
        flow_by_bus_injection=np.asarray(flow_by_bus, dtype=np.float64),
        angle_constant_rad=theta_constant,
        angle_by_generator=np.asarray(theta_by_generator, dtype=np.float64),
        angle_by_bus_injection=np.asarray(theta_by_bus, dtype=np.float64),
    )


def _coefficient_map(columns: list[int], values: FloatArray) -> dict[int, float]:
    return {
        int(column): float(value)
        for column, value in zip(columns, values, strict=True)
        if value != 0.0
    }


def _add_upper_coupling_row(
    master: ReducedMaster,
    name: str,
    generator_coefficients: FloatArray,
    bus_coefficients: FloatArray,
    rhs: float,
    *,
    kind: str,
) -> None:
    dispatch_columns = [
        master.index.dispatch_by_generator[int(generator)]
        for generator in master.index.generator_source_rows
    ]
    row = master.canonical.add_row(
        name,
        _coefficient_map(dispatch_columns, generator_coefficients),
        upper=float(rhs),
    )
    master.coupling_rows.append(
        CouplingRow(
            row_index=row,
            row_name=name,
            rhs=float(rhs),
            generator_coefficients=np.asarray(generator_coefficients, dtype=np.float64),
            bus_coefficients=np.asarray(bus_coefficients, dtype=np.float64),
            kind=kind,
        )
    )


def build_reduced_master(
    case: MatpowerCase,
    network: NetworkData,
    *,
    segments: int = 10,
) -> ReducedMaster:
    """Build the convex-hull-ready generator model with base DC constraints."""

    operator = build_injection_operator(case, network)
    model = CanonicalMILP()
    curves = build_pwl_costs(case, segments=segments)
    commitment: dict[int, int] = {}
    dispatch: dict[int, int] = {}
    segment_columns: dict[int, tuple[int | None, ...]] = {}
    for generator_index in operator.generator_source_rows:
        generator = int(generator_index)
        row = generator + 1
        curve = curves[generator]
        u = model.add_variable(
            f"u_g{row:04d}",
            objective=curve.committed_base_cost,
            lower=0.0,
            upper=1.0,
            integer=True,
        )
        p = model.add_variable(
            f"pg_g{row:04d}",
            lower=min(0.0, curve.pmin_mw),
            upper=max(0.0, curve.pmax_mw),
        )
        commitment[generator] = u
        dispatch[generator] = p
        y_columns: list[int | None] = []
        for segment, (width, slope) in enumerate(
            zip(curve.segment_widths_mw, curve.segment_slopes_per_mwh, strict=True),
            start=1,
        ):
            if width == 0.0:
                y_columns.append(None)
                continue
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
        segment_columns[generator] = tuple(y_columns)
        definition = {p: 1.0, u: -curve.pmin_mw}
        definition.update({column: -1.0 for column in y_columns if column is not None})
        model.add_row(
            f"exact_pmin_dispatch_g{row:04d}", definition, lower=0.0, upper=0.0
        )

    index = ReducedIndex(
        generator_source_rows=operator.generator_source_rows,
        commitment_by_generator=commitment,
        dispatch_by_generator=dispatch,
        segments_by_generator=segment_columns,
    )
    master = ReducedMaster(model, index, curves, operator)
    dispatch_columns = [dispatch[int(generator)] for generator in operator.generator_source_rows]
    balance_coefficients = np.ones(len(dispatch_columns), dtype=np.float64)
    balance_bus_coefficients = np.ones(len(network.bus_ids), dtype=np.float64)
    balance_row = model.add_row(
        "lag_balance",
        _coefficient_map(dispatch_columns, balance_coefficients),
        lower=operator.total_demand_mw,
        upper=operator.total_demand_mw,
    )
    master.coupling_rows.append(
        CouplingRow(
            row_index=balance_row,
            row_name="lag_balance",
            rhs=operator.total_demand_mw,
            generator_coefficients=balance_coefficients,
            bus_coefficients=balance_bus_coefficients,
            kind="balance_equality",
        )
    )

    for active_index, source_index in enumerate(network.active_branch_source_rows):
        rate = float(network.rate_a_mw[active_index])
        if rate > 0.0:
            generator_sensitivity = operator.flow_by_generator[active_index]
            bus_sensitivity = operator.flow_by_bus_injection[active_index]
            constant = float(operator.flow_constant_mw[active_index])
            source_row = int(source_index) + 1
            _add_upper_coupling_row(
                master,
                f"base_l{source_row:04d}_upper",
                generator_sensitivity,
                bus_sensitivity,
                rate - constant,
                kind="base_upper",
            )
            _add_upper_coupling_row(
                master,
                f"base_l{source_row:04d}_lower",
                -generator_sensitivity,
                -bus_sensitivity,
                rate + constant,
                kind="base_lower",
            )

        from_bus = int(network.from_bus_index[active_index])
        to_bus = int(network.to_bus_index[active_index])
        angle_constant = float(
            operator.angle_constant_rad[from_bus]
            - operator.angle_constant_rad[to_bus]
        )
        angle_generator = (
            operator.angle_by_generator[from_bus]
            - operator.angle_by_generator[to_bus]
        )
        angle_bus = (
            operator.angle_by_bus_injection[from_bus]
            - operator.angle_by_bus_injection[to_bus]
        )
        source_row = int(source_index) + 1
        angle_min = float(network.angle_min_rad[active_index])
        angle_max = float(network.angle_max_rad[active_index])
        if angle_min != 0.0 and angle_min > -2 * pi:
            _add_upper_coupling_row(
                master,
                f"angle_min_l{source_row:04d}",
                -angle_generator,
                -angle_bus,
                -angle_min + angle_constant,
                kind="angle_lower",
            )
        if angle_max != 0.0 and angle_max < 2 * pi:
            _add_upper_coupling_row(
                master,
                f"angle_max_l{source_row:04d}",
                angle_generator,
                angle_bus,
                angle_max - angle_constant,
                kind="angle_upper",
            )
    return master


def add_reduced_security_pairs(
    master: ReducedMaster,
    network: NetworkData,
    pairs: tuple[SecurityPair, ...],
) -> None:
    """Append deterministic post-contingency flow rows in dispatch space."""

    for pair in pairs:
        if pair.pair_id in master.security_pair_ids:
            continue
        monitored = pair.monitored_active_index
        outaged = pair.outage_active_index
        lodf = pair.lodf_value
        generator_sensitivity = (
            master.operator.flow_by_generator[monitored]
            + lodf * master.operator.flow_by_generator[outaged]
        )
        bus_sensitivity = (
            master.operator.flow_by_bus_injection[monitored]
            + lodf * master.operator.flow_by_bus_injection[outaged]
        )
        constant = float(
            master.operator.flow_constant_mw[monitored]
            + lodf * master.operator.flow_constant_mw[outaged]
        )
        rate = float(network.rate_a_mw[monitored])
        if pair.side == "upper":
            coefficients = generator_sensitivity
            bus_coefficients = bus_sensitivity
            rhs = rate - constant
        else:
            coefficients = -generator_sensitivity
            bus_coefficients = -bus_sensitivity
            rhs = rate + constant
        _add_upper_coupling_row(
            master,
            pair.pair_id,
            coefficients,
            bus_coefficients,
            rhs,
            kind=f"contingency_{pair.side}",
        )
        master.security_pair_ids.add(pair.pair_id)


def fix_commitments(
    master: ReducedMaster,
    fixed_off: npt.ArrayLike,
    fixed_on: npt.ArrayLike,
) -> None:
    off = np.asarray(fixed_off, dtype=bool)
    on = np.asarray(fixed_on, dtype=bool)
    expected = (master.index.generator_source_rows.size,)
    if off.shape != expected or on.shape != expected or np.any(off & on):
        raise ScopfError("Invalid disjunctive commitment masks")
    for position, generator_index in enumerate(master.index.generator_source_rows):
        column = master.index.commitment_by_generator[int(generator_index)]
        if off[position]:
            master.canonical.column_lower[column] = 0.0
            master.canonical.column_upper[column] = 0.0
        elif on[position]:
            master.canonical.column_lower[column] = 1.0
            master.canonical.column_upper[column] = 1.0


def reduced_dispatch(master: ReducedMaster, values: FloatArray) -> FloatArray:
    return np.asarray(
        [
            values[master.index.dispatch_by_generator[int(generator)]]
            for generator in master.index.generator_source_rows
        ],
        dtype=np.float64,
    )


def commitment_vector(master: ReducedMaster, values: FloatArray) -> FloatArray:
    return np.asarray(
        [
            values[master.index.commitment_by_generator[int(generator)]]
            for generator in master.index.generator_source_rows
        ],
        dtype=np.float64,
    )


def reconstruct_full_values(
    case: MatpowerCase,
    network: NetworkData,
    reduced: ReducedMaster,
    reduced_values: FloatArray,
    *,
    exact_commitment: npt.ArrayLike | None = None,
) -> tuple[MasterModel, FloatArray]:
    """Lift a reduced solution into the canonical angle/flow representation."""

    full = build_master(case, network, segments=10)
    values = np.zeros(full.canonical.num_columns, dtype=np.float64)
    forced = None if exact_commitment is None else np.asarray(exact_commitment, dtype=np.float64)
    if forced is not None and forced.shape != (reduced.index.generator_source_rows.size,):
        raise ScopfError("Exact commitment vector has the wrong size")
    for position, generator_index in enumerate(reduced.index.generator_source_rows):
        generator = int(generator_index)
        reduced_u = reduced.index.commitment_by_generator[generator]
        full_u = full.index.commitment_by_generator[generator]
        values[full_u] = reduced_values[reduced_u] if forced is None else forced[position]
        values[full.index.dispatch_by_generator[generator]] = reduced_values[
            reduced.index.dispatch_by_generator[generator]
        ]
        for reduced_segment, full_segment in zip(
            reduced.index.segments_by_generator[generator],
            full.index.segments_by_generator[generator],
            strict=True,
        ):
            if full_segment is not None:
                if reduced_segment is None:
                    raise ScopfError("Reduced/full PWL segment identities differ")
                values[full_segment] = reduced_values[reduced_segment]

    dispatch = reduced_dispatch(reduced, reduced_values)
    generation = sparse.coo_matrix(
        (
            dispatch,
            (
                reduced.operator.generator_bus_indices,
                np.zeros(dispatch.size, dtype=np.int64),
            ),
        ),
        shape=(len(network.bus_ids), 1),
    ).toarray().ravel()
    injection = generation - reduced.operator.demand_mw
    theta, flow = solve_dc(
        network,
        injection,
        balance_tolerance_mw=1e-4,
    )
    values[full.index.theta_by_bus] = theta
    values[full.index.flow_by_active_branch] = flow
    return full, values


def security_pair_record(pair: SecurityPair) -> dict[str, object]:
    return {
        "pair_id": pair.pair_id,
        "contingency_label": pair.contingency_label,
        "monitored_branch_source_row": pair.monitored_branch_source_row,
        "side": pair.side,
        "outage_column": pair.outage_column,
        "monitored_active_index": pair.monitored_active_index,
        "outage_active_index": pair.outage_active_index,
        "lodf_value": pair.lodf_value,
    }


def security_pair_from_record(
    record: dict[str, object], catalog: ContingencyCatalog
) -> SecurityPair:
    pair = SecurityPair(
        contingency_label=int(record["contingency_label"]),
        monitored_branch_source_row=int(record["monitored_branch_source_row"]),
        side=str(record["side"]),  # type: ignore[arg-type]
        outage_column=int(record["outage_column"]),
        monitored_active_index=int(record["monitored_active_index"]),
        outage_active_index=int(record["outage_active_index"]),
        lodf_value=float(record["lodf_value"]),
    )
    if pair.pair_id != record.get("pair_id"):
        raise ScopfError("Serialized security-pair identity mismatch")
    if pair.outage_column < 0 or pair.outage_column >= len(catalog.valid):
        raise ScopfError("Serialized security pair has an invalid outage column")
    outage = catalog.valid[pair.outage_column]
    if (
        outage.contingency_label != pair.contingency_label
        or outage.active_branch_index != pair.outage_active_index
    ):
        raise ScopfError("Serialized security pair does not match the raw contingency")
    observed_lodf = (
        catalog.lodf[pair.monitored_active_index, pair.outage_column]
        if catalog.lodf is not None
        else catalog.lodf_operator.lodf_columns(
            np.asarray([pair.outage_active_index], dtype=np.int64)
        )[pair.monitored_active_index, 0]
    )
    if float(observed_lodf) != pair.lodf_value:
        raise ScopfError("Serialized security pair LODF differs from the raw-case operator")
    return pair
