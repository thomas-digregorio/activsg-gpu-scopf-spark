"""Sparse DC network, contingency eligibility, and validated FP64 LODFs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu

from .errors import ProvenanceError
from .matpower import (
    ANGMAX,
    ANGMIN,
    BR_STATUS,
    BR_X,
    BUS_I,
    BUS_TYPE,
    F_BUS,
    RATE_A,
    SHIFT,
    T_BUS,
    TAP,
    ContingencyTable,
    MatpowerCase,
)
from .provenance import branch_source_id

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class NetworkData:
    base_mva: float
    bus_ids: IntArray
    reference_bus_index: int
    active_branch_source_rows: IntArray
    from_bus_index: IntArray
    to_bus_index: IntArray
    incidence: sparse.csr_matrix
    susceptance_pu: FloatArray
    phase_shift_rad: FloatArray
    rate_a_mw: FloatArray
    angle_min_rad: FloatArray
    angle_max_rad: FloatArray
    bbus: sparse.csc_matrix


@dataclass(frozen=True)
class ValidOutage:
    contingency_label: int
    contingency_source_row: int
    branch_source_row: int
    branch_source_id: str
    active_branch_index: int


@dataclass(frozen=True)
class ExcludedOutage:
    contingency_label: int
    contingency_source_rows: tuple[int, ...]
    branch_source_row: int | None
    reason: str


@dataclass(frozen=True)
class ContingencyCatalog:
    valid: tuple[ValidOutage, ...]
    excluded: tuple[ExcludedOutage, ...]
    deferred_generator_outages: tuple[ExcludedOutage, ...]
    lodf: FloatArray | None
    lodf_operator: LodfOperator
    validation_max_error_pu: float
    validation_columns: tuple[int, ...]
    build_chunk_columns: int


@dataclass(frozen=True)
class LodfOperator:
    """Sparse-factor-backed FP64 LODF column generator."""

    network: NetworkData
    factor: Any
    keep_buses: npt.NDArray[np.bool_]

    def transaction_columns(
        self, outage_active_indices: IntArray
    ) -> tuple[FloatArray, FloatArray]:
        count = len(outage_active_indices)
        if count == 0:
            return (
                np.empty(
                    (len(self.network.active_branch_source_rows), 0), dtype=np.float64
                ),
                np.empty(0, dtype=np.float64),
            )
        transaction_rhs = self.network.incidence[outage_active_indices].T.toarray()
        theta = np.zeros((len(self.network.bus_ids), count), dtype=np.float64)
        theta[self.keep_buses, :] = self.factor.solve(
            transaction_rhs[self.keep_buses, :]
        )
        ptdf = self.network.susceptance_pu[:, None] * np.asarray(
            self.network.incidence @ theta
        )
        denominator = 1.0 - ptdf[outage_active_indices, np.arange(count)]
        return np.asarray(ptdf, dtype=np.float64), np.asarray(denominator, dtype=np.float64)

    def lodf_columns(self, outage_active_indices: IntArray) -> FloatArray:
        ptdf, denominator = self.transaction_columns(outage_active_indices)
        valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-10)
        if not np.all(valid):
            bad = np.flatnonzero(~valid).tolist()
            raise ProvenanceError(f"Invalid LODF denominator in requested columns {bad}")
        lodf = ptdf / denominator[None, :]
        lodf[outage_active_indices, np.arange(len(outage_active_indices))] = -1.0
        return np.asarray(lodf, dtype=np.float64)


def build_network(case: MatpowerCase) -> NetworkData:
    bus_ids = case.bus[:, BUS_I].astype(np.int64)
    lookup = {int(bus_id): index for index, bus_id in enumerate(bus_ids)}
    active_source_rows = np.flatnonzero(case.branch[:, BR_STATUS] > 0).astype(np.int64)
    branch = case.branch[active_source_rows]
    x = branch[:, BR_X]
    if np.any(~np.isfinite(x)) or np.any(np.abs(x) <= 1e-12):
        rows = (active_source_rows[np.abs(x) <= 1e-12] + 1).tolist()
        raise ProvenanceError(f"In-service branch has invalid reactance at source rows {rows}")
    tap = np.where(branch[:, TAP] == 0, 1.0, branch[:, TAP])
    if np.any(~np.isfinite(tap)) or np.any(tap <= 0):
        raise ProvenanceError("In-service branch has a non-positive/non-finite tap ratio")
    from_index = np.asarray([lookup[int(value)] for value in branch[:, F_BUS]], dtype=np.int64)
    to_index = np.asarray([lookup[int(value)] for value in branch[:, T_BUS]], dtype=np.int64)
    line_count = len(active_source_rows)
    incidence = sparse.coo_matrix(
        (
            np.concatenate((np.ones(line_count), -np.ones(line_count))),
            (
                np.concatenate((np.arange(line_count), np.arange(line_count))),
                np.concatenate((from_index, to_index)),
            ),
        ),
        shape=(line_count, len(bus_ids)),
        dtype=np.float64,
    ).tocsr()
    susceptance = 1.0 / (x * tap)
    bbus = (incidence.T @ sparse.diags(susceptance) @ incidence).tocsc()
    graph_components = _component_count(from_index, to_index, len(bus_ids))
    if graph_components != 1:
        raise ProvenanceError(f"The in-service base network has {graph_components} islands")
    reference_candidates = np.flatnonzero(case.bus[:, BUS_TYPE] == 3)
    if reference_candidates.size == 0:
        raise ProvenanceError("No MATPOWER reference bus (BUS_TYPE 3) exists")
    return NetworkData(
        base_mva=case.base_mva,
        bus_ids=bus_ids,
        reference_bus_index=int(reference_candidates[0]),
        active_branch_source_rows=active_source_rows,
        from_bus_index=from_index,
        to_bus_index=to_index,
        incidence=incidence,
        susceptance_pu=susceptance,
        phase_shift_rad=branch[:, SHIFT] * pi / 180.0,
        rate_a_mw=branch[:, RATE_A].astype(np.float64),
        angle_min_rad=branch[:, ANGMIN] * pi / 180.0,
        angle_max_rad=branch[:, ANGMAX] * pi / 180.0,
        bbus=bbus,
    )


def _component_count(
    from_index: IntArray,
    to_index: IntArray,
    bus_count: int,
    *,
    remove: int | None = None,
) -> int:
    mask = np.ones(len(from_index), dtype=bool)
    if remove is not None:
        mask[remove] = False
    graph = sparse.coo_matrix(
        (
            np.ones(2 * int(np.count_nonzero(mask))),
            (
                np.concatenate((from_index[mask], to_index[mask])),
                np.concatenate((to_index[mask], from_index[mask])),
            ),
        ),
        shape=(bus_count, bus_count),
    )
    return int(csgraph.connected_components(graph, directed=False, return_labels=False))


def _bridge_active_indices(network: NetworkData) -> set[int]:
    """Return multigraph-aware bridge edge IDs in O(buses + branches)."""

    bus_count = len(network.bus_ids)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(bus_count)]
    for edge, (from_bus, to_bus) in enumerate(
        zip(network.from_bus_index, network.to_bus_index, strict=True)
    ):
        left = int(from_bus)
        right = int(to_bus)
        adjacency[left].append((right, edge))
        adjacency[right].append((left, edge))
    discovered = np.full(bus_count, -1, dtype=np.int64)
    low = np.full(bus_count, -1, dtype=np.int64)
    parent = np.full(bus_count, -1, dtype=np.int64)
    parent_edge = np.full(bus_count, -1, dtype=np.int64)
    next_neighbor = np.zeros(bus_count, dtype=np.int64)
    bridges: set[int] = set()
    tick = 0
    for root in range(bus_count):
        if discovered[root] >= 0:
            continue
        discovered[root] = tick
        low[root] = tick
        tick += 1
        stack = [root]
        while stack:
            node = stack[-1]
            cursor = int(next_neighbor[node])
            if cursor < len(adjacency[node]):
                neighbor, edge = adjacency[node][cursor]
                next_neighbor[node] += 1
                if edge == parent_edge[node]:
                    continue
                if discovered[neighbor] < 0:
                    parent[neighbor] = node
                    parent_edge[neighbor] = edge
                    discovered[neighbor] = tick
                    low[neighbor] = tick
                    tick += 1
                    stack.append(neighbor)
                else:
                    low[node] = min(low[node], discovered[neighbor])
                continue
            stack.pop()
            edge = int(parent_edge[node])
            if edge >= 0:
                parent_node = int(parent[node])
                if low[node] > discovered[parent_node]:
                    bridges.add(edge)
                low[parent_node] = min(low[parent_node], low[node])
    return bridges


def _reduced_factor(network: NetworkData, keep_lines: npt.NDArray[np.bool_] | None = None):
    if keep_lines is None:
        bbus = network.bbus
    else:
        c = network.incidence[keep_lines]
        bbus = (c.T @ sparse.diags(network.susceptance_pu[keep_lines]) @ c).tocsc()
    keep_buses = np.ones(len(network.bus_ids), dtype=bool)
    keep_buses[network.reference_bus_index] = False
    return splu(bbus[keep_buses][:, keep_buses].tocsc()), keep_buses


def solve_dc(
    network: NetworkData,
    net_injection_mw: FloatArray,
    *,
    outage_active_index: int | None = None,
    balance_tolerance_mw: float = 1e-8,
) -> tuple[FloatArray, FloatArray]:
    if abs(float(np.sum(net_injection_mw))) > balance_tolerance_mw:
        raise ProvenanceError("DC net injections must sum to zero")
    keep_lines = np.ones(len(network.active_branch_source_rows), dtype=bool)
    if outage_active_index is not None:
        keep_lines[outage_active_index] = False
    factor, keep_buses = _reduced_factor(network, keep_lines)
    c = network.incidence[keep_lines]
    b = network.susceptance_pu[keep_lines]
    shift = network.phase_shift_rad[keep_lines]
    rhs = net_injection_mw / network.base_mva + np.asarray(c.T @ (b * shift)).ravel()
    theta = np.zeros(len(network.bus_ids), dtype=np.float64)
    theta[keep_buses] = factor.solve(rhs[keep_buses])
    partial_flow = network.base_mva * b * (np.asarray(c @ theta).ravel() - shift)
    flow = np.zeros(len(network.active_branch_source_rows), dtype=np.float64)
    flow[keep_lines] = partial_flow
    return theta, flow


def _candidate_outages(
    case: MatpowerCase, network: NetworkData, table: ContingencyTable
) -> tuple[list[ValidOutage], list[ExcludedOutage], list[ExcludedOutage]]:
    grouped: dict[int, list] = defaultdict(list)
    for change in table.changes:
        grouped[change.label].append(change)
    active_lookup = {
        int(source_row): active_index
        for active_index, source_row in enumerate(network.active_branch_source_rows)
    }
    bridge_indices = _bridge_active_indices(network)
    valid: list[ValidOutage] = []
    excluded: list[ExcludedOutage] = []
    deferred: list[ExcludedOutage] = []
    for label in sorted(grouped):
        changes = grouped[label]
        source_rows = tuple(change.source_row for change in changes)
        if any(change.table == "CT_TGEN" for change in changes):
            deferred.append(
                ExcludedOutage(label, source_rows, None, "generator_outage_deferred_v1")
            )
            continue
        if len(changes) != 1:
            excluded.append(ExcludedOutage(label, source_rows, None, "not_single_branch_outage"))
            continue
        change = changes[0]
        branch_row = change.element_row
        if (
            change.table != "CT_TBRCH"
            or change.column != "BR_STATUS"
            or change.change_type != "CT_REP"
            or change.new_value != 0
        ):
            excluded.append(
                ExcludedOutage(label, source_rows, branch_row, "invalid_branch_outage_change")
            )
            continue
        if branch_row < 1 or branch_row > case.branch.shape[0]:
            excluded.append(ExcludedOutage(label, source_rows, branch_row, "invalid_branch_row"))
            continue
        source_index = branch_row - 1
        if case.branch[source_index, BR_STATUS] <= 0:
            excluded.append(ExcludedOutage(label, source_rows, branch_row, "branch_out_of_service"))
            continue
        reactance = case.branch[source_index, BR_X]
        if not np.isfinite(reactance) or abs(reactance) <= 1e-12:
            excluded.append(ExcludedOutage(label, source_rows, branch_row, "invalid_reactance"))
            continue
        active_index = active_lookup[source_index]
        if active_index in bridge_indices:
            excluded.append(ExcludedOutage(label, source_rows, branch_row, "islanding_bridge"))
            continue
        valid.append(
            ValidOutage(
                contingency_label=label,
                contingency_source_row=change.source_row,
                branch_source_row=branch_row,
                branch_source_id=branch_source_id(source_index),
                active_branch_index=active_index,
            )
        )
    return valid, excluded, deferred


def build_contingency_catalog(
    case: MatpowerCase,
    network: NetworkData,
    table: ContingencyTable,
    *,
    validation_columns: int = 3,
    validation_tolerance_pu: float = 1e-9,
    chunk_columns: int = 256,
    materialize_lodf: bool = True,
) -> ContingencyCatalog:
    if chunk_columns <= 0:
        raise ProvenanceError("LODF chunk_columns must be positive")
    candidates, excluded, deferred = _candidate_outages(case, network, table)
    factor, keep_buses = _reduced_factor(network)
    operator = LodfOperator(network, factor, keep_buses)
    outage_indices = np.asarray(
        [item.active_branch_index for item in candidates], dtype=np.int64
    )
    numerically_valid = np.ones(len(candidates), dtype=bool)
    lodf_all = (
        np.empty(
            (len(network.active_branch_source_rows), len(candidates)), dtype=np.float64
        )
        if materialize_lodf
        else None
    )
    for start in range(0, len(candidates), chunk_columns):
        stop = min(start + chunk_columns, len(candidates))
        chunk_indices = outage_indices[start:stop]
        ptdf, denominator = operator.transaction_columns(chunk_indices)
        valid_chunk = np.isfinite(denominator) & (np.abs(denominator) > 1e-10)
        numerically_valid[start:stop] = valid_chunk
        if lodf_all is not None:
            safe_denominator = np.where(valid_chunk, denominator, 1.0)
            lodf_chunk = ptdf / safe_denominator[None, :]
            lodf_chunk[:, ~valid_chunk] = 0.0
            if np.any(valid_chunk):
                local = np.flatnonzero(valid_chunk)
                lodf_chunk[chunk_indices[local], local] = -1.0
            lodf_all[:, start:stop] = lodf_chunk
    final_valid: list[ValidOutage] = []
    for column, item in enumerate(candidates):
        if numerically_valid[column]:
            final_valid.append(item)
        else:
            excluded.append(
                ExcludedOutage(
                    item.contingency_label,
                    (item.contingency_source_row,),
                    item.branch_source_row,
                    "invalid_lodf_denominator",
                )
            )
    if lodf_all is None:
        lodf = None
    elif np.all(numerically_valid):
        lodf = lodf_all
    else:
        lodf = np.ascontiguousarray(lodf_all[:, numerically_valid])
    final_outage_indices = np.asarray(
        [item.active_branch_index for item in final_valid], dtype=np.int64
    )
    selected = _evenly_spaced_indices(len(final_valid), validation_columns)
    if selected:
        if lodf is None:
            selected_lodf = operator.lodf_columns(final_outage_indices[list(selected)])
        else:
            selected_lodf = lodf[:, list(selected)]
        selected_outages = tuple(final_valid[column] for column in selected)
        local_columns = tuple(range(len(selected)))
        max_error = validate_lodf_columns(
            network, selected_outages, selected_lodf, local_columns
        )
    else:
        max_error = 0.0
    if max_error > validation_tolerance_pu:
        raise ProvenanceError(
            f"LODF explicit-solve validation error {max_error:.3e} p.u. exceeds "
            f"{validation_tolerance_pu:.3e}"
        )
    return ContingencyCatalog(
        valid=tuple(final_valid),
        excluded=tuple(sorted(excluded, key=lambda item: item.contingency_label)),
        deferred_generator_outages=tuple(deferred),
        lodf=None if lodf is None else np.asarray(lodf, dtype=np.float64),
        lodf_operator=LodfOperator(network, factor, keep_buses),
        validation_max_error_pu=max_error,
        validation_columns=selected,
        build_chunk_columns=chunk_columns,
    )


def _evenly_spaced_indices(length: int, count: int) -> tuple[int, ...]:
    if length == 0 or count <= 0:
        return ()
    return tuple(sorted(set(np.linspace(0, length - 1, min(count, length), dtype=int).tolist())))


def validate_lodf_columns(
    network: NetworkData,
    outages: tuple[ValidOutage, ...],
    lodf: FloatArray,
    selected_columns: tuple[int, ...],
) -> float:
    maximum = 0.0
    for column in selected_columns:
        outage_index = outages[column].active_branch_index
        injection = np.zeros(len(network.bus_ids), dtype=np.float64)
        injection[network.from_bus_index[outage_index]] = 1.0
        injection[network.to_bus_index[outage_index]] = -1.0
        _, base_flow = solve_dc(network, injection)
        _, explicit_flow = solve_dc(network, injection, outage_active_index=outage_index)
        predicted = base_flow + lodf[:, column] * base_flow[outage_index]
        maximum = max(
            maximum,
            float(np.max(np.abs(explicit_flow - predicted)) / network.base_mva),
        )
    return maximum


def contingency_catalog_report(catalog: ContingencyCatalog) -> dict[str, object]:
    def excluded_payload(item: ExcludedOutage) -> dict[str, object]:
        return {
            "contingency_label": item.contingency_label,
            "contingency_source_rows": list(item.contingency_source_rows),
            "branch_source_row": item.branch_source_row,
            "reason": item.reason,
        }

    return {
        "valid_branch_outages": len(catalog.valid),
        "excluded_branch_outages": [excluded_payload(item) for item in catalog.excluded],
        "deferred_generator_outages": [
            excluded_payload(item) for item in catalog.deferred_generator_outages
        ],
        "lodf_validation": {
            "selected_column_indices": list(catalog.validation_columns),
            "maximum_error_pu": catalog.validation_max_error_pu,
            "materialized": catalog.lodf is not None,
            "fp64_bytes": 0 if catalog.lodf is None else int(catalog.lodf.nbytes),
            "build_chunk_columns": catalog.build_chunk_columns,
        },
    }
