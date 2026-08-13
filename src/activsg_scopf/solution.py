"""Stable serialization of canonical variables into source-row records."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .matpower import GEN_STATUS, MatpowerCase
from .model import MasterModel
from .network import NetworkData
from .provenance import branch_source_id, generator_source_id

FloatArray = npt.NDArray[np.float64]


def serialize_solution(
    values: FloatArray,
    case: MatpowerCase,
    network: NetworkData,
    master: MasterModel,
) -> dict[str, Any]:
    generators: list[dict[str, Any]] = []
    for generator_index, source in enumerate(case.gen):
        online = source[GEN_STATUS] > 0
        if online:
            commitment = float(values[master.index.commitment_by_generator[generator_index]])
            dispatch = float(values[master.index.dispatch_by_generator[generator_index]])
            segments = [
                float(values[column])
                for column in master.index.segments_by_generator[generator_index]
            ]
        else:
            commitment = 0.0
            dispatch = 0.0
            segments = []
        generators.append(
            {
                "source_id": generator_source_id(generator_index),
                "source_row": generator_index + 1,
                "source_status": int(source[GEN_STATUS]),
                "commitment": commitment,
                "dispatch_mw": dispatch,
                "segment_dispatch_mw": segments,
            }
        )
    return {
        "generators": generators,
        "bus_angles_rad": [
            {
                "bus": int(bus_id),
                "angle_rad": float(values[master.index.theta_by_bus[bus_index]]),
            }
            for bus_index, bus_id in enumerate(network.bus_ids)
        ],
        "base_branch_flows_mw": [
            {
                "source_id": branch_source_id(int(source_index)),
                "source_row": int(source_index) + 1,
                "flow_mw": float(values[master.index.flow_by_active_branch[active_index]]),
            }
            for active_index, source_index in enumerate(network.active_branch_source_rows)
        ],
    }


def base_flow_vector(solution: dict[str, Any], network: NetworkData) -> FloatArray:
    records = solution["base_branch_flows_mw"]
    by_row = {int(record["source_row"]): float(record["flow_mw"]) for record in records}
    if len(by_row) != len(records):
        raise ValueError("Duplicate branch source rows in serialized solution")
    return np.asarray(
        [by_row[int(source_index) + 1] for source_index in network.active_branch_source_rows],
        dtype=np.float64,
    )

