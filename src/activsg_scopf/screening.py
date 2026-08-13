"""Deterministic exhaustive branch-pair screening on NumPy or CuPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .canonical import CanonicalMILP
from .errors import ScopfError
from .model import ModelIndex
from .network import ContingencyCatalog, NetworkData

FloatArray = npt.NDArray[np.float64]
Side = Literal["lower", "upper"]


@dataclass(frozen=True, order=True)
class SecurityPair:
    contingency_label: int
    monitored_branch_source_row: int
    side: Side
    outage_column: int
    monitored_active_index: int

    @property
    def pair_id(self) -> str:
        return (
            f"c{self.contingency_label:04d}_"
            f"m{self.monitored_branch_source_row:04d}_{self.side}"
        )


@dataclass(frozen=True)
class ScreenResult:
    violations: tuple[SecurityPair, ...]
    maximum_violation_pu: float
    maximum_pair_id: str | None
    evaluated_pairs: int


def screen_contingencies(
    base_flow_mw: FloatArray,
    network: NetworkData,
    catalog: ContingencyCatalog,
    *,
    tolerance_pu: float,
    backend: str,
    already_added: set[str] | None = None,
) -> ScreenResult:
    if backend == "numpy":
        xp = np
    elif backend == "cupy":
        try:
            import cupy as xp
        except ImportError as exc:
            raise ScopfError("CuPy screening was requested but CuPy is unavailable") from exc
    else:
        raise ScopfError(f"Unknown contingency screening backend: {backend}")
    flow = xp.asarray(base_flow_mw, dtype=xp.float64)
    lodf = xp.asarray(catalog.lodf, dtype=xp.float64)
    outage_indices_host = np.asarray(
        [item.active_branch_index for item in catalog.valid], dtype=np.int64
    )
    outage_indices = xp.asarray(outage_indices_host)
    post = flow[:, None] + lodf * flow[outage_indices][None, :]
    rates = xp.asarray(network.rate_a_mw, dtype=xp.float64)[:, None]
    eligible = rates > 0
    if post.size:
        eligible = xp.broadcast_to(eligible, post.shape).copy()
        eligible[outage_indices, xp.arange(len(catalog.valid))] = False
    upper_excess = xp.where(eligible, post - rates, -xp.inf)
    lower_excess = xp.where(eligible, -rates - post, -xp.inf)
    threshold_mw = tolerance_pu * network.base_mva
    upper_locations = xp.argwhere(upper_excess > threshold_mw)
    lower_locations = xp.argwhere(lower_excess > threshold_mw)
    if backend == "cupy":
        import cupy as cp

        upper_host = cp.asnumpy(upper_locations)
        lower_host = cp.asnumpy(lower_locations)
        upper_excess_host = cp.asnumpy(upper_excess)
        lower_excess_host = cp.asnumpy(lower_excess)
    else:
        upper_host = np.asarray(upper_locations)
        lower_host = np.asarray(lower_locations)
        upper_excess_host = np.asarray(upper_excess)
        lower_excess_host = np.asarray(lower_excess)
    pairs: list[SecurityPair] = []
    for side, locations in (("lower", lower_host), ("upper", upper_host)):
        for monitored, outage_column in locations:
            outage = catalog.valid[int(outage_column)]
            pair = SecurityPair(
                contingency_label=outage.contingency_label,
                monitored_branch_source_row=int(
                    network.active_branch_source_rows[int(monitored)] + 1
                ),
                side=side,
                outage_column=int(outage_column),
                monitored_active_index=int(monitored),
            )
            if already_added is None or pair.pair_id not in already_added:
                pairs.append(pair)
    pairs.sort(
        key=lambda item: (
            item.contingency_label,
            item.monitored_branch_source_row,
            item.side,
        )
    )
    maximum = 0.0
    maximum_id: str | None = None
    for side, excess in (("lower", lower_excess_host), ("upper", upper_excess_host)):
        if excess.size == 0:
            continue
        flat_index = int(np.argmax(excess))
        value = float(excess.ravel()[flat_index])
        if value > maximum:
            monitored, outage_column = np.unravel_index(flat_index, excess.shape)
            outage = catalog.valid[int(outage_column)]
            maximum = value
            maximum_id = (
                f"c{outage.contingency_label:04d}_"
                f"m{int(network.active_branch_source_rows[monitored]) + 1:04d}_{side}"
            )
    evaluated = int(
        sum(
            len(network.active_branch_source_rows) - 1
            for _ in catalog.valid
            if len(network.active_branch_source_rows) > 1
        )
        * 2
    )
    return ScreenResult(
        violations=tuple(pairs),
        maximum_violation_pu=max(maximum, 0.0) / network.base_mva,
        maximum_pair_id=maximum_id,
        evaluated_pairs=evaluated,
    )


def add_security_pairs(
    model: CanonicalMILP,
    index: ModelIndex,
    network: NetworkData,
    catalog: ContingencyCatalog,
    pairs: tuple[SecurityPair, ...],
) -> None:
    for pair in pairs:
        outage = catalog.valid[pair.outage_column]
        monitored = pair.monitored_active_index
        outaged = outage.active_branch_index
        coefficients = {int(index.flow_by_active_branch[monitored]): 1.0}
        lodf = float(catalog.lodf[monitored, pair.outage_column])
        outage_column = int(index.flow_by_active_branch[outaged])
        coefficients[outage_column] = coefficients.get(outage_column, 0.0) + lodf
        rate = float(network.rate_a_mw[monitored])
        if pair.side == "lower":
            model.add_row(pair.pair_id, coefficients, lower=-rate)
        else:
            model.add_row(pair.pair_id, coefficients, upper=rate)
