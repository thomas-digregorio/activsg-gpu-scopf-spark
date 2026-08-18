"""Deterministic exhaustive branch-pair screening on NumPy or CuPy."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    outage_active_index: int = field(compare=False)
    lodf_value: float = field(compare=False)

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


class ContingencyScreener:
    """Reusable chunked NumPy/CuPy screener with one persistent LODF transfer."""

    def __init__(
        self,
        network: NetworkData,
        catalog: ContingencyCatalog,
        *,
        backend: str,
        chunk_columns: int = 256,
    ) -> None:
        if chunk_columns <= 0:
            raise ScopfError("screen chunk_columns must be positive")
        if backend == "numpy":
            xp = np
        elif backend == "cupy":
            try:
                import cupy as xp
            except ImportError as exc:
                raise ScopfError("CuPy screening was requested but CuPy is unavailable") from exc
        else:
            raise ScopfError(f"Unknown contingency screening backend: {backend}")
        self.network = network
        self.catalog = catalog
        self.backend = backend
        self.xp = xp
        self.chunk_columns = chunk_columns
        self.outage_indices_host = np.asarray(
            [item.active_branch_index for item in catalog.valid], dtype=np.int64
        )
        self.rates = xp.asarray(network.rate_a_mw, dtype=xp.float64)[:, None]
        self.rate_limited_host = network.rate_a_mw > 0
        self.lodf = (
            None
            if catalog.lodf is None
            else xp.asarray(catalog.lodf, dtype=xp.float64)
        )

    def _host(self, value: object) -> np.ndarray:
        if self.backend == "cupy":
            import cupy as cp

            return cp.asnumpy(value)
        return np.asarray(value)

    def screen(
        self,
        base_flow_mw: FloatArray,
        *,
        tolerance_pu: float,
        already_added: set[str] | None = None,
    ) -> ScreenResult:
        xp = self.xp
        flow = xp.asarray(base_flow_mw, dtype=xp.float64)
        threshold_mw = tolerance_pu * self.network.base_mva
        pairs: list[SecurityPair] = []
        maximum = 0.0
        maximum_id: str | None = None
        for start in range(0, len(self.catalog.valid), self.chunk_columns):
            stop = min(start + self.chunk_columns, len(self.catalog.valid))
            outage_indices_host = self.outage_indices_host[start:stop]
            outage_indices = xp.asarray(outage_indices_host)
            if self.lodf is None:
                lodf_host = self.catalog.lodf_operator.lodf_columns(
                    outage_indices_host
                )
                lodf_chunk = xp.asarray(lodf_host, dtype=xp.float64)
            else:
                lodf_chunk = self.lodf[:, start:stop]
            post = flow[:, None] + lodf_chunk * flow[outage_indices][None, :]
            eligible = xp.broadcast_to(self.rates > 0, post.shape).copy()
            eligible[outage_indices, xp.arange(stop - start)] = False
            for side, excess in (
                ("lower", xp.where(eligible, -self.rates - post, -xp.inf)),
                ("upper", xp.where(eligible, post - self.rates, -xp.inf)),
            ):
                locations = xp.argwhere(excess > threshold_mw)
                if int(locations.shape[0]):
                    selected_lodf = lodf_chunk[locations[:, 0], locations[:, 1]]
                    locations_host = self._host(locations)
                    selected_lodf_host = self._host(selected_lodf)
                    for location, lodf_value in zip(
                        locations_host, selected_lodf_host, strict=True
                    ):
                        monitored = int(location[0])
                        outage_column = start + int(location[1])
                        outage = self.catalog.valid[outage_column]
                        pair = SecurityPair(
                            contingency_label=outage.contingency_label,
                            monitored_branch_source_row=int(
                                self.network.active_branch_source_rows[monitored] + 1
                            ),
                            side=side,
                            outage_column=outage_column,
                            monitored_active_index=monitored,
                            outage_active_index=outage.active_branch_index,
                            lodf_value=float(lodf_value),
                        )
                        if already_added is None or pair.pair_id not in already_added:
                            pairs.append(pair)
                if excess.size:
                    chunk_max = float(xp.max(excess).item())
                    if chunk_max > maximum:
                        flat_index = int(xp.argmax(excess).item())
                        monitored, local_column = np.unravel_index(
                            flat_index, excess.shape
                        )
                        outage = self.catalog.valid[start + int(local_column)]
                        maximum = chunk_max
                        maximum_id = (
                            f"c{outage.contingency_label:04d}_"
                            f"m{int(self.network.active_branch_source_rows[monitored]) + 1:04d}_"
                            f"{side}"
                        )
        pairs.sort()
        limited_count = int(np.count_nonzero(self.rate_limited_host))
        evaluated = 2 * sum(
            limited_count - int(self.rate_limited_host[outage.active_branch_index])
            for outage in self.catalog.valid
        )
        return ScreenResult(
            violations=tuple(pairs),
            maximum_violation_pu=max(maximum, 0.0) / self.network.base_mva,
            maximum_pair_id=maximum_id,
            evaluated_pairs=int(evaluated),
        )


def screen_contingencies(
    base_flow_mw: FloatArray,
    network: NetworkData,
    catalog: ContingencyCatalog,
    *,
    tolerance_pu: float,
    backend: str,
    already_added: set[str] | None = None,
    chunk_columns: int = 256,
) -> ScreenResult:
    screener = ContingencyScreener(
        network, catalog, backend=backend, chunk_columns=chunk_columns
    )
    return screener.screen(
        base_flow_mw, tolerance_pu=tolerance_pu, already_added=already_added
    )


def add_security_pairs(
    model: CanonicalMILP,
    index: ModelIndex,
    network: NetworkData,
    pairs: tuple[SecurityPair, ...],
) -> None:
    for pair in pairs:
        monitored = pair.monitored_active_index
        outaged = pair.outage_active_index
        coefficients = {int(index.flow_by_active_branch[monitored]): 1.0}
        lodf = pair.lodf_value
        outage_column = int(index.flow_by_active_branch[outaged])
        coefficients[outage_column] = coefficients.get(outage_column, 0.0) + lodf
        rate = float(network.rate_a_mw[monitored])
        if pair.side == "lower":
            model.add_row(pair.pair_id, coefficients, lower=-rate)
        else:
            model.add_row(pair.pair_id, coefficients, upper=rate)
