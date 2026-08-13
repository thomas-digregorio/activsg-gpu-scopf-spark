"""Common result contract and lazy solver dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..canonical import CanonicalMILP
from ..errors import ScopfError

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SolveResult:
    solver: str
    solver_version: str
    status: str
    optimal: bool
    has_incumbent: bool
    objective: float | None
    bound: float | None
    mip_gap: float | None
    solve_time_seconds: float
    values: FloatArray | None
    statistics: dict[str, Any]


def solve_canonical(
    model: CanonicalMILP,
    *,
    solver: str,
    time_limit_seconds: float,
    mip_relative_gap: float,
    threads: int = 0,
) -> SolveResult:
    if time_limit_seconds <= 0:
        raise ScopfError("Solver was not started because no deadline budget remained")
    if solver == "highs":
        from .highs import solve_highs

        return solve_highs(
            model,
            time_limit_seconds=time_limit_seconds,
            mip_relative_gap=mip_relative_gap,
            threads=threads,
        )
    if solver == "cuopt":
        from .cuopt import solve_cuopt

        return solve_cuopt(
            model,
            time_limit_seconds=time_limit_seconds,
            mip_relative_gap=mip_relative_gap,
            threads=threads,
        )
    raise ScopfError(f"Unknown canonical solver adapter: {solver}")

