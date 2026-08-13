"""Common result contract and lazy solver dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ..canonical import CanonicalMILP
from ..errors import ScopfError

FloatArray = npt.NDArray[np.float64]
DiagnosticEvent = Callable[..., None]


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


class SolverSession(Protocol):
    """A solver adapter that observes append-only changes to one canonical model."""

    mode: str

    def solve(self, *, time_limit_seconds: float) -> SolveResult: ...


@dataclass
class RebuildingSolverSession:
    """Compatibility session for adapters without registered incremental support."""

    model: CanonicalMILP
    solver: str
    mip_relative_gap: float
    threads: int
    mode: str = "rebuild_each_round"

    def solve(self, *, time_limit_seconds: float) -> SolveResult:
        return solve_canonical(
            self.model,
            solver=self.solver,
            time_limit_seconds=time_limit_seconds,
            mip_relative_gap=self.mip_relative_gap,
            threads=self.threads,
        )


def create_solver_session(
    model: CanonicalMILP,
    *,
    solver: str,
    mip_relative_gap: float,
    threads: int = 0,
    diagnostic_event: DiagnosticEvent | None = None,
    native_log_path: Path | None = None,
    mip_logging_interval_seconds: float = 5.0,
) -> SolverSession:
    if solver == "highs":
        from .highs import HighsSession

        return HighsSession(
            model,
            mip_relative_gap=mip_relative_gap,
            threads=threads,
            diagnostic_event=diagnostic_event,
            native_log_path=native_log_path,
            mip_logging_interval_seconds=mip_logging_interval_seconds,
        )
    if solver == "cuopt":
        return RebuildingSolverSession(model, solver, mip_relative_gap, threads)
    raise ScopfError(f"Unknown canonical solver adapter: {solver}")


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
