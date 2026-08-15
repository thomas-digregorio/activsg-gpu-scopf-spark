"""Common result contract and lazy solver dispatch."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ..canonical import CanonicalMILP
from ..errors import ScopfError

FloatArray = npt.NDArray[np.float64]
DiagnosticEvent = Callable[..., None]
NO_MIP_START_PRECHECK = "none"
HIGHS_FIXED_COMMITMENT_PRECHECK = "highs_fixed_commitment_lp_v1"
SUPPORTED_MIP_START_PRECHECKS = frozenset(
    {NO_MIP_START_PRECHECK, HIGHS_FIXED_COMMITMENT_PRECHECK}
)


@dataclass(frozen=True)
class SolveResult:
    solver: str
    solver_version: str
    status: str
    optimal: bool
    requested_gap_certified: bool
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

    def solve(self, *, time_limit_seconds: float | None) -> SolveResult: ...


@dataclass
class RebuildingSolverSession:
    """Rebuild each round while retaining the prior integer solution as a MIP start."""

    model: CanonicalMILP
    solver: str
    mip_relative_gap: float
    threads: int
    mip_acceptance_policy: str = "native_optimal_only"
    mip_certificate_residual_tolerance: float = 1e-6
    native_scaling_mode: str = "none"
    native_base_mva: float = 100.0
    log_to_console: bool = False
    cuopt_pdlp_profile: dict[str, Any] = field(default_factory=dict)
    track_incumbent_commitments: bool = False
    mip_start_precheck: str = NO_MIP_START_PRECHECK
    mip_start_precheck_time_limit_seconds: float = 10.0
    mode: str = field(init=False)
    previous_values: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.mip_start_precheck not in SUPPORTED_MIP_START_PRECHECKS:
            raise ScopfError(
                f"Unknown MIP-start precheck policy: {self.mip_start_precheck!r}"
            )
        if self.mip_start_precheck_time_limit_seconds <= 0.0:
            raise ScopfError("MIP-start precheck time limit must be positive")
        self.mode = (
            "rebuild_each_round_with_feasibility_checked_mip_start"
            if self.mip_start_precheck != NO_MIP_START_PRECHECK
            else "rebuild_each_round_with_partial_mip_start"
        )

    def solve(self, *, time_limit_seconds: float | None) -> SolveResult:
        if time_limit_seconds is None:
            raise ScopfError(f"Unbounded solves are not implemented for {self.solver}")
        mip_start_values = self.previous_values
        mip_start_mode = "integer_only"
        clip_mip_start_to_bounds = False
        precheck: dict[str, Any] | None = None
        solve_budget = float(time_limit_seconds)
        if (
            mip_start_values is not None
            and self.mip_start_precheck == HIGHS_FIXED_COMMITMENT_PRECHECK
        ):
            from .highs import complete_fixed_integer_start

            precheck_started = time.perf_counter()
            precheck_budget = min(
                solve_budget, self.mip_start_precheck_time_limit_seconds
            )
            completion = complete_fixed_integer_start(
                self.model,
                mip_start_values,
                time_limit_seconds=precheck_budget,
                threads=self.threads,
            )
            precheck_wall = time.perf_counter() - precheck_started
            solve_budget -= precheck_wall
            if solve_budget <= 0.0:
                raise ScopfError(
                    "MIP-start feasibility precheck exhausted the solver budget"
                )
            completion_values = completion.values
            maximum_row_violation = (
                None
                if completion_values is None
                else self.model.max_row_violation(completion_values)
            )
            _, column_lower, column_upper, integrality = (
                self.model.column_arrays()
            )
            maximum_column_violation = (
                None
                if completion_values is None
                else float(
                    max(
                        np.max(column_lower - completion_values),
                        np.max(completion_values - column_upper),
                        0.0,
                    )
                )
            )
            integer_columns = np.flatnonzero(integrality)
            maximum_commitment_difference = (
                None
                if completion_values is None
                else 0.0
                if not integer_columns.size
                else float(
                    np.max(
                        np.abs(
                            completion_values[integer_columns]
                            - np.rint(mip_start_values[integer_columns])
                        )
                    )
                )
            )
            completion_feasible = bool(
                completion.has_incumbent
                and completion_values is not None
                and maximum_row_violation is not None
                and maximum_row_violation <= 1e-6
                and maximum_column_violation is not None
                and maximum_column_violation <= 1e-6
                and maximum_commitment_difference is not None
                and maximum_commitment_difference <= 1e-6
            )
            precheck = {
                "policy": self.mip_start_precheck,
                "time_limit_seconds": precheck_budget,
                "wall_time_seconds": precheck_wall,
                "solver_status": completion.status,
                "has_incumbent": completion.has_incumbent,
                "maximum_canonical_row_violation": maximum_row_violation,
                "maximum_canonical_column_violation": (
                    maximum_column_violation
                ),
                "maximum_prior_commitment_difference": (
                    maximum_commitment_difference
                ),
                "prior_commitment_extendable": completion_feasible,
                "decision": (
                    "submit_completed_full_start"
                    if completion_feasible
                    else "solve_cold"
                ),
            }
            if completion_feasible:
                mip_start_values = completion_values
                mip_start_mode = "all_columns"
                clip_mip_start_to_bounds = True
            else:
                mip_start_values = None
        result = solve_canonical(
            self.model,
            solver=self.solver,
            time_limit_seconds=solve_budget,
            mip_relative_gap=self.mip_relative_gap,
            threads=self.threads,
            mip_start_values=mip_start_values,
            mip_start_mode=mip_start_mode,
            clip_mip_start_to_bounds=clip_mip_start_to_bounds,
            mip_acceptance_policy=self.mip_acceptance_policy,
            mip_certificate_residual_tolerance=(
                self.mip_certificate_residual_tolerance
            ),
            native_scaling_mode=self.native_scaling_mode,
            native_base_mva=self.native_base_mva,
            log_to_console=self.log_to_console,
            cuopt_pdlp_profile=self.cuopt_pdlp_profile,
            track_incumbent_commitments=self.track_incumbent_commitments,
        )
        if precheck is not None:
            result.statistics["mip_start_feasibility_precheck"] = precheck
        if result.values is not None:
            self.previous_values = result.values.copy()
        return result


def create_solver_session(
    model: CanonicalMILP,
    *,
    solver: str,
    mip_relative_gap: float,
    threads: int = 0,
    diagnostic_event: DiagnosticEvent | None = None,
    native_log_path: Path | None = None,
    mip_logging_interval_seconds: float = 5.0,
    mip_acceptance_policy: str = "native_optimal_only",
    mip_certificate_residual_tolerance: float = 1e-6,
    native_scaling_mode: str = "none",
    native_base_mva: float = 100.0,
    log_to_console: bool = False,
    cuopt_pdlp_profile: dict[str, Any] | None = None,
    track_incumbent_commitments: bool = False,
    mip_start_precheck: str = NO_MIP_START_PRECHECK,
    mip_start_precheck_time_limit_seconds: float = 10.0,
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
        return RebuildingSolverSession(
            model,
            solver,
            mip_relative_gap,
            threads,
            mip_acceptance_policy=mip_acceptance_policy,
            mip_certificate_residual_tolerance=mip_certificate_residual_tolerance,
            native_scaling_mode=native_scaling_mode,
            native_base_mva=native_base_mva,
            log_to_console=log_to_console,
            cuopt_pdlp_profile=dict(cuopt_pdlp_profile or {}),
            track_incumbent_commitments=track_incumbent_commitments,
            mip_start_precheck=mip_start_precheck,
            mip_start_precheck_time_limit_seconds=(
                mip_start_precheck_time_limit_seconds
            ),
        )
    raise ScopfError(f"Unknown canonical solver adapter: {solver}")


def solve_canonical(
    model: CanonicalMILP,
    *,
    solver: str,
    time_limit_seconds: float,
    mip_relative_gap: float,
    threads: int = 0,
    mip_start_values: FloatArray | None = None,
    mip_start_mode: str = "integer_only",
    clip_mip_start_to_bounds: bool = False,
    mip_acceptance_policy: str = "native_optimal_only",
    mip_certificate_residual_tolerance: float = 1e-6,
    native_scaling_mode: str = "none",
    native_base_mva: float = 100.0,
    log_to_console: bool = False,
    cuopt_pdlp_profile: dict[str, Any] | None = None,
    track_incumbent_commitments: bool = False,
) -> SolveResult:
    if time_limit_seconds <= 0:
        raise ScopfError("Solver was not started because no deadline budget remained")
    if solver == "highs":
        if mip_start_values is not None:
            raise ScopfError("Direct HiGHS solves do not accept external MIP starts")
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
            mip_start_values=mip_start_values,
            mip_start_mode=mip_start_mode,
            clip_mip_start_to_bounds=clip_mip_start_to_bounds,
            mip_acceptance_policy=mip_acceptance_policy,
            mip_certificate_residual_tolerance=mip_certificate_residual_tolerance,
            native_scaling_mode=native_scaling_mode,
            native_base_mva=native_base_mva,
            log_to_console=log_to_console,
            cuopt_pdlp_profile=cuopt_pdlp_profile,
            track_incumbent_commitments=track_incumbent_commitments,
        )
    raise ScopfError(f"Unknown canonical solver adapter: {solver}")
