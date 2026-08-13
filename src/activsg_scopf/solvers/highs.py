"""HiGHS adapter for the solver-neutral canonical model."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..canonical import CanonicalMILP
from ..errors import ScopfError
from ..paths import guard_output_path
from .common import SolveResult


def _require_ok(status: object, operation: str) -> None:
    import highspy

    if status != highspy.HighsStatus.kOk:
        raise ScopfError(f"HiGHS {operation} failed with {status}")


def _require_run_not_error(status: object) -> None:
    """Allow kWarning so model status/incumbent evidence can still be extracted."""

    import highspy

    if status == highspy.HighsStatus.kError:
        raise ScopfError(f"HiGHS solve failed with {status}")


class HighsSession:
    """Persistent HiGHS model with incremental row loading and partial MIP starts."""

    mode = "persistent_incremental"

    def __init__(
        self,
        model: CanonicalMILP,
        *,
        mip_relative_gap: float,
        threads: int = 0,
        diagnostic_event: Callable[..., None] | None = None,
        native_log_path: Path | None = None,
        mip_logging_interval_seconds: float = 5.0,
    ) -> None:
        import highspy

        self.model = model
        self.highs = highspy.Highs()
        self.loaded_columns = model.num_columns
        self.loaded_rows = 0
        self.solve_count = 0
        self.previous_values: np.ndarray | None = None
        self.last_row_duals: np.ndarray | None = None
        self.last_column_duals: np.ndarray | None = None
        self.diagnostic_event = diagnostic_event
        self._active_solve_number = 0
        output_enabled = diagnostic_event is not None or native_log_path is not None
        _require_ok(
            self.highs.setOptionValue("output_flag", output_enabled),
            "output_flag configuration",
        )
        if output_enabled:
            _require_ok(
                self.highs.setOptionValue("log_to_console", False),
                "console logging configuration",
            )
            _require_ok(
                self.highs.setOptionValue(
                    "mip_min_logging_interval",
                    float(mip_logging_interval_seconds),
                ),
                "MIP logging interval",
            )
            _require_ok(
                self.highs.setOptionValue("mip_report_level", 2),
                "MIP report level",
            )
        if native_log_path is not None:
            native_log_path = guard_output_path(native_log_path)
            native_log_path.parent.mkdir(parents=True, exist_ok=True)
            _require_ok(
                self.highs.setOptionValue("log_file", str(native_log_path)),
                "native log file",
            )
        if diagnostic_event is not None:
            self.highs.cbLogging.subscribe(self._record_native_log)
            self.highs.cbMipLogging.subscribe(self._record_mip_progress)
        _require_ok(
            self.highs.setOptionValue("mip_rel_gap", float(mip_relative_gap)),
            "MIP gap",
        )
        _require_ok(self.highs.setOptionValue("random_seed", 0), "random seed")
        if threads > 0:
            _require_ok(
                self.highs.setOptionValue("threads", int(threads)), "thread count"
            )
        objective, column_lower, column_upper, integrality = model.column_arrays()
        self.integer_columns = np.flatnonzero(integrality).astype(np.int32)
        column_indices = np.arange(model.num_columns, dtype=np.int32)
        _require_ok(
            self.highs.addVars(model.num_columns, column_lower, column_upper),
            "variable loading",
        )
        _require_ok(
            self.highs.changeColsCost(model.num_columns, column_indices, objective),
            "objective loading",
        )
        if self.integer_columns.size:
            types = np.full(
                self.integer_columns.size,
                highspy.HighsVarType.kInteger.value,
                dtype=np.uint8,
            )
            _require_ok(
                self.highs.changeColsIntegrality(
                    self.integer_columns.size, self.integer_columns, types
                ),
                "integrality loading",
            )
        self._sync_rows()

    def _emit(self, event: str, **fields: Any) -> None:
        if self.diagnostic_event is not None:
            self.diagnostic_event(event, **fields)

    def _record_native_log(self, callback_event: Any) -> None:
        message = str(callback_event.message).strip()
        if message:
            self._emit(
                "highs_native_log",
                solve_number=self._active_solve_number or None,
                message=message,
            )

    def _record_mip_progress(self, callback_event: Any) -> None:
        data = callback_event.data_out
        self._emit(
            "highs_mip_progress",
            solve_number=self._active_solve_number,
            highs_running_time_seconds=float(data.running_time),
            mip_node_count=int(data.mip_node_count),
            mip_primal_bound=float(data.mip_primal_bound),
            mip_dual_bound=float(data.mip_dual_bound),
            mip_gap=float(data.mip_gap),
            simplex_iteration_count=int(data.simplex_iteration_count),
        )

    def _sync_rows(self) -> tuple[int, float]:
        if self.model.num_columns != self.loaded_columns:
            raise ScopfError("Persistent HiGHS sessions support appended rows only")
        rows_added = self.model.num_rows - self.loaded_rows
        if rows_added < 0:
            raise ScopfError("Canonical rows cannot be removed from a persistent session")
        if rows_added == 0:
            return 0, 0.0
        started = time.perf_counter()
        matrix = self.model.matrix_csr()[self.loaded_rows : self.model.num_rows].tocsr()
        row_lower, row_upper = self.model.row_bound_arrays()
        _require_ok(
            self.highs.addRows(
                rows_added,
                row_lower[self.loaded_rows :],
                row_upper[self.loaded_rows :],
                matrix.nnz,
                matrix.indptr.astype(np.int32),
                matrix.indices.astype(np.int32),
                matrix.data,
            ),
            "incremental constraint loading",
        )
        self.loaded_rows = self.model.num_rows
        return rows_added, time.perf_counter() - started

    def solve(self, *, time_limit_seconds: float | None) -> SolveResult:
        import highspy

        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ScopfError("Solver was not started because no deadline budget remained")
        call_started = time.perf_counter()
        solve_number = self.solve_count + 1
        self._active_solve_number = solve_number
        self._emit(
            "highs_row_sync_started",
            solve_number=solve_number,
            loaded_rows=self.loaded_rows,
            canonical_rows=self.model.num_rows,
        )
        rows_added, row_add_time = self._sync_rows()
        self._emit(
            "highs_row_sync_finished",
            solve_number=solve_number,
            rows_added=rows_added,
            wall_time_seconds=row_add_time,
        )
        mip_start_status: str | None = None
        if (
            rows_added > 0
            and self.previous_values is not None
            and self.integer_columns.size
        ):
            mip_start_started = time.perf_counter()
            self._emit(
                "highs_mip_start_started",
                solve_number=solve_number,
                integer_columns=int(self.integer_columns.size),
            )
            start_values = np.rint(self.previous_values[self.integer_columns])
            status = self.highs.setSolution(
                self.integer_columns.size, self.integer_columns, start_values
            )
            _require_run_not_error(status)
            mip_start_status = status.name.removeprefix("k")
            self._emit(
                "highs_mip_start_finished",
                solve_number=solve_number,
                return_status=mip_start_status,
                wall_time_seconds=time.perf_counter() - mip_start_started,
            )
        setup_time = time.perf_counter() - call_started
        native_time_limit = (
            None if time_limit_seconds is None else float(time_limit_seconds) - setup_time
        )
        if native_time_limit is not None and native_time_limit <= 0:
            raise ScopfError(
                "Solver was not started because incremental session setup exhausted "
                "the remaining deadline budget"
            )
        highs_run_time_before = float(self.highs.getRunTime())
        if native_time_limit is not None:
            _require_ok(
                self.highs.setOptionValue("time_limit", native_time_limit),
                "time limit",
            )
        self._emit(
            "highs_run_started",
            solve_number=solve_number,
            requested_call_budget_seconds=(
                None if time_limit_seconds is None else float(time_limit_seconds)
            ),
            native_time_limit_seconds=native_time_limit,
            session_setup_wall_time_seconds=setup_time,
            highs_run_time_before_seconds=highs_run_time_before,
        )
        run_started = time.perf_counter()
        run_return_status = self.highs.run()
        run_wall_time = time.perf_counter() - run_started
        _require_run_not_error(run_return_status)
        highs_run_time_after = float(self.highs.getRunTime())
        status = self.highs.getModelStatus()
        info = self.highs.getInfo()
        solution = self.highs.getSolution()
        has_incumbent = bool(
            solution.value_valid and np.isfinite(info.objective_function_value)
        )
        values = (
            np.asarray(solution.col_value, dtype=np.float64) if has_incumbent else None
        )
        if values is not None:
            self.previous_values = values.copy()
        self.last_row_duals = (
            np.asarray(solution.row_dual, dtype=np.float64)
            if solution.dual_valid
            else None
        )
        self.last_column_duals = (
            np.asarray(solution.col_dual, dtype=np.float64)
            if solution.dual_valid
            else None
        )
        objective_value = float(info.objective_function_value) if has_incumbent else None
        bound = float(info.mip_dual_bound) if np.isfinite(info.mip_dual_bound) else None
        gap = float(info.mip_gap) if np.isfinite(info.mip_gap) else None
        self.solve_count += 1
        self._emit(
            "highs_run_finished",
            solve_number=solve_number,
            run_return_status=run_return_status.name.removeprefix("k"),
            model_status=status.name.removeprefix("k"),
            wall_time_seconds=run_wall_time,
            has_incumbent=has_incumbent,
            objective=objective_value,
            bound=bound,
            mip_gap=gap,
            mip_node_count=int(info.mip_node_count),
            simplex_iteration_count=int(info.simplex_iteration_count),
        )
        return SolveResult(
            solver="highs",
            solver_version=self.highs.version(),
            status=status.name.removeprefix("k"),
            optimal=status == highspy.HighsModelStatus.kOptimal,
            has_incumbent=has_incumbent,
            objective=objective_value,
            bound=bound,
            mip_gap=gap,
            solve_time_seconds=run_wall_time,
            values=values,
            statistics={
                "session_mode": self.mode,
                "session_solve_number": self.solve_count,
                "run_return_status": run_return_status.name.removeprefix("k"),
                "incremental_rows_added": rows_added,
                "incremental_row_load_wall_time_seconds": row_add_time,
                "partial_integer_mip_start_return_status": mip_start_status,
                "requested_call_budget_seconds": (
                    None if time_limit_seconds is None else float(time_limit_seconds)
                ),
                "native_time_limit_seconds": native_time_limit,
                "session_setup_wall_time_seconds": setup_time,
                "highs_run_time_before_seconds": highs_run_time_before,
                "highs_run_time_after_seconds": highs_run_time_after,
                "mip_node_count": int(info.mip_node_count),
                "max_integrality_violation": float(info.max_integrality_violation),
                "max_primal_infeasibility": float(info.max_primal_infeasibility),
                "simplex_iteration_count": int(info.simplex_iteration_count),
            },
        )


def solve_highs(
    model: CanonicalMILP,
    *,
    time_limit_seconds: float,
    mip_relative_gap: float,
    threads: int = 0,
) -> SolveResult:
    return HighsSession(
        model, mip_relative_gap=mip_relative_gap, threads=threads
    ).solve(time_limit_seconds=time_limit_seconds)
