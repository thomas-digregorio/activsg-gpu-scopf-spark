"""HiGHS adapter for the solver-neutral canonical model."""

from __future__ import annotations

import numpy as np

from ..canonical import CanonicalMILP
from ..errors import ScopfError
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


def solve_highs(
    model: CanonicalMILP,
    *,
    time_limit_seconds: float,
    mip_relative_gap: float,
    threads: int = 0,
) -> SolveResult:
    import highspy

    highs = highspy.Highs()
    _require_ok(highs.setOptionValue("output_flag", False), "output_flag configuration")
    _require_ok(highs.setOptionValue("time_limit", float(time_limit_seconds)), "time limit")
    _require_ok(highs.setOptionValue("mip_rel_gap", float(mip_relative_gap)), "MIP gap")
    _require_ok(highs.setOptionValue("random_seed", 0), "random seed")
    if threads > 0:
        _require_ok(highs.setOptionValue("threads", int(threads)), "thread count")
    objective, column_lower, column_upper, integrality = model.column_arrays()
    column_indices = np.arange(model.num_columns, dtype=np.int32)
    _require_ok(
        highs.addVars(model.num_columns, column_lower, column_upper),
        "variable loading",
    )
    _require_ok(
        highs.changeColsCost(model.num_columns, column_indices, objective),
        "objective loading",
    )
    integer_columns = np.flatnonzero(integrality).astype(np.int32)
    if integer_columns.size:
        types = np.full(integer_columns.size, highspy.HighsVarType.kInteger.value, dtype=np.uint8)
        _require_ok(
            highs.changeColsIntegrality(integer_columns.size, integer_columns, types),
            "integrality loading",
        )
    matrix = model.matrix_csr()
    row_lower, row_upper = model.row_bound_arrays()
    _require_ok(
        highs.addRows(
            model.num_rows,
            row_lower,
            row_upper,
            matrix.nnz,
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            matrix.data,
        ),
        "constraint loading",
    )
    run_return_status = highs.run()
    _require_run_not_error(run_return_status)
    status = highs.getModelStatus()
    info = highs.getInfo()
    solution = highs.getSolution()
    has_incumbent = bool(solution.value_valid and np.isfinite(info.objective_function_value))
    values = np.asarray(solution.col_value, dtype=np.float64) if has_incumbent else None
    objective_value = float(info.objective_function_value) if has_incumbent else None
    bound = float(info.mip_dual_bound) if np.isfinite(info.mip_dual_bound) else None
    gap = float(info.mip_gap) if np.isfinite(info.mip_gap) else None
    return SolveResult(
        solver="highs",
        solver_version=highs.version(),
        status=status.name.removeprefix("k"),
        optimal=status == highspy.HighsModelStatus.kOptimal,
        has_incumbent=has_incumbent,
        objective=objective_value,
        bound=bound,
        mip_gap=gap,
        solve_time_seconds=float(highs.getRunTime()),
        values=values,
        statistics={
            "run_return_status": run_return_status.name.removeprefix("k"),
            "mip_node_count": int(info.mip_node_count),
            "max_integrality_violation": float(info.max_integrality_violation),
            "max_primal_infeasibility": float(info.max_primal_infeasibility),
            "simplex_iteration_count": int(info.simplex_iteration_count),
        },
    )
