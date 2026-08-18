"""Solver-neutral sparse MILP representation shared by both adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import inf

import numpy as np
import numpy.typing as npt
from scipy import sparse

from .errors import ScopfError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]


@dataclass
class CanonicalMILP:
    """Append-only sparse minimization model with ranged rows."""

    variable_names: list[str] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    column_lower: list[float] = field(default_factory=list)
    column_upper: list[float] = field(default_factory=list)
    integrality: list[int] = field(default_factory=list)
    row_names: list[str] = field(default_factory=list)
    row_lower: list[float] = field(default_factory=list)
    row_upper: list[float] = field(default_factory=list)
    _row_indices: list[list[int]] = field(default_factory=list)
    _row_values: list[list[float]] = field(default_factory=list)
    _variable_name_set: set[str] = field(default_factory=set, repr=False)
    _row_name_set: set[str] = field(default_factory=set, repr=False)

    def add_variable(
        self,
        name: str,
        *,
        objective: float = 0.0,
        lower: float = -inf,
        upper: float = inf,
        integer: bool = False,
    ) -> int:
        if name in self._variable_name_set:
            raise ScopfError(f"Duplicate canonical variable name: {name}")
        if lower > upper:
            raise ScopfError(f"Invalid bounds for {name}: {lower} > {upper}")
        index = len(self.variable_names)
        self.variable_names.append(name)
        self._variable_name_set.add(name)
        self.objective.append(float(objective))
        self.column_lower.append(float(lower))
        self.column_upper.append(float(upper))
        self.integrality.append(int(integer))
        return index

    def add_row(
        self,
        name: str,
        coefficients: Mapping[int, float],
        *,
        lower: float = -inf,
        upper: float = inf,
    ) -> int:
        if name in self._row_name_set:
            raise ScopfError(f"Duplicate canonical row name: {name}")
        if lower > upper:
            raise ScopfError(f"Invalid row bounds for {name}: {lower} > {upper}")
        combined = {int(i): float(value) for i, value in coefficients.items() if value != 0}
        if any(i < 0 or i >= len(self.variable_names) for i in combined):
            raise ScopfError(f"Row {name} references an invalid column")
        ordered = sorted(combined.items())
        self.row_names.append(name)
        self._row_name_set.add(name)
        self.row_lower.append(float(lower))
        self.row_upper.append(float(upper))
        self._row_indices.append([item[0] for item in ordered])
        self._row_values.append([item[1] for item in ordered])
        return len(self.row_names) - 1

    @property
    def num_columns(self) -> int:
        return len(self.variable_names)

    @property
    def num_rows(self) -> int:
        return len(self.row_names)

    def matrix_csr(self) -> sparse.csr_matrix:
        indptr = np.zeros(self.num_rows + 1, dtype=np.int64)
        for row, indices in enumerate(self._row_indices):
            indptr[row + 1] = indptr[row] + len(indices)
        indices = np.asarray([i for row in self._row_indices for i in row], dtype=np.int32)
        values = np.asarray([v for row in self._row_values for v in row], dtype=np.float64)
        return sparse.csr_matrix(
            (values, indices, indptr), shape=(self.num_rows, self.num_columns)
        )

    def column_arrays(self) -> tuple[FloatArray, FloatArray, FloatArray, IntArray]:
        return (
            np.asarray(self.objective, dtype=np.float64),
            np.asarray(self.column_lower, dtype=np.float64),
            np.asarray(self.column_upper, dtype=np.float64),
            np.asarray(self.integrality, dtype=np.int32),
        )

    def row_bound_arrays(self) -> tuple[FloatArray, FloatArray]:
        return (
            np.asarray(self.row_lower, dtype=np.float64),
            np.asarray(self.row_upper, dtype=np.float64),
        )

    def row_entries(self, row: int) -> tuple[list[int], list[float]]:
        return self._row_indices[row], self._row_values[row]

    def max_row_violation(self, values: FloatArray) -> float:
        activity = self.matrix_csr() @ values
        lower, upper = self.row_bound_arrays()
        return float(max(np.max(lower - activity), np.max(activity - upper), 0.0))
