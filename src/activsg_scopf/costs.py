"""Exact source polynomial retention and equal-width PWL conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .errors import ProvenanceError
from .matpower import GEN_STATUS, PMAX, PMIN, MatpowerCase
from .provenance import generator_source_id

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PwlCost:
    generator_source_row: int
    generator_source_id: str
    pmin_mw: float
    pmax_mw: float
    coefficients: tuple[float, ...]
    segment_widths_mw: FloatArray
    segment_slopes_per_mwh: FloatArray
    committed_base_cost: float
    maximum_absolute_error: float
    maximum_signed_error: float

    def polynomial_value(self, power_mw: float) -> float:
        return float(np.polyval(self.coefficients, power_mw))

    def pwl_value(self, power_mw: float) -> float:
        if power_mw < self.pmin_mw or power_mw > self.pmax_mw:
            raise ValueError("PWL evaluation is outside the exact PMIN/PMAX range")
        remaining = power_mw - self.pmin_mw
        value = self.committed_base_cost
        for width, slope in zip(
            self.segment_widths_mw, self.segment_slopes_per_mwh, strict=True
        ):
            used = min(max(remaining, 0.0), float(width))
            value += used * float(slope)
            remaining -= used
        return value


def _segment_max_error(coefficients: FloatArray, left: float, right: float) -> tuple[float, float]:
    if right == left:
        return 0.0, 0.0
    f_left = float(np.polyval(coefficients, left))
    f_right = float(np.polyval(coefficients, right))
    slope = (f_right - f_left) / (right - left)
    candidates = [left, right]
    derivative = np.polyder(coefficients)
    if derivative.size:
        equation = derivative.copy()
        equation[-1] -= slope
        for root in np.roots(np.trim_zeros(equation, trim="f")) if np.any(equation) else []:
            if abs(root.imag) <= 1e-10 and left < root.real < right:
                candidates.append(float(root.real))
    errors = []
    for point in candidates:
        chord = f_left + slope * (point - left)
        errors.append(chord - float(np.polyval(coefficients, point)))
    signed = max(errors, key=abs)
    return abs(float(signed)), float(signed)


def build_pwl_costs(case: MatpowerCase, *, segments: int = 10) -> dict[int, PwlCost]:
    if segments != 10:
        raise ProvenanceError("The registered model requires exactly 10 PWL segments")
    curves: dict[int, PwlCost] = {}
    for generator_index, (generator, cost_row) in enumerate(
        zip(case.gen, case.gencost, strict=True)
    ):
        if generator[GEN_STATUS] <= 0:
            continue
        pmin = float(generator[PMIN])
        pmax = float(generator[PMAX])
        count = int(cost_row[3])
        coefficients = np.asarray(cost_row[4 : 4 + count], dtype=np.float64)
        endpoints = np.linspace(pmin, pmax, segments + 1, dtype=np.float64)
        widths = np.diff(endpoints)
        values = np.polyval(coefficients, endpoints)
        slopes = np.divide(
            np.diff(values),
            widths,
            out=np.zeros_like(widths),
            where=widths != 0,
        )
        if np.any(np.diff(slopes) < -1e-10):
            raise ProvenanceError(
                f"Source production cost at generator row {generator_index + 1} is not convex"
            )
        errors = [
            _segment_max_error(coefficients, float(left), float(right))
            for left, right in zip(endpoints[:-1], endpoints[1:], strict=True)
        ]
        maximum = max(errors, key=lambda item: item[0]) if errors else (0.0, 0.0)
        curves[generator_index] = PwlCost(
            generator_source_row=generator_index + 1,
            generator_source_id=generator_source_id(generator_index),
            pmin_mw=pmin,
            pmax_mw=pmax,
            coefficients=tuple(float(value) for value in coefficients),
            segment_widths_mw=widths,
            segment_slopes_per_mwh=slopes,
            committed_base_cost=float(values[0]),
            maximum_absolute_error=maximum[0],
            maximum_signed_error=maximum[1],
        )
    return curves


def pwl_approximation_report(curves: dict[int, PwlCost]) -> dict[str, object]:
    maximum = max((curve.maximum_absolute_error for curve in curves.values()), default=0.0)
    return {
        "description": "source-derived production-cost curves; not submitted market offers",
        "segment_count": 10,
        "maximum_absolute_error": float(maximum),
        "generators": [
            {
                "source_id": curve.generator_source_id,
                "source_row": curve.generator_source_row,
                "pmin_mw": curve.pmin_mw,
                "pmax_mw": curve.pmax_mw,
                "polynomial_coefficients_high_to_low": list(curve.coefficients),
                "committed_base_cost": curve.committed_base_cost,
                "segment_widths_mw": curve.segment_widths_mw.tolist(),
                "segment_slopes_per_mwh": curve.segment_slopes_per_mwh.tolist(),
                "maximum_absolute_error": curve.maximum_absolute_error,
                "maximum_signed_error": curve.maximum_signed_error,
            }
            for curve in curves.values()
        ],
    }

