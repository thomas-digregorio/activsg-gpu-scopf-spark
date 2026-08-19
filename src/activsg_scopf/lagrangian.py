"""Replayable Lagrangian lower bounds for disjunctive commitment regions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import fsum
from typing import Any

import numpy as np
import numpy.typing as npt

from .canonical import CanonicalMILP
from .commitment_cuts import (
    CommitmentCardinalityCut,
    CommitmentUpperCut,
    build_commitment_cardinality_cut,
)
from .errors import ScopfError
from .reduced import ReducedMaster

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RegionMasks:
    fixed_off: npt.NDArray[np.bool_]
    fixed_on: npt.NDArray[np.bool_]

    @classmethod
    def root(cls, generator_count: int) -> RegionMasks:
        return cls(
            fixed_off=np.zeros(generator_count, dtype=bool),
            fixed_on=np.zeros(generator_count, dtype=bool),
        )

    def validate(self, generator_count: int) -> None:
        expected = (generator_count,)
        if self.fixed_off.shape != expected or self.fixed_on.shape != expected:
            raise ScopfError("Disjunctive region mask has the wrong shape")
        if np.any(self.fixed_off & self.fixed_on):
            raise ScopfError("A generator cannot be fixed both off and on")

    def split(self, generator_position: int) -> tuple[RegionMasks, RegionMasks]:
        self.validate(self.fixed_off.size)
        if not 0 <= generator_position < self.fixed_off.size:
            raise ScopfError("Disjunctive split generator is out of range")
        if self.fixed_off[generator_position] or self.fixed_on[generator_position]:
            raise ScopfError("Disjunctive split generator is already fixed")
        off = self.fixed_off.copy()
        off[generator_position] = True
        on = self.fixed_on.copy()
        on[generator_position] = True
        return (
            RegionMasks(off, self.fixed_on.copy()),
            RegionMasks(self.fixed_off.copy(), on),
        )

    def as_dict(self, source_rows: npt.ArrayLike) -> dict[str, list[int]]:
        rows = np.asarray(source_rows, dtype=np.int64)
        self.validate(rows.size)
        return {
            "fixed_off_generator_source_rows": rows[self.fixed_off].tolist(),
            "fixed_on_generator_source_rows": rows[self.fixed_on].tolist(),
        }


@dataclass(frozen=True)
class LagrangianEvaluation:
    raw_lower_bound: float
    conservative_lower_bound: float
    safety_margin_dollars: float
    projected_dual_sign_violation: float
    coupling_duals: tuple[tuple[str, float], ...]
    effective_dispatch_coefficients: FloatArray
    on_subproblem_values: FloatArray
    minimizing_commitment: npt.NDArray[np.int8]
    commitment_cut_duals: tuple[tuple[str, float], ...] = ()

    def as_dict(
        self,
        generator_source_rows: npt.ArrayLike,
        *,
        compact: bool = False,
    ) -> dict[str, Any]:
        rows = np.asarray(generator_source_rows, dtype=np.int64)
        certificate: dict[str, Any] = {
            "certificate_kind": (
                "separable_binary_generator_lagrangian_with_commitment_upper_cuts_v3"
                if self.commitment_cut_duals
                else "separable_binary_generator_lagrangian_v1"
            ),
            "raw_lower_bound": self.raw_lower_bound,
            "conservative_lower_bound": self.conservative_lower_bound,
            "safety_margin_dollars": self.safety_margin_dollars,
            "projected_dual_sign_violation": self.projected_dual_sign_violation,
            "coupling_row_duals": [
                {"row_name": name, "canonical_row_dual": value}
                for name, value in self.coupling_duals
            ],
            "commitment_cut_duals": [
                {"cut_id": cut_id, "canonical_row_dual": value}
                for cut_id, value in self.commitment_cut_duals
            ],
        }
        if not compact:
            certificate["generator_subproblems"] = [
                {
                    "source_row": int(row),
                    "effective_dispatch_coefficient_per_mwh": float(coefficient),
                    "on_value": float(on_value),
                    "minimizing_commitment": int(commitment),
                }
                for row, coefficient, on_value, commitment in zip(
                    rows,
                    self.effective_dispatch_coefficients,
                    self.on_subproblem_values,
                    self.minimizing_commitment,
                    strict=True,
                )
            ]
            return certificate

        coupling_names = [name for name, _value in self.coupling_duals]
        certificate.update(
            {
                "serialization": (
                    "sparse_nonzero_dual_order_independent_identity_v3"
                ),
                "coupling_row_count": len(coupling_names),
                "coupling_row_name_set_sha256": hashlib.sha256(
                    "\n".join(sorted(coupling_names)).encode("utf-8")
                ).hexdigest(),
                "coupling_row_duals": [
                    {"row_name": name, "canonical_row_dual": value}
                    for name, value in self.coupling_duals
                    if value != 0.0
                ],
                "generator_subproblem_count": int(rows.size),
                "generator_source_row_sha256": hashlib.sha256(
                    rows.tobytes()
                ).hexdigest(),
                "minimizing_committed_generator_source_rows": rows[
                    self.minimizing_commitment == 1
                ].tolist(),
                "derived_generator_vectors_are_recomputed_not_hash_gated": True,
            }
        )
        return certificate


@dataclass(frozen=True)
class LagrangianMultiplierSearchModel:
    """A bounded LP used only to propose replayable Lagrangian multipliers."""

    canonical: CanonicalMILP
    selected_coupling_positions: npt.NDArray[np.int64]
    coupling_columns: npt.NDArray[np.int64]
    commitment_cut_columns: npt.NDArray[np.int64]
    epigraph_columns: npt.NDArray[np.int64]
    initial_values: FloatArray
    audit: dict[str, Any]


def evaluate_lagrangian_bound_cupy(
    master: ReducedMaster,
    row_dual: npt.ArrayLike,
    region: RegionMasks,
    *,
    commitment_cuts: tuple[CommitmentUpperCut, ...] = (),
    commitment_cut_dual: npt.ArrayLike | None = None,
) -> dict[str, Any]:
    """Evaluate all exact generator subproblems with CuPy FP64 primitives."""

    try:
        import cupy as cp
    except ImportError as exc:
        raise ScopfError("GPU Lagrangian evaluation requires CuPy") from exc
    generator_count = master.index.generator_source_rows.size
    region.validate(generator_count)
    for cut in commitment_cuts:
        cut.validate(generator_count)
    dual_host = np.asarray(row_dual, dtype=np.float64)
    if dual_host.shape != (master.canonical.num_rows,):
        raise ScopfError("GPU Lagrangian row dual has the wrong shape")
    coupling_rows = sorted(master.coupling_rows, key=lambda row: row.row_name)
    coupling_indices = np.asarray(
        [row.row_index for row in coupling_rows], dtype=np.int64
    )
    coupling_dual = cp.asarray(dual_host[coupling_indices], dtype=cp.float64)
    upper_mask = cp.asarray(
        [row.kind != "balance_equality" for row in coupling_rows],
        dtype=cp.bool_,
    )
    coupling_dual = cp.where(upper_mask, cp.minimum(coupling_dual, 0.0), coupling_dual)
    rhs = cp.asarray([row.rhs for row in coupling_rows], dtype=cp.float64)
    coefficients = cp.asarray(
        np.stack([row.generator_coefficients for row in coupling_rows]),
        dtype=cp.float64,
    )
    effective = -(coupling_dual @ coefficients)
    curves = [master.costs[int(index)] for index in master.index.generator_source_rows]
    pmin = cp.asarray([curve.pmin_mw for curve in curves], dtype=cp.float64)
    base_cost = cp.asarray(
        [curve.committed_base_cost for curve in curves], dtype=cp.float64
    )
    widths = cp.asarray(
        np.stack([curve.segment_widths_mw for curve in curves]), dtype=cp.float64
    )
    slopes = cp.asarray(
        np.stack([curve.segment_slopes_per_mwh for curve in curves]), dtype=cp.float64
    )
    on_value = base_cost + effective * pmin + cp.sum(
        cp.minimum(0.0, (slopes + effective[:, None]) * widths), axis=1
    )
    if commitment_cuts:
        cut_coefficients = cp.asarray(
            np.stack([cut.coefficients for cut in commitment_cuts]),
            dtype=cp.float64,
        )
        cut_rhs = cp.asarray([cut.rhs for cut in commitment_cuts], dtype=cp.float64)
        supplied_cut_dual = (
            np.zeros(len(commitment_cuts), dtype=np.float64)
            if commitment_cut_dual is None
            else np.asarray(commitment_cut_dual, dtype=np.float64)
        )
        if supplied_cut_dual.shape != (len(commitment_cuts),) or not np.all(
            np.isfinite(supplied_cut_dual)
        ):
            raise ScopfError("GPU Lagrangian commitment-cut dual has invalid values")
        cut_dual = cp.minimum(cp.asarray(supplied_cut_dual), 0.0)
        on_value = on_value - cut_dual @ cut_coefficients
        cut_constant = cut_dual @ cut_rhs
    else:
        cut_dual = cp.empty(0, dtype=cp.float64)
        cut_constant = cp.asarray(0.0, dtype=cp.float64)
    fixed_off = cp.asarray(region.fixed_off)
    fixed_on = cp.asarray(region.fixed_on)
    commitment = cp.where(fixed_off, 0, cp.where(fixed_on | (on_value < 0.0), 1, 0))
    local_value = cp.where(commitment > 0, on_value, 0.0)
    raw_bound = cp.sum(coupling_dual * rhs) + cut_constant + cp.sum(local_value)
    cp.cuda.get_current_stream().synchronize()
    return {
        "backend": "cupy_fp64",
        "raw_lower_bound": float(raw_bound.item()),
        "minimizing_commitment": cp.asnumpy(commitment).astype(np.int8),
        "effective_dispatch_coefficients": cp.asnumpy(effective),
        "on_subproblem_values": cp.asnumpy(on_value),
        "projected_commitment_cut_dual": cp.asnumpy(cut_dual),
        "commitment_feasibility_cut_count": len(commitment_cuts),
        "device_id": int(cp.cuda.Device().id),
    }


def _coordinate_ascent_commitment_cut_arrays(
    *,
    xp: Any,
    base_on_values: Any,
    coupling_constant: Any,
    cut_coefficients: Any,
    cut_rhs: Any,
    initial_cut_dual: Any,
    fixed_off: Any,
    fixed_on: Any,
    cycles: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Maximize each commitment-cut multiplier over its exact breakpoints.

    With the network-coupling multipliers fixed, the Lagrangian dual is a
    concave piecewise-linear function of any one commitment-cut multiplier.
    Its only finite breakpoints are generator on/off indifference points.
    Evaluating the current point, zero, and every nonpositive breakpoint
    therefore gives an exact one-dimensional coordinate maximizer.  Keeping
    the current multiplier as the first candidate makes every coordinate move
    monotone even when several candidates tie.
    """

    if cycles < 0:
        raise ScopfError("Commitment-cut coordinate-ascent cycles must be nonnegative")
    cut_count = int(cut_rhs.size)
    generator_count = int(base_on_values.size)
    if cut_coefficients.shape != (cut_count, generator_count):
        raise ScopfError("Commitment-cut coordinate matrix has an invalid shape")
    if initial_cut_dual.shape != (cut_count,):
        raise ScopfError("Commitment-cut coordinate initial dual has an invalid shape")
    if fixed_off.shape != (generator_count,) or fixed_on.shape != (generator_count,):
        raise ScopfError("Commitment-cut coordinate region masks have an invalid shape")

    z = xp.minimum(initial_cut_dual.copy(), 0.0)
    zero_candidate = xp.zeros(1, dtype=xp.float64)

    def local_values(on_values: Any) -> Any:
        return xp.where(
            fixed_off,
            0.0,
            xp.where(fixed_on, on_values, xp.minimum(0.0, on_values)),
        )

    def state() -> tuple[Any, Any, Any]:
        on_values = base_on_values - z @ cut_coefficients
        commitment = xp.where(
            fixed_off,
            0,
            xp.where(fixed_on | (on_values < 0.0), 1, 0),
        )
        raw = coupling_constant + z @ cut_rhs + xp.sum(local_values(on_values))
        return raw, on_values, commitment

    initial_raw, _initial_on, _initial_commitment = state()
    cycle_raw = xp.empty(cycles + 1, dtype=xp.float64)
    cycle_raw[0] = initial_raw
    maximum_tie_tolerance = xp.asarray(0.0, dtype=xp.float64)
    for cycle in range(cycles):
        for cut_index in range(cut_count):
            coefficients = cut_coefficients[cut_index]
            without_coordinate = (
                base_on_values
                - z @ cut_coefficients
                + z[cut_index] * coefficients
            )
            nonzero = coefficients != 0.0
            breakpoints = without_coordinate / xp.where(
                nonzero, coefficients, 1.0
            )
            valid_breakpoint = (
                nonzero & xp.isfinite(breakpoints) & (breakpoints <= 0.0)
            )
            breakpoints = xp.where(
                valid_breakpoint, breakpoints, z[cut_index]
            )
            # At an exact generator indifference point, ``on_value == 0`` and
            # the canonical minimizer chooses the unit off.  For an at-least
            # cut this can leave the same violated minimizer even though the
            # dual is optimal at that kink, causing the cutting-plane loop to
            # cycle.  Include both adjacent FP64 values and, among bitwise
            # equal dual maxima, choose the minimizer with the smallest
            # current-cut violation.  The raw bound remains exactly monotone;
            # this only resolves a degenerate argmin face.
            lower_neighbors = xp.where(
                valid_breakpoint,
                xp.nextafter(breakpoints, -xp.inf),
                z[cut_index],
            )
            upper_neighbors = xp.where(
                valid_breakpoint,
                xp.minimum(0.0, xp.nextafter(breakpoints, xp.inf)),
                z[cut_index],
            )
            candidates = xp.concatenate(
                (
                    z[cut_index : cut_index + 1],
                    zero_candidate,
                    breakpoints,
                    lower_neighbors,
                    upper_neighbors,
                )
            )
            candidate_on = (
                without_coordinate[None, :]
                - candidates[:, None] * coefficients[None, :]
            )
            current_on = (
                without_coordinate - z[cut_index] * coefficients
            )
            # Compare coordinate moves as local objective deltas.  Forming the
            # full million-dollar Lagrangian value for every breakpoint loses
            # small but decisive improvements to FP64 cancellation, which can
            # leave a strongly violated cut at multiplier zero.  The constant
            # network and other-cut terms cancel analytically.
            linear_delta = (candidates - z[cut_index]) * cut_rhs[cut_index]
            local_delta_terms = (
                local_values(candidate_on) - local_values(current_on)[None, :]
            )
            candidate_delta = linear_delta + xp.sum(local_delta_terms, axis=1)
            candidate_delta = xp.where(
                xp.isfinite(candidate_delta), candidate_delta, -xp.inf
            )
            maximum_delta = xp.max(candidate_delta)
            # Bound ordinary FP64 summation noise in the algebraically zero
            # move across a degenerate kink.  The current candidate guarantees
            # maximum_delta >= 0; only candidates within this forward-error
            # envelope may use the feasibility-oriented tie break.
            delta_absolute_scale = xp.abs(linear_delta) + xp.sum(
                xp.abs(local_delta_terms), axis=1
            )
            tie_tolerance = (
                64.0
                * np.finfo(np.float64).eps
                * xp.maximum(1.0, xp.max(delta_absolute_scale))
            )
            maximum_tie_tolerance = xp.maximum(
                maximum_tie_tolerance, tie_tolerance
            )
            candidate_commitment = xp.where(
                fixed_off[None, :],
                0,
                xp.where(fixed_on[None, :] | (candidate_on < 0.0), 1, 0),
            )
            candidate_violation = (
                candidate_commitment @ coefficients - cut_rhs[cut_index]
            )
            tie_score = xp.where(
                candidate_delta >= maximum_delta - tie_tolerance,
                xp.abs(candidate_violation),
                xp.inf,
            )
            z[cut_index] = candidates[xp.argmin(tie_score)]
        cycle_raw[cycle + 1] = state()[0]
    final_raw, final_on, final_commitment = state()
    return (
        z,
        final_raw,
        final_on,
        final_commitment,
        cycle_raw,
        maximum_tie_tolerance,
    )


def optimize_commitment_cut_duals_coordinate_numpy(
    master: ReducedMaster,
    row_dual: npt.ArrayLike,
    region: RegionMasks,
    *,
    commitment_cuts: tuple[CommitmentUpperCut, ...],
    initial_commitment_cut_dual: npt.ArrayLike | None = None,
    cycles: int = 1,
) -> tuple[FloatArray, dict[str, Any]]:
    """CPU reference for exact commitment-cut coordinate ascent."""

    generator_count = master.index.generator_source_rows.size
    region.validate(generator_count)
    for cut in commitment_cuts:
        cut.validate(generator_count)
    initial = (
        np.zeros(len(commitment_cuts), dtype=np.float64)
        if initial_commitment_cut_dual is None
        else np.asarray(initial_commitment_cut_dual, dtype=np.float64)
    )
    if initial.shape != (len(commitment_cuts),) or not np.all(np.isfinite(initial)):
        raise ScopfError("Commitment-cut coordinate initial dual has invalid values")
    base = evaluate_lagrangian_bound(
        master,
        row_dual,
        region,
        safety_margin_dollars=0.0,
    )
    coupling_by_name = dict(base.coupling_duals)
    coupling_constant = fsum(
        coupling_by_name[row.row_name] * row.rhs for row in master.coupling_rows
    )
    coefficients = (
        np.stack([cut.coefficients for cut in commitment_cuts])
        if commitment_cuts
        else np.empty((0, generator_count), dtype=np.float64)
    )
    rhs = np.asarray([cut.rhs for cut in commitment_cuts], dtype=np.float64)
    z, raw, on_values, commitment, cycle_raw, tie_tolerance = (
        _coordinate_ascent_commitment_cut_arrays(
        xp=np,
        base_on_values=np.asarray(base.on_subproblem_values, dtype=np.float64),
        coupling_constant=np.asarray(coupling_constant, dtype=np.float64),
        cut_coefficients=coefficients,
        cut_rhs=rhs,
        initial_cut_dual=initial,
        fixed_off=region.fixed_off,
        fixed_on=region.fixed_on,
            cycles=cycles,
        )
    )
    if not np.all(np.diff(cycle_raw) >= -1e-8):
        raise ScopfError("Commitment-cut coordinate ascent weakened a completed cycle")
    return np.asarray(z, dtype=np.float64), {
        "backend": "numpy_fp64_exact_commitment_cut_coordinate_ascent",
        "cycles": cycles,
        "coordinate_count": len(commitment_cuts),
        "initial_raw_lower_bound": float(cycle_raw[0]),
        "best_raw_lower_bound": float(raw),
        "improvement_dollars": float(raw - cycle_raw[0]),
        "cycle_raw_lower_bounds": np.asarray(cycle_raw, dtype=np.float64),
        "best_minimizing_commitment": np.asarray(commitment, dtype=np.int8),
        "best_on_subproblem_values": np.asarray(on_values, dtype=np.float64),
        "best_commitment_cut_dual": np.asarray(z, dtype=np.float64),
        "maximum_roundoff_tie_tolerance_dollars": float(tie_tolerance),
    }


def optimize_commitment_cut_duals_coordinate_cupy(
    master: ReducedMaster,
    row_dual: npt.ArrayLike,
    region: RegionMasks,
    *,
    commitment_cuts: tuple[CommitmentUpperCut, ...],
    initial_commitment_cut_dual: npt.ArrayLike | None = None,
    cycles: int = 8,
) -> tuple[FloatArray, dict[str, Any]]:
    """GPU-resident exact coordinate ascent for commitment-cut multipliers."""

    try:
        import cupy as cp
    except ImportError as exc:
        raise ScopfError("GPU commitment-cut coordinate ascent requires CuPy") from exc
    generator_count = master.index.generator_source_rows.size
    region.validate(generator_count)
    for cut in commitment_cuts:
        cut.validate(generator_count)
    dual_host = np.asarray(row_dual, dtype=np.float64)
    if dual_host.shape != (master.canonical.num_rows,) or not np.all(
        np.isfinite(dual_host)
    ):
        raise ScopfError("GPU commitment-cut coordinate row dual has invalid values")
    initial = (
        np.zeros(len(commitment_cuts), dtype=np.float64)
        if initial_commitment_cut_dual is None
        else np.asarray(initial_commitment_cut_dual, dtype=np.float64)
    )
    if initial.shape != (len(commitment_cuts),) or not np.all(np.isfinite(initial)):
        raise ScopfError("GPU commitment-cut coordinate initial dual has invalid values")

    coupling_rows = sorted(master.coupling_rows, key=lambda row: row.row_name)
    coupling_indices = np.asarray(
        [row.row_index for row in coupling_rows], dtype=np.int64
    )
    coupling_dual = cp.asarray(dual_host[coupling_indices], dtype=cp.float64)
    upper = cp.asarray(
        [row.kind != "balance_equality" for row in coupling_rows], dtype=cp.bool_
    )
    coupling_dual = cp.where(upper, cp.minimum(coupling_dual, 0.0), coupling_dual)
    coupling_rhs = cp.asarray([row.rhs for row in coupling_rows], dtype=cp.float64)
    coupling_coefficients = cp.asarray(
        np.stack([row.generator_coefficients for row in coupling_rows]),
        dtype=cp.float64,
    )
    effective = -(coupling_dual @ coupling_coefficients)
    curves = [master.costs[int(index)] for index in master.index.generator_source_rows]
    pmin = cp.asarray([curve.pmin_mw for curve in curves], dtype=cp.float64)
    base_cost = cp.asarray(
        [curve.committed_base_cost for curve in curves], dtype=cp.float64
    )
    widths = cp.asarray(
        np.stack([curve.segment_widths_mw for curve in curves]), dtype=cp.float64
    )
    slopes = cp.asarray(
        np.stack([curve.segment_slopes_per_mwh for curve in curves]), dtype=cp.float64
    )
    base_on = base_cost + effective * pmin + cp.sum(
        cp.minimum(0.0, (slopes + effective[:, None]) * widths), axis=1
    )
    cut_coefficients = (
        cp.asarray(
            np.stack([cut.coefficients for cut in commitment_cuts]),
            dtype=cp.float64,
        )
        if commitment_cuts
        else cp.empty((0, generator_count), dtype=cp.float64)
    )
    cut_rhs = cp.asarray([cut.rhs for cut in commitment_cuts], dtype=cp.float64)
    z, raw, on_values, commitment, cycle_raw, tie_tolerance = (
        _coordinate_ascent_commitment_cut_arrays(
        xp=cp,
        base_on_values=base_on,
        coupling_constant=coupling_dual @ coupling_rhs,
        cut_coefficients=cut_coefficients,
        cut_rhs=cut_rhs,
        initial_cut_dual=cp.asarray(initial, dtype=cp.float64),
        fixed_off=cp.asarray(region.fixed_off),
        fixed_on=cp.asarray(region.fixed_on),
            cycles=cycles,
        )
    )
    cp.cuda.get_current_stream().synchronize()
    cycle_host = cp.asnumpy(cycle_raw)
    if not np.all(np.isfinite(cycle_host)) or not np.all(
        np.diff(cycle_host) >= -1e-8
    ):
        raise ScopfError("GPU commitment-cut coordinate ascent lost monotonicity")
    initial_raw = float(cycle_host[0])
    final_raw = float(raw.item())
    return cp.asnumpy(z), {
        "backend": "cupy_fp64_exact_commitment_cut_coordinate_ascent",
        "cycles": cycles,
        "coordinate_count": len(commitment_cuts),
        "initial_raw_lower_bound": initial_raw,
        "best_raw_lower_bound": final_raw,
        "improvement_dollars": final_raw - initial_raw,
        "cycle_raw_lower_bounds": cycle_host,
        "best_minimizing_commitment": cp.asnumpy(commitment).astype(np.int8),
        "best_on_subproblem_values": cp.asnumpy(on_values),
        "best_commitment_cut_dual": cp.asnumpy(z),
        "maximum_roundoff_tie_tolerance_dollars": float(tie_tolerance.item()),
        "device_state_persistent_across_coordinates": True,
        "host_transfer_during_coordinates": False,
        "coordinate_policy": (
            "stable_local_delta_roundoff_envelope_feasibility_tie_break_v4"
        ),
        "device_id": int(cp.cuda.Device().id),
    }


def optimize_lagrangian_bound_cupy(
    master: ReducedMaster,
    row_dual: npt.ArrayLike,
    region: RegionMasks,
    *,
    relaxation_primal_objective: float,
    iterations: int,
    polyak_fraction: float,
    commitment_cuts: tuple[CommitmentUpperCut, ...] = (),
    initial_commitment_cut_dual: npt.ArrayLike | None = None,
    coupling_row_scales: npt.ArrayLike | None = None,
    commitment_cut_scales: npt.ArrayLike | None = None,
) -> tuple[FloatArray, dict[str, Any]]:
    """Polish coupling multipliers while all numerical state remains on GPU.

    The initial cuOpt PDLP multiplier is already valid.  Projected Polyak
    supergradient steps can improve it but never replace the best valid iterate
    with a worse one.  Optional positive diagonal row scales precondition the
    update in the same way as an exact row reformulation: for scaled row
    ``s * h(x)``, the canonical multiplier changes by ``step * s**2 * h``.
    Every evaluated canonical multiplier therefore remains a valid certificate.
    The target is any finite upper bound on this concave dual maximum.
    """

    if iterations < 0:
        raise ScopfError("GPU Lagrangian iteration count must be nonnegative")
    if not np.isfinite(polyak_fraction) or not 0.0 < polyak_fraction <= 2.0:
        raise ScopfError("GPU Lagrangian Polyak fraction must be in (0, 2]")
    if not np.isfinite(relaxation_primal_objective):
        raise ScopfError("GPU Lagrangian polishing requires a finite LP objective")
    try:
        import cupy as cp
    except ImportError as exc:
        raise ScopfError("GPU Lagrangian optimization requires CuPy") from exc

    generator_count = master.index.generator_source_rows.size
    region.validate(generator_count)
    for cut in commitment_cuts:
        cut.validate(generator_count)
    full_dual = np.asarray(row_dual, dtype=np.float64)
    if full_dual.shape != (master.canonical.num_rows,) or not np.all(
        np.isfinite(full_dual)
    ):
        raise ScopfError("GPU Lagrangian initial dual has invalid shape or values")
    coupling_rows = sorted(master.coupling_rows, key=lambda row: row.row_name)
    coupling_indices_host = np.asarray(
        [row.row_index for row in coupling_rows], dtype=np.int64
    )
    supplied_coupling_scales = (
        np.ones(len(coupling_rows), dtype=np.float64)
        if coupling_row_scales is None
        else np.asarray(coupling_row_scales, dtype=np.float64)
    )
    if supplied_coupling_scales.shape != (len(coupling_rows),) or not np.all(
        np.isfinite(supplied_coupling_scales) & (supplied_coupling_scales > 0.0)
    ):
        raise ScopfError("GPU Lagrangian coupling-row scales are invalid")
    coupling_scale = cp.asarray(supplied_coupling_scales, dtype=cp.float64)
    upper_host = np.asarray(
        [row.kind != "balance_equality" for row in coupling_rows], dtype=bool
    )
    y = cp.asarray(full_dual[coupling_indices_host], dtype=cp.float64)
    upper = cp.asarray(upper_host)
    y = cp.where(upper, cp.minimum(y, 0.0), y)
    rhs = cp.asarray([row.rhs for row in coupling_rows], dtype=cp.float64)
    coefficients = cp.asarray(
        np.stack([row.generator_coefficients for row in coupling_rows]),
        dtype=cp.float64,
    )
    curves = [master.costs[int(index)] for index in master.index.generator_source_rows]
    pmin = cp.asarray([curve.pmin_mw for curve in curves], dtype=cp.float64)
    base_cost = cp.asarray(
        [curve.committed_base_cost for curve in curves], dtype=cp.float64
    )
    widths = cp.asarray(
        np.stack([curve.segment_widths_mw for curve in curves]), dtype=cp.float64
    )
    slopes = cp.asarray(
        np.stack([curve.segment_slopes_per_mwh for curve in curves]), dtype=cp.float64
    )
    fixed_off = cp.asarray(region.fixed_off)
    fixed_on = cp.asarray(region.fixed_on)
    if commitment_cuts:
        cut_coefficients = cp.asarray(
            np.stack([cut.coefficients for cut in commitment_cuts]),
            dtype=cp.float64,
        )
        cut_rhs = cp.asarray([cut.rhs for cut in commitment_cuts], dtype=cp.float64)
        supplied_cut_dual = (
            np.zeros(len(commitment_cuts), dtype=np.float64)
            if initial_commitment_cut_dual is None
            else np.asarray(initial_commitment_cut_dual, dtype=np.float64)
        )
        if supplied_cut_dual.shape != (len(commitment_cuts),) or not np.all(
            np.isfinite(supplied_cut_dual)
        ):
            raise ScopfError("GPU Lagrangian initial commitment-cut dual is invalid")
        supplied_cut_scales = (
            np.ones(len(commitment_cuts), dtype=np.float64)
            if commitment_cut_scales is None
            else np.asarray(commitment_cut_scales, dtype=np.float64)
        )
        if supplied_cut_scales.shape != (len(commitment_cuts),) or not np.all(
            np.isfinite(supplied_cut_scales) & (supplied_cut_scales > 0.0)
        ):
            raise ScopfError("GPU Lagrangian commitment-cut scales are invalid")
        cut_scale = cp.asarray(supplied_cut_scales, dtype=cp.float64)
        z = cp.minimum(cp.asarray(supplied_cut_dual), 0.0)
    else:
        cut_coefficients = cp.empty((0, generator_count), dtype=cp.float64)
        cut_rhs = cp.empty(0, dtype=cp.float64)
        supplied_cut_scales = np.empty(0, dtype=np.float64)
        cut_scale = cp.empty(0, dtype=cp.float64)
        z = cp.empty(0, dtype=cp.float64)
    target = cp.asarray(float(relaxation_primal_objective), dtype=cp.float64)
    best_q = cp.asarray(-cp.inf, dtype=cp.float64)
    best_y = y.copy()
    best_z = z.copy()
    best_commitment = cp.zeros(generator_count, dtype=cp.int8)
    zero_denominator_iterations = cp.asarray(0, dtype=cp.int64)
    for _ in range(iterations + 1):
        effective = -(y @ coefficients)
        adjusted_slopes = slopes + effective[:, None]
        on_value = base_cost + effective * pmin + cp.sum(
            cp.minimum(0.0, adjusted_slopes * widths), axis=1
        )
        on_value = on_value - z @ cut_coefficients
        commitment = cp.where(
            fixed_off, 0, cp.where(fixed_on | (on_value < 0.0), 1, 0)
        ).astype(cp.int8)
        segment_dispatch = cp.where(
            (commitment[:, None] > 0) & (adjusted_slopes < 0.0), widths, 0.0
        )
        dispatch = commitment * pmin + cp.sum(segment_dispatch, axis=1)
        local_value = cp.where(commitment > 0, on_value, 0.0)
        q = y @ rhs + z @ cut_rhs + cp.sum(local_value)
        better = q > best_q
        best_q = cp.where(better, q, best_q)
        best_y = cp.where(better, y, best_y)
        best_z = cp.where(better, z, best_z)
        best_commitment = cp.where(better, commitment, best_commitment)
        residual = rhs - coefficients @ dispatch
        cut_residual = cut_rhs - cut_coefficients @ commitment
        projected_active = (~upper) | (y < 0.0) | (residual < 0.0)
        cut_projected_active = (z < 0.0) | (cut_residual < 0.0)
        scaled_residual = coupling_scale * residual
        scaled_cut_residual = cut_scale * cut_residual
        denominator = cp.sum(
            cp.where(projected_active, scaled_residual * scaled_residual, 0.0)
        ) + cp.sum(
            cp.where(
                cut_projected_active,
                scaled_cut_residual * scaled_cut_residual,
                0.0,
            )
        )
        zero_denominator_iterations += denominator <= 0.0
        step = cp.where(
            denominator > 0.0,
            float(polyak_fraction) * cp.maximum(target - q, 0.0) / denominator,
            0.0,
        )
        y = y + step * coupling_scale * scaled_residual
        y = cp.where(upper, cp.minimum(y, 0.0), y)
        z = cp.minimum(z + step * cut_scale * scaled_cut_residual, 0.0)

    cp.cuda.get_current_stream().synchronize()
    best_y_host = cp.asnumpy(best_y)
    polished = np.zeros_like(full_dual)
    polished[coupling_indices_host] = best_y_host
    initial_gpu = evaluate_lagrangian_bound_cupy(
        master,
        full_dual,
        region,
        commitment_cuts=commitment_cuts,
        commitment_cut_dual=(
            np.zeros(len(commitment_cuts), dtype=np.float64)
            if initial_commitment_cut_dual is None
            else initial_commitment_cut_dual
        ),
    )
    return polished, {
        "backend": "cupy_fp64_projected_polyak_supergradient",
        "iterations": iterations,
        "polyak_fraction": float(polyak_fraction),
        "relaxation_primal_objective_target": float(relaxation_primal_objective),
        "initial_raw_lower_bound": float(initial_gpu["raw_lower_bound"]),
        "best_raw_lower_bound": float(best_q.item()),
        "improvement_dollars": float(
            best_q.item() - float(initial_gpu["raw_lower_bound"])
        ),
        "best_minimizing_commitment": cp.asnumpy(best_commitment),
        "best_commitment_cut_dual": cp.asnumpy(best_z),
        "commitment_feasibility_cut_count": len(commitment_cuts),
        "diagonal_preconditioning": {
            "enabled": bool(
                coupling_row_scales is not None
                or commitment_cut_scales is not None
            ),
            "policy": "exact_positive_row_scaling_projected_polyak_v1",
            "coupling_scale_minimum": float(np.min(supplied_coupling_scales)),
            "coupling_scale_maximum": float(np.max(supplied_coupling_scales)),
            "commitment_cut_scale_minimum": (
                None
                if not supplied_cut_scales.size
                else float(np.min(supplied_cut_scales))
            ),
            "commitment_cut_scale_maximum": (
                None
                if not supplied_cut_scales.size
                else float(np.max(supplied_cut_scales))
            ),
            "certificate_validity_changed": False,
        },
        "zero_projected_subgradient_iterations": int(
            zero_denominator_iterations.item()
        ),
        "coupling_row_count": int(coupling_indices_host.size),
        "generator_subproblem_count": int(generator_count),
        "device_state_persistent_across_iterations": True,
        "host_transfer_during_iterations": False,
        "device_id": int(cp.cuda.Device().id),
    }


def _generator_breakpoint_state_arrays(
    master: ReducedMaster,
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int8]]:
    """Return every exact extreme state of each one-hour generator model.

    The off state is column zero.  The remaining columns are the committed
    PMIN point followed by the ten cumulative PWL segment endpoints.  A
    linear objective over one generator's convex PWL epigraph attains its
    minimum at one of these states, so enumerating them does not approximate
    the registered production-cost model or its exact PMIN/PMAX bounds.
    """

    curves = [master.costs[int(index)] for index in master.index.generator_source_rows]
    if not curves:
        raise ScopfError("Lagrangian state enumeration requires at least one generator")
    segment_count = int(curves[0].segment_widths_mw.size)
    if any(
        curve.segment_widths_mw.shape != (segment_count,)
        or curve.segment_slopes_per_mwh.shape != (segment_count,)
        for curve in curves
    ):
        raise ScopfError("Generator PWL state dimensions are inconsistent")

    generator_count = len(curves)
    state_count = segment_count + 2
    dispatch = np.zeros((generator_count, state_count), dtype=np.float64)
    cost = np.zeros_like(dispatch)
    commitment = np.zeros((generator_count, state_count), dtype=np.int8)
    commitment[:, 1:] = 1
    for position, curve in enumerate(curves):
        cumulative_dispatch = curve.pmin_mw + np.concatenate(
            (
                np.zeros(1, dtype=np.float64),
                np.cumsum(curve.segment_widths_mw, dtype=np.float64),
            )
        )
        cumulative_cost = curve.committed_base_cost + np.concatenate(
            (
                np.zeros(1, dtype=np.float64),
                np.cumsum(
                    curve.segment_widths_mw * curve.segment_slopes_per_mwh,
                    dtype=np.float64,
                ),
            )
        )
        dispatch[position, 1:] = cumulative_dispatch
        cost[position, 1:] = cumulative_cost
        if not (
            np.all(np.isfinite(cumulative_dispatch))
            and np.all(np.isfinite(cumulative_cost))
            and abs(float(cumulative_dispatch[0]) - curve.pmin_mw) <= 1e-12
            and abs(float(cumulative_dispatch[-1]) - curve.pmax_mw) <= 1e-9
        ):
            raise ScopfError("Generator breakpoint enumeration changed PMIN/PMAX")
    return dispatch, cost, commitment


def build_lagrangian_multiplier_search_model(
    master: ReducedMaster,
    row_dual: npt.ArrayLike,
    region: RegionMasks,
    *,
    commitment_cuts: tuple[CommitmentUpperCut, ...] = (),
    commitment_cut_dual: npt.ArrayLike | None = None,
    maximum_new_violated_coupling_rows: int = 256,
    multiplier_bound: float = 10_000.0,
    commitment_cut_multiplier_bound: float = 1_000_000.0,
    epigraph_bound: float = 100_000_000.0,
) -> LagrangianMultiplierSearchModel:
    """Build a finite-state LP that searches a restricted dual support.

    This LP is never itself a lower-bound authority.  It maximizes a
    hypograph of the separable generator Lagrangian over all exact PWL
    breakpoint states, but only on a deterministic subset of coupling rows.
    Its returned multipliers are projected to the valid sign cone and scored
    later by :func:`evaluate_lagrangian_bound`.  Finite variable bounds are
    therefore safe search restrictions rather than assumptions about the
    original SCOPF optimum.
    """

    if maximum_new_violated_coupling_rows < 0:
        raise ScopfError("Multiplier search active-row limit must be nonnegative")
    for value, label in (
        (multiplier_bound, "coupling multiplier"),
        (commitment_cut_multiplier_bound, "commitment-cut multiplier"),
        (epigraph_bound, "epigraph"),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ScopfError(f"Multiplier search {label} bound must be positive")

    generator_count = master.index.generator_source_rows.size
    region.validate(generator_count)
    for cut in commitment_cuts:
        cut.validate(generator_count)
    full_dual = np.asarray(row_dual, dtype=np.float64)
    if full_dual.shape != (master.canonical.num_rows,) or not np.all(
        np.isfinite(full_dual)
    ):
        raise ScopfError("Multiplier search initial row dual is invalid")
    supplied_cut_dual = (
        np.zeros(len(commitment_cuts), dtype=np.float64)
        if commitment_cut_dual is None
        else np.asarray(commitment_cut_dual, dtype=np.float64)
    )
    if supplied_cut_dual.shape != (len(commitment_cuts),) or not np.all(
        np.isfinite(supplied_cut_dual)
    ):
        raise ScopfError("Multiplier search initial cut dual is invalid")

    coupling_rows = sorted(master.coupling_rows, key=lambda row: row.row_name)
    coupling_indices = np.asarray(
        [row.row_index for row in coupling_rows], dtype=np.int64
    )
    coupling_dual = full_dual[coupling_indices].copy()
    upper = np.asarray(
        [row.kind != "balance_equality" for row in coupling_rows], dtype=bool
    )
    coupling_dual[upper] = np.minimum(coupling_dual[upper], 0.0)
    cut_dual = np.minimum(supplied_cut_dual, 0.0)
    rhs = np.asarray([row.rhs for row in coupling_rows], dtype=np.float64)
    coefficients = np.stack(
        [row.generator_coefficients for row in coupling_rows]
    ).astype(np.float64, copy=False)
    cut_coefficients = (
        np.stack([cut.coefficients for cut in commitment_cuts]).astype(
            np.float64, copy=False
        )
        if commitment_cuts
        else np.empty((0, generator_count), dtype=np.float64)
    )
    dispatch_states, state_cost, state_commitment = (
        _generator_breakpoint_state_arrays(master)
    )
    effective = -(coupling_dual @ coefficients)
    cut_adjustment = (
        -(cut_dual @ cut_coefficients)
        if commitment_cuts
        else np.zeros(generator_count, dtype=np.float64)
    )
    values = (
        state_cost
        + effective[:, None] * dispatch_states
        + cut_adjustment[:, None] * state_commitment
    )
    valid_state = np.ones(values.shape, dtype=bool)
    valid_state[:, 0] = ~region.fixed_on
    valid_state[:, 1:] = (~region.fixed_off)[:, None]
    values[~valid_state] = np.inf
    minimizing_state = np.argmin(values, axis=1)
    minimizing_dispatch = dispatch_states[
        np.arange(generator_count, dtype=np.int64), minimizing_state
    ]
    residual = rhs - coefficients @ minimizing_dispatch

    mandatory = (~upper) | (coupling_dual != 0.0)
    inactive_violated = np.flatnonzero(
        upper & ~mandatory & (residual < -1e-12)
    )
    ranked_violated = sorted(
        (int(position) for position in inactive_violated),
        key=lambda position: (
            float(residual[position]),
            coupling_rows[position].row_name,
        ),
    )
    selected_mask = mandatory.copy()
    selected_mask[
        np.asarray(
            ranked_violated[:maximum_new_violated_coupling_rows], dtype=np.int64
        )
    ] = True
    selected = np.flatnonzero(selected_mask).astype(np.int64)
    if not selected.size:
        raise ScopfError("Multiplier search selected no coupling rows")

    search = CanonicalMILP()
    coupling_columns: list[int] = []
    for position in selected:
        row = coupling_rows[int(position)]
        is_upper = bool(upper[int(position)])
        coupling_columns.append(
            search.add_variable(
                f"lambda__{row.row_name}",
                objective=-float(row.rhs),
                lower=-float(multiplier_bound),
                upper=0.0 if is_upper else float(multiplier_bound),
            )
        )
    cut_columns = [
        search.add_variable(
            f"mu__{cut.cut_id}",
            objective=-float(cut.rhs),
            lower=-float(commitment_cut_multiplier_bound),
            upper=0.0,
        )
        for cut in commitment_cuts
    ]
    epigraph_columns = [
        search.add_variable(
            f"generator_value__g{int(source_row) + 1:04d}",
            objective=-1.0,
            lower=-float(epigraph_bound),
            upper=float(epigraph_bound),
        )
        for source_row in master.index.generator_source_rows
    ]

    unselected = np.flatnonzero(~selected_mask).astype(np.int64)
    fixed_effective = (
        -(coupling_dual[unselected] @ coefficients[unselected])
        if unselected.size
        else np.zeros(generator_count, dtype=np.float64)
    )
    state_row_count = 0
    duplicate_state_count = 0
    for generator in range(generator_count):
        seen: set[tuple[float, float, int]] = set()
        for state in np.flatnonzero(valid_state[generator]):
            signature = (
                float(dispatch_states[generator, state]),
                float(state_cost[generator, state]),
                int(state_commitment[generator, state]),
            )
            if signature in seen:
                duplicate_state_count += 1
                continue
            seen.add(signature)
            power = float(dispatch_states[generator, state])
            committed = float(state_commitment[generator, state])
            row_coefficients: dict[int, float] = {
                epigraph_columns[generator]: 1.0
            }
            selected_values = coefficients[selected, generator] * power
            for column, coefficient in zip(
                coupling_columns, selected_values, strict=True
            ):
                if coefficient != 0.0:
                    row_coefficients[column] = float(coefficient)
            if committed:
                for column, coefficient in zip(
                    cut_columns,
                    cut_coefficients[:, generator],
                    strict=True,
                ):
                    if coefficient != 0.0:
                        row_coefficients[column] = float(coefficient)
            state_rhs = float(
                state_cost[generator, state]
                + fixed_effective[generator] * power
            )
            search.add_row(
                f"state__g{int(master.index.generator_source_rows[generator]) + 1:04d}"
                f"__s{int(state):02d}",
                row_coefficients,
                upper=state_rhs,
            )
            state_row_count += 1

    initial_values = np.zeros(search.num_columns, dtype=np.float64)
    initial_values[np.asarray(coupling_columns, dtype=np.int64)] = coupling_dual[
        selected
    ]
    if cut_columns:
        initial_values[np.asarray(cut_columns, dtype=np.int64)] = cut_dual
    initial_values[np.asarray(epigraph_columns, dtype=np.int64)] = np.min(
        values, axis=1
    )
    initial_residual = search.max_row_violation(initial_values)
    if initial_residual > 1e-8:
        raise ScopfError(
            "Multiplier search failed to embed its initial exact certificate"
        )
    if np.any(initial_values < np.asarray(search.column_lower) - 1e-9) or np.any(
        initial_values > np.asarray(search.column_upper) + 1e-9
    ):
        raise ScopfError("Multiplier search finite bounds exclude the initial certificate")

    return LagrangianMultiplierSearchModel(
        canonical=search,
        selected_coupling_positions=selected,
        coupling_columns=np.asarray(coupling_columns, dtype=np.int64),
        commitment_cut_columns=np.asarray(cut_columns, dtype=np.int64),
        epigraph_columns=np.asarray(epigraph_columns, dtype=np.int64),
        initial_values=initial_values,
        audit={
            "policy": "restricted_active_row_finite_generator_state_dual_lp_v1",
            "coupling_row_count": len(coupling_rows),
            "mandatory_coupling_row_count": int(np.count_nonzero(mandatory)),
            "inactive_violated_coupling_row_count": int(inactive_violated.size),
            "selected_coupling_row_count": int(selected.size),
            "selected_new_violated_coupling_row_count": int(
                np.count_nonzero(selected_mask & ~mandatory)
            ),
            "maximum_new_violated_coupling_rows": int(
                maximum_new_violated_coupling_rows
            ),
            "commitment_cut_count": len(commitment_cuts),
            "generator_state_row_count": state_row_count,
            "duplicate_generator_state_count": duplicate_state_count,
            "columns": search.num_columns,
            "rows": search.num_rows,
            "nonzeros": int(search.matrix_csr().nnz),
            "initial_canonical_row_residual": initial_residual,
            "multiplier_bound": float(multiplier_bound),
            "commitment_cut_multiplier_bound": float(
                commitment_cut_multiplier_bound
            ),
            "epigraph_bound": float(epigraph_bound),
            "finite_bounds_are_search_restrictions_only": True,
            "search_lp_solution_is_never_bound_authority": True,
            "exact_source_pmin_pmax_changed": False,
        },
    )


def optimize_lagrangian_bound_cupy_smoothed(
    master: ReducedMaster,
    row_dual: npt.ArrayLike,
    region: RegionMasks,
    *,
    temperatures_dollars: tuple[float, ...],
    iterations_per_temperature: int,
    learning_rate: float,
    commitment_cuts: tuple[CommitmentUpperCut, ...] = (),
    initial_commitment_cut_dual: npt.ArrayLike | None = None,
    coupling_row_scales: npt.ArrayLike | None = None,
    commitment_cut_scales: npt.ArrayLike | None = None,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
) -> tuple[FloatArray, dict[str, Any]]:
    """Optimize a smooth dual surrogate while retaining exact GPU bounds.

    For temperature ``tau``, each generator minimum is replaced during the
    search by ``-tau * log(sum(exp(-state/tau)))``.  This is a smooth lower
    approximation of the exact finite-state minimum.  Adam is used only to
    propose multipliers; every proposal is scored with the original
    nonsmoothed minimum, and only the best exact score is returned.  Thus no
    convergence claim or smoothing-error estimate is needed for validity.

    Canonical upper-row and commitment-cut multipliers are projected onto the
    nonpositive cone after every update.  Positive diagonal row scales merely
    reparameterize the dual variables and do not alter the certificate.
    """

    if iterations_per_temperature < 1:
        raise ScopfError("Smoothed GPU Lagrangian iterations must be positive")
    if not temperatures_dollars or any(
        not np.isfinite(value) or value <= 0.0 for value in temperatures_dollars
    ):
        raise ScopfError("Smoothed GPU Lagrangian temperatures must be finite and positive")
    if any(
        temperatures_dollars[index + 1] > temperatures_dollars[index]
        for index in range(len(temperatures_dollars) - 1)
    ):
        raise ScopfError("Smoothed GPU Lagrangian temperatures must be nonincreasing")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ScopfError("Smoothed GPU Lagrangian learning rate must be positive")
    if not (0.0 <= adam_beta1 < 1.0 and 0.0 <= adam_beta2 < 1.0):
        raise ScopfError("Smoothed GPU Lagrangian Adam factors must be in [0, 1)")
    if not np.isfinite(adam_epsilon) or adam_epsilon <= 0.0:
        raise ScopfError("Smoothed GPU Lagrangian Adam epsilon must be positive")
    try:
        import cupy as cp
    except ImportError as exc:
        raise ScopfError("Smoothed GPU Lagrangian optimization requires CuPy") from exc

    generator_count = master.index.generator_source_rows.size
    region.validate(generator_count)
    for cut in commitment_cuts:
        cut.validate(generator_count)
    full_dual = np.asarray(row_dual, dtype=np.float64)
    if full_dual.shape != (master.canonical.num_rows,) or not np.all(
        np.isfinite(full_dual)
    ):
        raise ScopfError("Smoothed GPU Lagrangian initial row dual is invalid")

    coupling_rows = sorted(master.coupling_rows, key=lambda row: row.row_name)
    coupling_indices_host = np.asarray(
        [row.row_index for row in coupling_rows], dtype=np.int64
    )
    upper_host = np.asarray(
        [row.kind != "balance_equality" for row in coupling_rows], dtype=bool
    )
    supplied_coupling_scales = (
        np.ones(len(coupling_rows), dtype=np.float64)
        if coupling_row_scales is None
        else np.asarray(coupling_row_scales, dtype=np.float64)
    )
    if supplied_coupling_scales.shape != (len(coupling_rows),) or not np.all(
        np.isfinite(supplied_coupling_scales) & (supplied_coupling_scales > 0.0)
    ):
        raise ScopfError("Smoothed GPU Lagrangian coupling scales are invalid")

    supplied_cut_dual = (
        np.zeros(len(commitment_cuts), dtype=np.float64)
        if initial_commitment_cut_dual is None
        else np.asarray(initial_commitment_cut_dual, dtype=np.float64)
    )
    supplied_cut_scales = (
        np.ones(len(commitment_cuts), dtype=np.float64)
        if commitment_cut_scales is None
        else np.asarray(commitment_cut_scales, dtype=np.float64)
    )
    if supplied_cut_dual.shape != (len(commitment_cuts),) or not np.all(
        np.isfinite(supplied_cut_dual)
    ):
        raise ScopfError("Smoothed GPU Lagrangian initial cut dual is invalid")
    if supplied_cut_scales.shape != (len(commitment_cuts),) or not np.all(
        np.isfinite(supplied_cut_scales) & (supplied_cut_scales > 0.0)
    ):
        raise ScopfError("Smoothed GPU Lagrangian cut scales are invalid")

    dispatch_host, state_cost_host, state_commitment_host = (
        _generator_breakpoint_state_arrays(master)
    )
    coupling_scale = cp.asarray(supplied_coupling_scales, dtype=cp.float64)
    cut_scale = cp.asarray(supplied_cut_scales, dtype=cp.float64)
    upper = cp.asarray(upper_host, dtype=cp.bool_)
    rhs = cp.asarray([row.rhs for row in coupling_rows], dtype=cp.float64)
    coefficients = cp.asarray(
        np.stack([row.generator_coefficients for row in coupling_rows]),
        dtype=cp.float64,
    )
    cut_coefficients = (
        cp.asarray(
            np.stack([cut.coefficients for cut in commitment_cuts]),
            dtype=cp.float64,
        )
        if commitment_cuts
        else cp.empty((0, generator_count), dtype=cp.float64)
    )
    cut_rhs = cp.asarray([cut.rhs for cut in commitment_cuts], dtype=cp.float64)
    dispatch_states = cp.asarray(dispatch_host, dtype=cp.float64)
    state_cost = cp.asarray(state_cost_host, dtype=cp.float64)
    state_commitment = cp.asarray(state_commitment_host, dtype=cp.float64)
    valid_state = cp.ones(dispatch_states.shape, dtype=cp.bool_)
    fixed_off = cp.asarray(region.fixed_off, dtype=cp.bool_)
    fixed_on = cp.asarray(region.fixed_on, dtype=cp.bool_)
    valid_state[:, 0] = ~fixed_on
    valid_state[:, 1:] = (~fixed_off)[:, None]

    initial_y = cp.asarray(full_dual[coupling_indices_host], dtype=cp.float64)
    initial_y = cp.where(upper, cp.minimum(initial_y, 0.0), initial_y)
    initial_z = cp.minimum(cp.asarray(supplied_cut_dual, dtype=cp.float64), 0.0)
    # Optimize scaled coordinates v and w where canonical multipliers are
    # y = scale*v and z = scale*w.
    v = initial_y / coupling_scale
    w = initial_z / cut_scale
    best_y = initial_y.copy()
    best_z = initial_z.copy()
    best_q = cp.asarray(-cp.inf, dtype=cp.float64)
    best_commitment = cp.zeros(generator_count, dtype=cp.int8)
    stage_records: list[dict[str, Any]] = []

    def state_values(y: Any, z: Any) -> Any:
        effective = -(y @ coefficients)
        cut_adjustment = (
            -(z @ cut_coefficients)
            if commitment_cuts
            else cp.zeros(generator_count, dtype=cp.float64)
        )
        values = (
            state_cost
            + effective[:, None] * dispatch_states
            + cut_adjustment[:, None] * state_commitment
        )
        return cp.where(valid_state, values, cp.inf)

    def exact_score(y: Any, z: Any, values: Any) -> tuple[Any, Any]:
        state_index = cp.argmin(values, axis=1)
        local = cp.take_along_axis(values, state_index[:, None], axis=1)[:, 0]
        constant = y @ rhs + (z @ cut_rhs if commitment_cuts else 0.0)
        return constant + cp.sum(local), (state_index > 0).astype(cp.int8)

    initial_values = state_values(initial_y, initial_z)
    best_q, best_commitment = exact_score(initial_y, initial_z, initial_values)
    initial_q = best_q.copy()
    maximum_softmin_error_bound = 0.0
    total_iterations = 0
    first_temperature = float(temperatures_dollars[0])
    for stage, temperature in enumerate(temperatures_dollars, start=1):
        tau = cp.asarray(float(temperature), dtype=cp.float64)
        # Reset moments at each continuation temperature.  Start the new stage
        # from the best exact certificate, not a possibly inferior smooth
        # terminal point.
        v = best_y / coupling_scale
        w = best_z / cut_scale
        first_moment_v = cp.zeros_like(v)
        second_moment_v = cp.zeros_like(v)
        first_moment_w = cp.zeros_like(w)
        second_moment_w = cp.zeros_like(w)
        stage_initial_q = best_q.copy()
        stage_learning_rate = float(learning_rate) * max(
            0.05, np.sqrt(float(temperature) / first_temperature)
        )
        for iteration in range(1, iterations_per_temperature + 1):
            y = coupling_scale * v
            y = cp.where(upper, cp.minimum(y, 0.0), y)
            z = cp.minimum(cut_scale * w, 0.0)
            values = state_values(y, z)
            row_minimum = cp.min(values, axis=1, keepdims=True)
            exponential = cp.where(
                valid_state,
                cp.exp(-(values - row_minimum) / tau),
                0.0,
            )
            weights = exponential / cp.sum(exponential, axis=1, keepdims=True)
            expected_dispatch = cp.sum(weights * dispatch_states, axis=1)
            expected_commitment = cp.sum(weights * state_commitment, axis=1)
            gradient_v = coupling_scale * (rhs - coefficients @ expected_dispatch)
            gradient_w = cut_scale * (
                cut_rhs - cut_coefficients @ expected_commitment
            )

            first_moment_v = (
                adam_beta1 * first_moment_v + (1.0 - adam_beta1) * gradient_v
            )
            second_moment_v = (
                adam_beta2 * second_moment_v
                + (1.0 - adam_beta2) * gradient_v * gradient_v
            )
            first_moment_w = (
                adam_beta1 * first_moment_w + (1.0 - adam_beta1) * gradient_w
            )
            second_moment_w = (
                adam_beta2 * second_moment_w
                + (1.0 - adam_beta2) * gradient_w * gradient_w
            )
            corrected_v = first_moment_v / (1.0 - adam_beta1**iteration)
            corrected_v2 = second_moment_v / (1.0 - adam_beta2**iteration)
            corrected_w = first_moment_w / (1.0 - adam_beta1**iteration)
            corrected_w2 = second_moment_w / (1.0 - adam_beta2**iteration)
            v = v + stage_learning_rate * corrected_v / (
                cp.sqrt(corrected_v2) + adam_epsilon
            )
            w = w + stage_learning_rate * corrected_w / (
                cp.sqrt(corrected_w2) + adam_epsilon
            )
            v = cp.where(upper, cp.minimum(v, 0.0), v)
            w = cp.minimum(w, 0.0)

            candidate_y = coupling_scale * v
            candidate_z = cut_scale * w
            candidate_values = state_values(candidate_y, candidate_z)
            candidate_q, candidate_commitment = exact_score(
                candidate_y, candidate_z, candidate_values
            )
            better = candidate_q > best_q
            best_q = cp.where(better, candidate_q, best_q)
            best_y = cp.where(better, candidate_y, best_y)
            best_z = cp.where(better, candidate_z, best_z)
            best_commitment = cp.where(
                better, candidate_commitment, best_commitment
            )
            total_iterations += 1

        cp.cuda.get_current_stream().synchronize()
        finite_stage = bool(
            cp.all(cp.isfinite(best_y)).item()
            and cp.all(cp.isfinite(best_z)).item()
            and cp.isfinite(best_q).item()
        )
        if not finite_stage:
            raise ScopfError("Smoothed GPU Lagrangian stage produced nonfinite state")
        stage_records.append(
            {
                "stage": stage,
                "temperature_dollars": float(temperature),
                "iterations": iterations_per_temperature,
                "learning_rate": stage_learning_rate,
                "initial_exact_raw_lower_bound": float(stage_initial_q.item()),
                "best_exact_raw_lower_bound": float(best_q.item()),
                "improvement_dollars": float(
                    best_q.item() - stage_initial_q.item()
                ),
            }
        )
        valid_counts = np.where(
            region.fixed_off,
            1,
            np.where(region.fixed_on, dispatch_host.shape[1] - 1, dispatch_host.shape[1]),
        )
        maximum_softmin_error_bound = max(
            maximum_softmin_error_bound,
            float(temperature) * float(np.sum(np.log(valid_counts))),
        )

    cp.cuda.get_current_stream().synchronize()
    polished = np.zeros_like(full_dual)
    polished[coupling_indices_host] = cp.asnumpy(best_y)
    best_z_host = cp.asnumpy(best_z)
    initial_q_host = float(initial_q.item())
    best_q_host = float(best_q.item())
    return polished, {
        "backend": "cupy_fp64_smoothed_finite_state_adam_exact_bound_tracking_v1",
        "temperature_schedule_dollars": [
            float(value) for value in temperatures_dollars
        ],
        "iterations_per_temperature": iterations_per_temperature,
        "total_iterations": total_iterations,
        "base_learning_rate": float(learning_rate),
        "adam_beta1": float(adam_beta1),
        "adam_beta2": float(adam_beta2),
        "adam_epsilon": float(adam_epsilon),
        "initial_raw_lower_bound": initial_q_host,
        "best_raw_lower_bound": best_q_host,
        "improvement_dollars": best_q_host - initial_q_host,
        "best_minimizing_commitment": cp.asnumpy(best_commitment),
        "best_commitment_cut_dual": best_z_host,
        "stage_records": stage_records,
        "generator_state_count": int(dispatch_host.shape[1]),
        "generator_subproblem_count": int(generator_count),
        "commitment_cut_count": len(commitment_cuts),
        "maximum_smooth_underestimate_bound_dollars": (
            maximum_softmin_error_bound
        ),
        "exact_bound_scored_every_iteration": True,
        "smoothing_used_as_certificate": False,
        "exact_source_pmin_pmax_changed": False,
        "device_state_persistent_within_each_temperature_stage": True,
        "host_transfer_during_iterations": False,
        "diagonal_reparameterization": {
            "enabled": bool(
                coupling_row_scales is not None
                or commitment_cut_scales is not None
            ),
            "coupling_scale_minimum": float(np.min(supplied_coupling_scales)),
            "coupling_scale_maximum": float(np.max(supplied_coupling_scales)),
            "commitment_cut_scale_minimum": (
                None
                if not supplied_cut_scales.size
                else float(np.min(supplied_cut_scales))
            ),
            "commitment_cut_scale_maximum": (
                None
                if not supplied_cut_scales.size
                else float(np.max(supplied_cut_scales))
            ),
            "certificate_validity_changed": False,
        },
        "device_id": int(cp.cuda.Device().id),
    }


def canonical_row_duals(
    master: ReducedMaster,
    native_row_dual: npt.ArrayLike,
    *,
    native_scaling_mode: str,
    base_mva: float,
) -> FloatArray:
    """Undo the adapter's exact row scaling for one-row-per-row models."""

    from .solvers.cuopt import native_scaling_vectors

    dual = np.asarray(native_row_dual, dtype=np.float64)
    if dual.shape != (master.canonical.num_rows,):
        raise ScopfError(
            "Reduced model requires exactly one native constraint per canonical row"
        )
    _, row_scale = native_scaling_vectors(
        master.canonical, mode=native_scaling_mode, base_mva=base_mva
    )
    return dual * row_scale


def evaluate_lagrangian_bound(
    master: ReducedMaster,
    row_dual: npt.ArrayLike,
    region: RegionMasks,
    *,
    safety_margin_dollars: float,
    commitment_cuts: tuple[CommitmentUpperCut, ...] = (),
    commitment_cut_dual: npt.ArrayLike | None = None,
) -> LagrangianEvaluation:
    """Evaluate a valid lower bound over the exact binary generator sets.

    cuOpt uses a minimization row-dual convention in which equality rows are
    free and upper-bound rows have non-positive multipliers.  Upper-row signs
    are projected toward the valid cone before evaluation; this can weaken but
    cannot invalidate the Lagrangian bound.
    """

    if not np.isfinite(safety_margin_dollars) or safety_margin_dollars < 0.0:
        raise ScopfError("Lagrangian certificate safety margin must be nonnegative")
    generator_count = master.index.generator_source_rows.size
    region.validate(generator_count)
    for cut in commitment_cuts:
        cut.validate(generator_count)
    dual = np.asarray(row_dual, dtype=np.float64)
    if dual.shape != (master.canonical.num_rows,) or not np.all(np.isfinite(dual)):
        raise ScopfError("Lagrangian row dual has invalid shape or values")

    effective = np.zeros(generator_count, dtype=np.float64)
    constant_terms: list[float] = []
    coupling_duals: list[tuple[str, float]] = []
    maximum_sign_violation = 0.0
    for coupling in sorted(master.coupling_rows, key=lambda row: row.row_name):
        observed = float(dual[coupling.row_index])
        if coupling.kind == "balance_equality":
            projected = observed
        else:
            maximum_sign_violation = max(maximum_sign_violation, max(observed, 0.0))
            projected = min(observed, 0.0)
        coupling_duals.append((coupling.row_name, projected))
        constant_terms.append(projected * coupling.rhs)
        effective -= projected * coupling.generator_coefficients

    supplied_cut_dual = (
        np.zeros(len(commitment_cuts), dtype=np.float64)
        if commitment_cut_dual is None
        else np.asarray(commitment_cut_dual, dtype=np.float64)
    )
    if supplied_cut_dual.shape != (len(commitment_cuts),) or not np.all(
        np.isfinite(supplied_cut_dual)
    ):
        raise ScopfError("Lagrangian commitment-cut dual has invalid shape or values")
    projected_cut_dual = np.minimum(supplied_cut_dual, 0.0)
    commitment_cut_duals: list[tuple[str, float]] = []
    commitment_adjustment = np.zeros(generator_count, dtype=np.float64)
    for cut, observed, projected in zip(
        commitment_cuts,
        supplied_cut_dual,
        projected_cut_dual,
        strict=True,
    ):
        maximum_sign_violation = max(maximum_sign_violation, max(float(observed), 0.0))
        commitment_cut_duals.append((cut.cut_id, float(projected)))
        constant_terms.append(float(projected) * float(cut.rhs))
        commitment_adjustment -= float(projected) * cut.coefficients

    on_values = np.empty(generator_count, dtype=np.float64)
    minimizing = np.empty(generator_count, dtype=np.int8)
    local_terms: list[float] = []
    for position, generator_index in enumerate(master.index.generator_source_rows):
        curve = master.costs[int(generator_index)]
        dispatch_coefficient = float(effective[position])
        on_value = fsum(
            [curve.committed_base_cost, dispatch_coefficient * curve.pmin_mw]
            + [
                min(0.0, (float(slope) + dispatch_coefficient) * float(width))
                for slope, width in zip(
                    curve.segment_slopes_per_mwh,
                    curve.segment_widths_mw,
                    strict=True,
                )
            ]
        ) + float(commitment_adjustment[position])
        on_values[position] = on_value
        if region.fixed_off[position]:
            commitment = 0
            local_value = 0.0
        elif region.fixed_on[position] or on_value < 0.0:
            commitment = 1
            local_value = on_value
        else:
            commitment = 0
            local_value = 0.0
        minimizing[position] = commitment
        local_terms.append(local_value)

    raw_bound = fsum(constant_terms + local_terms)
    return LagrangianEvaluation(
        raw_lower_bound=raw_bound,
        conservative_lower_bound=raw_bound - safety_margin_dollars,
        safety_margin_dollars=float(safety_margin_dollars),
        projected_dual_sign_violation=maximum_sign_violation,
        coupling_duals=tuple(coupling_duals),
        effective_dispatch_coefficients=effective,
        on_subproblem_values=on_values,
        minimizing_commitment=minimizing,
        commitment_cut_duals=tuple(commitment_cut_duals),
    )


def replay_lagrangian_certificate(
    master: ReducedMaster,
    certificate: dict[str, Any],
    region: RegionMasks,
    *,
    commitment_cuts_by_id: dict[str, CommitmentUpperCut] | None = None,
) -> LagrangianEvaluation:
    by_name = {
        str(record["row_name"]): float(record["canonical_row_dual"])
        for record in certificate["coupling_row_duals"]
    }
    if len(by_name) != len(certificate["coupling_row_duals"]):
        raise ScopfError("Lagrangian certificate contains duplicate coupling rows")
    expected_ordered_names = [row.row_name for row in master.coupling_rows]
    expected_names = set(expected_ordered_names)
    compact_v2 = certificate.get("serialization") == (
        "sparse_nonzero_dual_identity_hashed_v2"
    )
    compact_v3 = certificate.get("serialization") == (
        "sparse_nonzero_dual_order_independent_identity_v3"
    )
    if compact_v2:
        if (
            int(certificate.get("coupling_row_count", -1))
            != len(expected_ordered_names)
            or str(certificate.get("coupling_row_name_sha256"))
            != hashlib.sha256(
                "\n".join(expected_ordered_names).encode("utf-8")
            ).hexdigest()
            or not set(by_name).issubset(expected_names)
        ):
            raise ScopfError("Compact Lagrangian certificate row identity mismatch")
    elif compact_v3:
        if (
            int(certificate.get("coupling_row_count", -1))
            != len(expected_ordered_names)
            or str(certificate.get("coupling_row_name_set_sha256"))
            != hashlib.sha256(
                "\n".join(sorted(expected_ordered_names)).encode("utf-8")
            ).hexdigest()
            or not set(by_name).issubset(expected_names)
        ):
            raise ScopfError("Compact Lagrangian certificate row-set identity mismatch")
    elif set(by_name) != expected_names:
        raise ScopfError("Lagrangian certificate coupling-row identity mismatch")
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    for row in master.coupling_rows:
        row_dual[row.row_index] = by_name.get(row.row_name, 0.0)
    cut_records = certificate.get(
        "commitment_cut_duals",
        certificate.get("commitment_feasibility_cut_duals", []),
    )
    available = commitment_cuts_by_id or {}
    if len({str(record["cut_id"]) for record in cut_records}) != len(cut_records):
        raise ScopfError("Lagrangian certificate contains duplicate feasibility cuts")
    if any(str(record["cut_id"]) not in available for record in cut_records):
        raise ScopfError("Lagrangian certificate references an unknown feasibility cut")
    cuts = tuple(available[str(record["cut_id"])] for record in cut_records)
    cut_dual = np.asarray(
        [float(record["canonical_row_dual"]) for record in cut_records],
        dtype=np.float64,
    )
    replayed = evaluate_lagrangian_bound(
        master,
        row_dual,
        region,
        safety_margin_dollars=float(certificate["safety_margin_dollars"]),
        commitment_cuts=cuts,
        commitment_cut_dual=cut_dual,
    )
    if compact_v2:
        source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64) + 1
        committed_rows = source_rows[replayed.minimizing_commitment == 1].tolist()
        compact_checks = {
            "generator_subproblem_count": int(source_rows.size),
            "generator_source_row_sha256": hashlib.sha256(source_rows.tobytes()).hexdigest(),
            "effective_dispatch_coefficient_sha256": hashlib.sha256(
                replayed.effective_dispatch_coefficients.tobytes()
            ).hexdigest(),
            "on_subproblem_value_sha256": hashlib.sha256(
                replayed.on_subproblem_values.tobytes()
            ).hexdigest(),
            "minimizing_commitment_sha256": hashlib.sha256(
                replayed.minimizing_commitment.tobytes()
            ).hexdigest(),
            "minimizing_committed_generator_source_rows": committed_rows,
        }
        if any(certificate.get(key) != value for key, value in compact_checks.items()):
            raise ScopfError("Compact Lagrangian generator-subproblem replay mismatch")
    elif compact_v3:
        source_rows = np.asarray(master.index.generator_source_rows, dtype=np.int64) + 1
        if (
            int(certificate.get("generator_subproblem_count", -1))
            != int(source_rows.size)
            or str(certificate.get("generator_source_row_sha256"))
            != hashlib.sha256(source_rows.tobytes()).hexdigest()
        ):
            raise ScopfError(
                "Compact Lagrangian generator identity replay mismatch"
            )
    return replayed


def bus_prices_from_coupling_duals(
    master: ReducedMaster, row_dual: npt.ArrayLike
) -> FloatArray:
    """Return demand-derivative prices implied by one reduced LP dual."""

    dual = np.asarray(row_dual, dtype=np.float64)
    prices = np.zeros(master.operator.demand_mw.size, dtype=np.float64)
    for coupling in master.coupling_rows:
        value = float(dual[coupling.row_index])
        if coupling.kind != "balance_equality":
            value = min(value, 0.0)
        prices += value * coupling.bus_coefficients
    return prices


def choose_split_generator(
    commitments: npt.ArrayLike,
    on_values: npt.ArrayLike,
    region: RegionMasks,
    *,
    excluded_positions: set[int] | None = None,
) -> int:
    """Choose a free unit, preferring the most fractional LP commitment."""

    values = np.asarray(commitments, dtype=np.float64)
    margins = np.asarray(on_values, dtype=np.float64)
    region.validate(values.size)
    if margins.shape != values.shape:
        raise ScopfError("Split scores have inconsistent shapes")
    free = ~(region.fixed_off | region.fixed_on)
    for position in excluded_positions or set():
        if position < 0 or position >= values.size:
            raise ScopfError("Excluded split-generator position is out of range")
        free[position] = False
    if not np.any(free):
        raise ScopfError("Cannot split a fully fixed commitment region")
    fractionality = np.minimum(values, 1.0 - values)
    fractional_free = free & (fractionality > 1e-7)
    if np.any(fractional_free):
        candidates = np.flatnonzero(fractional_free)
        return int(candidates[np.argmax(fractionality[candidates])])
    candidates = np.flatnonzero(free)
    return int(candidates[np.argmin(np.abs(margins[candidates]))])


def verify_disjunctive_cover(
    generator_count: int,
    split_records: list[dict[str, Any]],
    leaf_regions: dict[str, RegionMasks],
) -> bool:
    """Verify that recorded binary splits leave a disjoint exhaustive cover."""

    active: dict[str, RegionMasks] = {"r": RegionMasks.root(generator_count)}
    for record in split_records:
        parent_id = str(record["parent_region_id"])
        off_id = str(record["off_child_region_id"])
        on_id = str(record["on_child_region_id"])
        position = int(record["generator_position"])
        if parent_id not in active or off_id in active or on_id in active:
            return False
        try:
            off, on = active[parent_id].split(position)
        except ScopfError:
            return False
        del active[parent_id]
        active[off_id] = off
        active[on_id] = on
    if set(active) != set(leaf_regions):
        return False
    return all(
        np.array_equal(active[name].fixed_off, region.fixed_off)
        and np.array_equal(active[name].fixed_on, region.fixed_on)
        for name, region in leaf_regions.items()
    )


def verify_cardinality_disjunctive_cover(
    generator_source_rows: npt.ArrayLike,
    split_records: list[dict[str, Any]],
    leaf_regions: dict[
        str, tuple[RegionMasks, tuple[CommitmentCardinalityCut, ...]]
    ],
) -> bool:
    """Verify binary/cardinality splits form one disjoint exhaustive cover."""

    rows = np.asarray(generator_source_rows, dtype=np.int64)
    if rows.ndim != 1 or len(set(int(row) for row in rows)) != int(rows.size):
        return False
    by_row = {int(row): position for position, row in enumerate(rows)}
    active: dict[
        str, tuple[RegionMasks, tuple[CommitmentCardinalityCut, ...]]
    ] = {"r": (RegionMasks.root(rows.size), ())}
    for record in split_records:
        parent_id = str(record["parent_region_id"])
        off_id = str(record["off_child_region_id"])
        on_id = str(record["on_child_region_id"])
        if parent_id not in active or off_id in active or on_id in active:
            return False
        masks, cuts = active.pop(parent_id)
        split_kind = str(record.get("split_kind", "binary_commitment_v1"))
        if split_kind == "binary_commitment_v1":
            try:
                off_masks, on_masks = masks.split(int(record["generator_position"]))
            except (ScopfError, KeyError, TypeError, ValueError):
                return False
            active[off_id] = (off_masks, cuts)
            active[on_id] = (on_masks, cuts)
            continue
        if split_kind != "binary_commitment_cardinality_sum_v1":
            return False
        try:
            subset_rows = tuple(int(row) for row in record["subset_source_rows"])
            positions = np.asarray([by_row[row] for row in subset_rows], dtype=np.int64)
            floor_value = int(record["floor_value"])
            ceil_value = int(record["ceil_value"])
            if ceil_value != floor_value + 1:
                return False
            at_most = build_commitment_cardinality_cut(
                generator_source_rows=rows,
                subset_positions=positions,
                subset_id=str(record["subset_id"]),
                branch_side="at_most",
                integer_threshold=floor_value,
            )
            at_least = build_commitment_cardinality_cut(
                generator_source_rows=rows,
                subset_positions=positions,
                subset_id=str(record["subset_id"]),
                branch_side="at_least",
                integer_threshold=ceil_value,
            )
        except (ScopfError, KeyError, TypeError, ValueError):
            return False
        if (
            record.get("at_most_cut_id") != at_most.cut_id
            or record.get("at_least_cut_id") != at_least.cut_id
        ):
            return False
        active[off_id] = (masks, cuts + (at_most,))
        active[on_id] = (masks, cuts + (at_least,))
    if set(active) != set(leaf_regions):
        return False
    for region_id, (expected_masks, expected_cuts) in active.items():
        observed_masks, observed_cuts = leaf_regions[region_id]
        if not (
            np.array_equal(expected_masks.fixed_off, observed_masks.fixed_off)
            and np.array_equal(expected_masks.fixed_on, observed_masks.fixed_on)
            and tuple(cut.cut_id for cut in expected_cuts)
            == tuple(cut.cut_id for cut in observed_cuts)
        ):
            return False
    return True
