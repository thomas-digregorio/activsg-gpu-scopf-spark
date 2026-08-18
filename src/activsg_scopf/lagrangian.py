"""Replayable Lagrangian lower bounds for disjunctive commitment regions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import fsum
from typing import Any

import numpy as np
import numpy.typing as npt

from .commitment_cuts import CommitmentFeasibilityCut
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
                "separable_binary_generator_lagrangian_with_feasibility_cuts_v2"
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
            "commitment_feasibility_cut_duals": [
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
                "serialization": "sparse_nonzero_dual_identity_hashed_v2",
                "coupling_row_count": len(coupling_names),
                "coupling_row_name_sha256": hashlib.sha256(
                    "\n".join(coupling_names).encode("utf-8")
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
                "effective_dispatch_coefficient_sha256": hashlib.sha256(
                    self.effective_dispatch_coefficients.tobytes()
                ).hexdigest(),
                "on_subproblem_value_sha256": hashlib.sha256(
                    self.on_subproblem_values.tobytes()
                ).hexdigest(),
                "minimizing_commitment_sha256": hashlib.sha256(
                    self.minimizing_commitment.tobytes()
                ).hexdigest(),
                "minimizing_committed_generator_source_rows": rows[
                    self.minimizing_commitment == 1
                ].tolist(),
            }
        )
        return certificate


def evaluate_lagrangian_bound_cupy(
    master: ReducedMaster,
    row_dual: npt.ArrayLike,
    region: RegionMasks,
    *,
    commitment_cuts: tuple[CommitmentFeasibilityCut, ...] = (),
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
    coupling_indices = np.asarray(
        [row.row_index for row in master.coupling_rows], dtype=np.int64
    )
    coupling_dual = cp.asarray(dual_host[coupling_indices], dtype=cp.float64)
    upper_mask = cp.asarray(
        [row.kind != "balance_equality" for row in master.coupling_rows],
        dtype=cp.bool_,
    )
    coupling_dual = cp.where(upper_mask, cp.minimum(coupling_dual, 0.0), coupling_dual)
    rhs = cp.asarray([row.rhs for row in master.coupling_rows], dtype=cp.float64)
    coefficients = cp.asarray(
        np.stack([row.generator_coefficients for row in master.coupling_rows]),
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


def optimize_lagrangian_bound_cupy(
    master: ReducedMaster,
    row_dual: npt.ArrayLike,
    region: RegionMasks,
    *,
    relaxation_primal_objective: float,
    iterations: int,
    polyak_fraction: float,
    commitment_cuts: tuple[CommitmentFeasibilityCut, ...] = (),
    initial_commitment_cut_dual: npt.ArrayLike | None = None,
) -> tuple[FloatArray, dict[str, Any]]:
    """Polish coupling multipliers while all numerical state remains on GPU.

    The initial cuOpt PDLP multiplier is already valid.  Projected Polyak
    supergradient steps can improve it but never replace the best valid iterate
    with a worse one.  The target is the relaxation primal objective, an upper
    bound on this concave dual maximum.
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
    coupling_indices_host = np.asarray(
        [row.row_index for row in master.coupling_rows], dtype=np.int64
    )
    upper_host = np.asarray(
        [row.kind != "balance_equality" for row in master.coupling_rows], dtype=bool
    )
    y = cp.asarray(full_dual[coupling_indices_host], dtype=cp.float64)
    upper = cp.asarray(upper_host)
    y = cp.where(upper, cp.minimum(y, 0.0), y)
    rhs = cp.asarray([row.rhs for row in master.coupling_rows], dtype=cp.float64)
    coefficients = cp.asarray(
        np.stack([row.generator_coefficients for row in master.coupling_rows]),
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
        z = cp.minimum(cp.asarray(supplied_cut_dual), 0.0)
    else:
        cut_coefficients = cp.empty((0, generator_count), dtype=cp.float64)
        cut_rhs = cp.empty(0, dtype=cp.float64)
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
        denominator = cp.sum(
            cp.where(projected_active, residual * residual, 0.0)
        ) + cp.sum(cp.where(cut_projected_active, cut_residual * cut_residual, 0.0))
        zero_denominator_iterations += denominator <= 0.0
        step = cp.where(
            denominator > 0.0,
            float(polyak_fraction) * cp.maximum(target - q, 0.0) / denominator,
            0.0,
        )
        y = y + step * residual
        y = cp.where(upper, cp.minimum(y, 0.0), y)
        z = cp.minimum(z + step * cut_residual, 0.0)

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
        "zero_projected_subgradient_iterations": int(
            zero_denominator_iterations.item()
        ),
        "coupling_row_count": int(coupling_indices_host.size),
        "generator_subproblem_count": int(generator_count),
        "device_state_persistent_across_iterations": True,
        "host_transfer_during_iterations": False,
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
    commitment_cuts: tuple[CommitmentFeasibilityCut, ...] = (),
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
    for coupling in master.coupling_rows:
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
    commitment_cuts_by_id: dict[str, CommitmentFeasibilityCut] | None = None,
) -> LagrangianEvaluation:
    by_name = {
        str(record["row_name"]): float(record["canonical_row_dual"])
        for record in certificate["coupling_row_duals"]
    }
    if len(by_name) != len(certificate["coupling_row_duals"]):
        raise ScopfError("Lagrangian certificate contains duplicate coupling rows")
    expected_ordered_names = [row.row_name for row in master.coupling_rows]
    expected_names = set(expected_ordered_names)
    compact = certificate.get("serialization") == (
        "sparse_nonzero_dual_identity_hashed_v2"
    )
    if compact:
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
    elif set(by_name) != expected_names:
        raise ScopfError("Lagrangian certificate coupling-row identity mismatch")
    row_dual = np.zeros(master.canonical.num_rows, dtype=np.float64)
    for row in master.coupling_rows:
        row_dual[row.row_index] = by_name.get(row.row_name, 0.0)
    cut_records = certificate.get("commitment_feasibility_cut_duals", [])
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
    if compact:
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
