"""Replayable GPU Phase-I certificates for disjunctive LP regions."""

from __future__ import annotations

import hashlib
from math import fsum, isfinite
from typing import Any

import numpy as np
import numpy.typing as npt

from .canonical import CanonicalMILP
from .errors import ScopfError

FloatArray = npt.NDArray[np.float64]


def phase_one_semantic_row_key(row_name: str) -> str:
    """Remove the diagnostic insertion index from a Phase-I row identity."""

    parts = row_name.split("__", 3)
    if (
        len(parts) != 4
        or parts[0] != "phase1"
        or not parts[1].isdigit()
        or parts[2] not in {"lower", "upper"}
        or not parts[3]
    ):
        raise ScopfError(f"Malformed Phase-I row name: {row_name!r}")
    return f"phase1__{parts[2]}__{parts[3]}"


def sufficient_phase_one_violation_bound(
    source: CanonicalMILP, *, base_mva: float
) -> float:
    """Return a finite box-derived rho bound that cannot cut off Phase I."""

    if not isfinite(base_mva) or base_mva <= 0.0:
        raise ScopfError("Phase-I bound derivation requires positive finite base MVA")
    lower = np.asarray(source.column_lower, dtype=np.float64)
    upper = np.asarray(source.column_upper, dtype=np.float64)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ScopfError("Phase-I bound derivation requires finite source column bounds")
    row_lower, row_upper = source.row_bound_arrays()
    required = 0.0
    for row in range(source.num_rows):
        indices, values = source.row_entries(row)
        coefficients = np.asarray(values, dtype=np.float64)
        selected_lower = lower[np.asarray(indices, dtype=np.int64)]
        selected_upper = upper[np.asarray(indices, dtype=np.int64)]
        maximum_activity = float(
            np.sum(
                np.where(
                    coefficients >= 0.0,
                    coefficients * selected_upper,
                    coefficients * selected_lower,
                )
            )
        )
        minimum_activity = float(
            np.sum(
                np.where(
                    coefficients >= 0.0,
                    coefficients * selected_lower,
                    coefficients * selected_upper,
                )
            )
        )
        if isfinite(float(row_upper[row])):
            required = max(
                required,
                (maximum_activity - float(row_upper[row])) / float(base_mva),
            )
        if isfinite(float(row_lower[row])):
            required = max(
                required,
                (float(row_lower[row]) - minimum_activity) / float(base_mva),
            )
    if not isfinite(required):
        raise ScopfError("Phase-I box-derived violation bound is nonfinite")
    return max(1.0, float(np.nextafter(max(0.0, required), np.inf)))


def build_phase_one_model(
    source: CanonicalMILP,
    *,
    base_mva: float,
    maximum_violation_pu: float,
) -> CanonicalMILP:
    """Minimize one common per-unit violation over every source row.

    Each ranged source row becomes one or two upper inequalities.  The shared
    Phase-I variable relaxes every native-scaled row by the same amount, so an
    optimal value of zero is equivalent to feasibility of the source LP.
    """

    if not isfinite(base_mva) or base_mva <= 0.0:
        raise ScopfError("Phase-I requires positive finite base MVA")
    if not isfinite(maximum_violation_pu) or maximum_violation_pu <= 0.0:
        raise ScopfError("Phase-I maximum violation must be positive and finite")
    lower = np.asarray(source.column_lower, dtype=np.float64)
    upper = np.asarray(source.column_upper, dtype=np.float64)
    violation_upper = sufficient_phase_one_violation_bound(
        source, base_mva=base_mva
    )
    if violation_upper > maximum_violation_pu:
        raise ScopfError(
            "Phase-I box-derived violation bound exceeds the registered safety cap"
        )

    phase = CanonicalMILP()
    for name, lo, hi in zip(source.variable_names, lower, upper, strict=True):
        phase.add_variable(name, lower=float(lo), upper=float(hi), integer=False)
    violation_column = phase.add_variable(
        "phase1_violation_pu",
        objective=1.0,
        lower=0.0,
        upper=violation_upper,
    )
    row_lower, row_upper = source.row_bound_arrays()
    for row, source_name in enumerate(source.row_names):
        indices, values = source.row_entries(row)
        original = {int(index): float(value) for index, value in zip(indices, values, strict=True)}
        if isfinite(float(row_lower[row])):
            coefficients = {index: -value for index, value in original.items()}
            coefficients[violation_column] = -float(base_mva)
            phase.add_row(
                f"phase1__{row:05d}__lower__{source_name}",
                coefficients,
                upper=-float(row_lower[row]),
            )
        if isfinite(float(row_upper[row])):
            coefficients = dict(original)
            coefficients[violation_column] = -float(base_mva)
            phase.add_row(
                f"phase1__{row:05d}__upper__{source_name}",
                coefficients,
                upper=float(row_upper[row]),
            )
    if phase.num_rows == 0:
        raise ScopfError("Phase-I source model has no finite row bounds")
    return phase


def evaluate_phase_one_dual(
    model: CanonicalMILP,
    row_dual: npt.ArrayLike,
    *,
    safety_margin_pu: float,
) -> dict[str, Any]:
    """Evaluate a valid box-constrained dual lower bound in canonical units."""

    if not isfinite(safety_margin_pu) or safety_margin_pu < 0.0:
        raise ScopfError("Phase-I certificate safety margin must be nonnegative")
    dual = np.asarray(row_dual, dtype=np.float64)
    if dual.shape != (model.num_rows,) or not np.all(np.isfinite(dual)):
        raise ScopfError("Phase-I row dual has invalid shape or values")
    row_lower, row_upper = model.row_bound_arrays()
    if np.any(np.isfinite(row_lower)) or not np.all(np.isfinite(row_upper)):
        raise ScopfError("Phase-I replay expects upper inequalities only")
    lower = np.asarray(model.column_lower, dtype=np.float64)
    upper = np.asarray(model.column_upper, dtype=np.float64)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ScopfError("Phase-I replay requires finite column bounds")

    maximum_sign_violation = float(max(0.0, np.max(dual))) if dual.size else 0.0
    projected = np.minimum(dual, 0.0)
    objective = np.asarray(model.objective, dtype=np.float64)
    reduced = objective - np.asarray(model.matrix_csr().T @ projected).ravel()
    minimizing_values = np.where(reduced >= 0.0, lower, upper)
    raw_bound = fsum(
        [float(value) for value in row_upper * projected]
        + [float(value) for value in reduced * minimizing_values]
    )
    conservative = raw_bound - float(safety_margin_pu)
    return {
        "raw_lower_bound_pu": raw_bound,
        "conservative_lower_bound_pu": conservative,
        "safety_margin_pu": float(safety_margin_pu),
        "maximum_projected_dual_sign_violation": maximum_sign_violation,
        "projected_row_dual": projected,
        "implied_reduced_cost": reduced,
    }


def phase_one_certificate(
    model: CanonicalMILP,
    row_dual: npt.ArrayLike,
    *,
    safety_margin_pu: float,
    infeasibility_threshold_pu: float,
) -> dict[str, Any]:
    """Serialize a self-contained numerical Phase-I infeasibility certificate."""

    if not isfinite(infeasibility_threshold_pu) or infeasibility_threshold_pu < 0.0:
        raise ScopfError("Phase-I infeasibility threshold must be nonnegative")
    evaluated = evaluate_phase_one_dual(
        model, row_dual, safety_margin_pu=safety_margin_pu
    )
    projected = np.asarray(evaluated.pop("projected_row_dual"), dtype=np.float64)
    evaluated.pop("implied_reduced_cost")
    records = [
        {
            "row_name": name,
            "semantic_row_key": phase_one_semantic_row_key(name),
            "canonical_row_dual": float(value),
        }
        for name, value in zip(model.row_names, projected, strict=True)
    ]
    semantic_records = sorted(records, key=lambda record: record["semantic_row_key"])
    semantic_bytes = b"".join(
        str(record["semantic_row_key"]).encode("utf-8")
        + b"\0"
        + np.asarray([record["canonical_row_dual"]], dtype=np.float64).tobytes()
        for record in semantic_records
    )
    return {
        "certificate_kind": "box_dual_phase_one_infeasibility_v1",
        "formal_exact_rational_certificate": False,
        **evaluated,
        "infeasibility_threshold_pu": float(infeasibility_threshold_pu),
        "prune_certified": bool(
            float(evaluated["conservative_lower_bound_pu"])
            > float(infeasibility_threshold_pu)
        ),
        "row_identity_policy": "semantic_source_row_and_side_order_independent_v1",
        "canonical_row_duals": records,
        "canonical_row_dual_sha256": hashlib.sha256(projected.tobytes()).hexdigest(),
        "semantic_row_dual_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
    }


def replay_phase_one_certificate(
    model: CanonicalMILP, certificate: dict[str, Any]
) -> dict[str, Any]:
    """Replay a serialized Phase-I box-dual certificate independently."""

    records = certificate["canonical_row_duals"]
    serialized_dual = np.asarray(
        [float(record["canonical_row_dual"]) for record in records],
        dtype=np.float64,
    )
    observed_ordered_hash = hashlib.sha256(serialized_dual.tobytes()).hexdigest()
    if observed_ordered_hash != certificate["canonical_row_dual_sha256"]:
        raise ScopfError("Phase-I certificate ordered row-dual hash mismatch")
    by_semantic_key = {
        str(
            record.get(
                "semantic_row_key",
                phase_one_semantic_row_key(str(record["row_name"])),
            )
        ): float(record["canonical_row_dual"])
        for record in records
    }
    model_semantic_keys = [
        phase_one_semantic_row_key(name) for name in model.row_names
    ]
    if (
        len(by_semantic_key) != len(records)
        or len(set(model_semantic_keys)) != len(model_semantic_keys)
        or set(by_semantic_key) != set(model_semantic_keys)
    ):
        raise ScopfError("Phase-I certificate row identity mismatch")
    dual = np.asarray(
        [by_semantic_key[key] for key in model_semantic_keys], dtype=np.float64
    )
    semantic_records = sorted(by_semantic_key.items())
    semantic_bytes = b"".join(
        key.encode("utf-8")
        + b"\0"
        + np.asarray([value], dtype=np.float64).tobytes()
        for key, value in semantic_records
    )
    expected_semantic_hash = certificate.get("semantic_row_dual_sha256")
    if (
        expected_semantic_hash is not None
        and hashlib.sha256(semantic_bytes).hexdigest() != expected_semantic_hash
    ):
        raise ScopfError("Phase-I certificate semantic row-dual hash mismatch")
    replayed = evaluate_phase_one_dual(
        model,
        dual,
        safety_margin_pu=float(certificate["safety_margin_pu"]),
    )
    replayed.pop("projected_row_dual")
    replayed.pop("implied_reduced_cost")
    threshold = float(certificate["infeasibility_threshold_pu"])
    replayed["prune_certified"] = bool(
        float(replayed["conservative_lower_bound_pu"]) > threshold
    )
    replayed["infeasibility_threshold_pu"] = threshold
    return replayed
