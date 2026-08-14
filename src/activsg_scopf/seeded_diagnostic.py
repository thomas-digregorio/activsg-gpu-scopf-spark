"""One-shot ACTIVSg2000 round-2 cuOpt diagnostic with a full CPU MIP start."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .canonical import CanonicalMILP
from .config import RunConfig
from .deadline import PeakMemorySampler
from .environment import environment_manifest, validate_platform
from .errors import DeadlineExceeded, ProvenanceError, ScopfError
from .matpower import GEN_STATUS, read_contingency_table, read_matpower_case, sha256_file
from .model import MasterModel, build_master
from .network import (
    ContingencyCatalog,
    NetworkData,
    build_contingency_catalog,
    build_network,
    contingency_catalog_report,
)
from .official import frozen_identity
from .paths import guard_input_path, guard_output_path, guard_runtime_environment
from .provenance import build_source_manifest, write_json_atomic
from .screening import ContingencyScreener, SecurityPair, add_security_pairs
from .solution import serialize_solution
from .solvers.common import SolveResult
from .solvers.cuopt import FULL_MIP_START, prepare_mip_start, solve_cuopt
from .verify import verify_serialized_solution

FloatArray = npt.NDArray[np.float64]

DIAGNOSTIC_KIND = "seeded_round2_diagnostic"
DIAGNOSTIC_IDENTITIES = {
    "activsg2000-gpu-round2-cpu-seed-diagnostic-v1": {
        "tag": "diagnostic-2000-gpu-round2-cpu-seed-v1",
        "mip_start_bound_policy": "preserve_as_serialized",
    },
    "activsg2000-gpu-round2-cpu-seed-diagnostic-v2": {
        "tag": "diagnostic-2000-gpu-round2-cpu-seed-v2",
        "mip_start_bound_policy": "project_numerical_excess_to_exact_bound",
    },
}
CPU_RESULT_SHA256 = "5573425a8e625c0c964b2c61d33ca80666de74a350e9431a96e5ce9b90e02e3f"
GPU_V3_RESULT_SHA256 = "9b9e7d3b0dfdad0faeb740f639307844920c18eb85c07ac9705f8b84e034366c"
CPU_RESULT_COMMIT = "173fd0c10b8af5ed431956f2eea9726032df6f9e"
GPU_V3_RESULT_COMMIT = "921ab8e59ce71866ba004127d73278d7918c1d9c"
EXPECTED_MODEL_DIMENSIONS = {"columns": 9220, "rows": 8961, "nonzeros": 27120}
EXPECTED_BASE_MODEL_ROWS = 8788
EXPECTED_SECURITY_ROWS = 173
EXPECTED_CPU_OBJECTIVE = 1133479.3855011363
EXPECTED_SOLVER_TIME_LIMIT_SECONDS = 1657.606618394988
EXPECTED_TOTAL_DEADLINE_SECONDS = 1800.0
PAIR_PATTERN = re.compile(r"^c(?P<contingency>\d+)_m(?P<monitored>\d+)_(?P<side>lower|upper)$")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read diagnostic evidence {path}: {exc}") from exc


def _diagnostic_input(
    config: RunConfig, name: str, *, expected_sha256: str
) -> tuple[Path, dict[str, Any]]:
    record = config.raw["diagnostic_inputs"][name]
    if record.get("sha256") != expected_sha256:
        raise ProvenanceError(
            f"Diagnostic configuration changed the registered {name} hash"
        )
    path = guard_input_path(config.root / str(record["path"]))
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ProvenanceError(
            f"Diagnostic input hash mismatch for {name}: expected {expected_sha256}, "
            f"observed {observed}"
        )
    return path, _read_json(path)


def validate_diagnostic_identity(config: RunConfig) -> None:
    """Fail closed unless every registered diagnostic choice remains exact."""

    if config.case_name != "ACTIVSg2000":
        raise ScopfError("The seeded round-2 diagnostic is registered only for ACTIVSg2000")
    if config.benchmark_kind != DIAGNOSTIC_KIND:
        raise ScopfError("Configuration is not a seeded round-2 diagnostic")
    identity = DIAGNOSTIC_IDENTITIES.get(config.benchmark_id)
    if identity is None:
        raise ScopfError(f"Unregistered seeded diagnostic id {config.benchmark_id!r}")
    if config.raw["benchmark"].get("required_git_tag") != identity["tag"]:
        raise ScopfError(
            f"Expected frozen diagnostic tag {identity['tag']!r}"
        )
    exact_model = {
        "interval_hours": 1.0,
        "pwl_segments": 10,
        "mip_relative_gap_tolerance": 1e-3,
        "model_residual_tolerance_pu": 1e-6,
        "security_violation_tolerance_pu": 1e-5,
        "lodf_validation_tolerance_pu": 1e-9,
        "lodf_validation_columns": 3,
        "lodf_build_chunk_columns": 256,
        "screen_chunk_columns": 256,
    }
    for key, expected in exact_model.items():
        if config.model.get(key) != expected:
            raise ScopfError(
                f"Seeded diagnostic changed model value {key}: expected {expected!r}, "
                f"observed {config.model.get(key)!r}"
            )
    runtime = config.runtime
    if float(runtime.get("deadline_seconds", -1.0)) != EXPECTED_TOTAL_DEADLINE_SECONDS:
        raise ScopfError("Seeded diagnostic requires the registered 1800-second boundary")
    profile = config.raw["platforms"].get("dgx_spark", {})
    required_profile = {
        "solver": "cuopt",
        "screening": "cupy",
        "solver_threads": 0,
        "cuopt_version": "26.06.00",
        "mip_acceptance_policy": (
            "finite_incumbent_bound_gap_and_native_residuals_v1"
        ),
        "mip_certificate_residual_tolerance": 1e-6,
    }
    for key, expected in required_profile.items():
        if profile.get(key) != expected:
            raise ScopfError(
                f"Seeded diagnostic changed dgx_spark profile {key}: "
                f"expected {expected!r}, observed {profile.get(key)!r}"
            )
    diagnostic = config.raw.get("diagnostic", {})
    required_diagnostic = {
        "source_round": 2,
        "expected_security_rows": EXPECTED_SECURITY_ROWS,
        "mip_start_mode": FULL_MIP_START,
        "console_logging": True,
        "constraint_generation_enabled": False,
        "solver_time_limit_seconds": EXPECTED_SOLVER_TIME_LIMIT_SECONDS,
        "canonical_feasibility_tolerance": 1e-6,
        "objective_identity_tolerance": 1e-5,
    }
    for key, expected in required_diagnostic.items():
        if diagnostic.get(key) != expected:
            raise ScopfError(
                f"Seeded diagnostic changed {key}: expected {expected!r}, "
                f"observed {diagnostic.get(key)!r}"
            )
    observed_bound_policy = diagnostic.get(
        "mip_start_bound_policy", "preserve_as_serialized"
    )
    if observed_bound_policy != identity["mip_start_bound_policy"]:
        raise ScopfError(
            "Seeded diagnostic changed mip_start_bound_policy: expected "
            f"{identity['mip_start_bound_policy']!r}, observed "
            f"{observed_bound_policy!r}"
        )


def validate_reference_evidence(
    config: RunConfig,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    cpu_path, cpu = _diagnostic_input(
        config, "secure_cpu_result", expected_sha256=CPU_RESULT_SHA256
    )
    gpu_path, gpu = _diagnostic_input(
        config, "gpu_v3_round2_result", expected_sha256=GPU_V3_RESULT_SHA256
    )
    source_hashes = (
        config.raw["raw_inputs"]["case_sha256"],
        config.raw["raw_inputs"]["contingency_sha256"],
    )
    for label, evidence in (("CPU", cpu), ("GPU v3", gpu)):
        observed = evidence.get("source_manifest", {}).get("source_identity", {})
        if (
            observed.get("case_sha256"),
            observed.get("contingency_sha256"),
        ) != source_hashes:
            raise ProvenanceError(f"{label} evidence uses different raw source hashes")
        if evidence.get("case_name") != config.case_name:
            raise ProvenanceError(f"{label} evidence uses a different case")
    if (
        cpu.get("status") != "optimal_verified"
        or cpu.get("verification", {}).get("passed") is not True
        or cpu.get("benchmark_id") != "activsg2000-gap-v1-1e-3"
        or cpu.get("frozen_identity", {}).get("tag") != "experiment-2000-gap-v1"
        or cpu.get("frozen_identity", {}).get("commit") != CPU_RESULT_COMMIT
    ):
        raise ProvenanceError("Registered CPU evidence is not the accepted secure result")
    if abs(float(cpu.get("objective", np.inf)) - EXPECTED_CPU_OBJECTIVE) > 1e-9:
        raise ProvenanceError("Registered CPU evidence objective changed")
    if (
        gpu.get("status") != "incomplete_no_incumbent"
        or gpu.get("benchmark_id") != "activsg2000-gpu-gap-v3-1e-3"
        or gpu.get("frozen_identity", {}).get("tag") != "experiment-2000-gpu-gap-v3"
        or gpu.get("frozen_identity", {}).get("commit") != GPU_V3_RESULT_COMMIT
    ):
        raise ProvenanceError("Registered GPU v3 evidence is not the failed round-2 result")
    dimensions = gpu.get("final_model_dimensions", {})
    if any(dimensions.get(key) != value for key, value in EXPECTED_MODEL_DIMENSIONS.items()):
        raise ProvenanceError("GPU v3 evidence does not identify the exact round-2 master")
    pair_ids = list(gpu.get("added_security_pair_ids", []))
    if (
        len(pair_ids) != EXPECTED_SECURITY_ROWS
        or len(set(pair_ids)) != EXPECTED_SECURITY_ROWS
        or pair_ids != sorted(pair_ids)
    ):
        raise ProvenanceError("GPU v3 security-pair identity is incomplete or noncanonical")
    rounds = gpu.get("constraint_generation_rounds", [])
    if (
        len(rounds) != 2
        or rounds[0].get("added_pair_ids") != pair_ids
        or rounds[1].get("rows_before_solve") != EXPECTED_MODEL_DIMENSIONS["rows"]
        or rounds[1].get("solver_budget_seconds") != EXPECTED_SOLVER_TIME_LIMIT_SECONDS
    ):
        raise ProvenanceError("GPU v3 evidence does not reproduce the exact round-2 boundary")
    return cpu_path, cpu, gpu_path, gpu


def security_pairs_from_ids(
    pair_ids: list[str],
    network: NetworkData,
    catalog: ContingencyCatalog,
) -> tuple[SecurityPair, ...]:
    """Reconstruct exact canonical security rows from immutable stable IDs."""

    if len(pair_ids) != len(set(pair_ids)) or pair_ids != sorted(pair_ids):
        raise ProvenanceError("Security-pair IDs must be unique and canonically ordered")
    outage_by_label = {
        outage.contingency_label: (column, outage)
        for column, outage in enumerate(catalog.valid)
    }
    if len(outage_by_label) != len(catalog.valid):
        raise ProvenanceError("Valid outage labels are not unique")
    monitored_by_row = {
        int(source_index) + 1: active_index
        for active_index, source_index in enumerate(network.active_branch_source_rows)
    }
    parsed: list[tuple[str, int, int, str]] = []
    needed_columns: list[int] = []
    for pair_id in pair_ids:
        match = PAIR_PATTERN.fullmatch(pair_id)
        if match is None:
            raise ProvenanceError(f"Invalid security-pair ID {pair_id!r}")
        contingency = int(match.group("contingency"))
        monitored_row = int(match.group("monitored"))
        if contingency not in outage_by_label:
            raise ProvenanceError(f"Unknown contingency label in {pair_id}")
        if monitored_row not in monitored_by_row:
            raise ProvenanceError(f"Unknown monitored branch row in {pair_id}")
        outage_column, _ = outage_by_label[contingency]
        parsed.append((pair_id, contingency, monitored_row, match.group("side")))
        needed_columns.append(outage_column)
    unique_columns = sorted(set(needed_columns))
    generated_lodf: dict[int, FloatArray] = {}
    if catalog.lodf is None and unique_columns:
        outage_indices = np.asarray(
            [catalog.valid[column].active_branch_index for column in unique_columns],
            dtype=np.int64,
        )
        columns = catalog.lodf_operator.lodf_columns(outage_indices)
        generated_lodf = {
            outage_column: columns[:, local]
            for local, outage_column in enumerate(unique_columns)
        }
    pairs: list[SecurityPair] = []
    for pair_id, contingency, monitored_row, side in parsed:
        outage_column, outage = outage_by_label[contingency]
        monitored = monitored_by_row[monitored_row]
        lodf_column = (
            catalog.lodf[:, outage_column]
            if catalog.lodf is not None
            else generated_lodf[outage_column]
        )
        pair = SecurityPair(
            contingency_label=contingency,
            monitored_branch_source_row=monitored_row,
            side=side,  # type: ignore[arg-type]
            outage_column=outage_column,
            monitored_active_index=monitored,
            outage_active_index=outage.active_branch_index,
            lodf_value=float(lodf_column[monitored]),
        )
        if pair.pair_id != pair_id:
            raise ProvenanceError(f"Security-pair ID did not round-trip: {pair_id}")
        if monitored == outage.active_branch_index:
            raise ProvenanceError(f"Security pair monitors its outaged branch: {pair_id}")
        pairs.append(pair)
    return tuple(pairs)


def deserialize_solution_values(
    solution: dict[str, Any],
    case: Any,
    network: NetworkData,
    master: MasterModel,
) -> FloatArray:
    """Rebuild one complete canonical vector from stable serialized source rows."""

    values = np.full(master.canonical.num_columns, np.nan, dtype=np.float64)
    assigned = np.zeros(master.canonical.num_columns, dtype=bool)

    def set_value(column: int, value: object, *, label: str) -> None:
        if assigned[column]:
            raise ProvenanceError(f"Canonical column assigned twice while loading {label}")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ProvenanceError(f"Nonfinite canonical value while loading {label}")
        values[column] = numeric
        assigned[column] = True

    generators = solution.get("generators", [])
    by_generator = {int(record["source_row"]): record for record in generators}
    if len(by_generator) != len(generators) or len(by_generator) != case.gen.shape[0]:
        raise ProvenanceError("CPU solution must contain each generator source row once")
    for generator_index, source in enumerate(case.gen):
        row = generator_index + 1
        record = by_generator[row]
        if source[GEN_STATUS] <= 0:
            if (
                abs(float(record["commitment"])) > 1e-12
                or abs(float(record["dispatch_mw"])) > 1e-12
                or record.get("segment_dispatch_mw")
            ):
                raise ProvenanceError(f"Source-offline generator row {row} is not zero")
            continue
        set_value(
            master.index.commitment_by_generator[generator_index],
            record["commitment"],
            label=f"generator {row} commitment",
        )
        set_value(
            master.index.dispatch_by_generator[generator_index],
            record["dispatch_mw"],
            label=f"generator {row} dispatch",
        )
        segments = list(record.get("segment_dispatch_mw", []))
        columns = master.index.segments_by_generator[generator_index]
        if len(segments) != len(columns):
            raise ProvenanceError(f"Generator row {row} does not contain all PWL segments")
        for segment, (column, segment_value) in enumerate(
            zip(columns, segments, strict=True), start=1
        ):
            if column is None:
                if abs(float(segment_value)) > 1e-12:
                    raise ProvenanceError(
                        f"Zero-width segment is nonzero at generator {row}, segment {segment}"
                    )
                continue
            set_value(
                column,
                segment_value,
                label=f"generator {row} segment {segment}",
            )
    angle_records = solution.get("bus_angles_rad", [])
    angle_by_bus = {int(record["bus"]): record for record in angle_records}
    if len(angle_by_bus) != len(angle_records) or len(angle_by_bus) != len(network.bus_ids):
        raise ProvenanceError("CPU solution must contain each bus angle once")
    for bus_index, bus_id in enumerate(network.bus_ids):
        set_value(
            int(master.index.theta_by_bus[bus_index]),
            angle_by_bus[int(bus_id)]["angle_rad"],
            label=f"bus {int(bus_id)} angle",
        )
    flow_records = solution.get("base_branch_flows_mw", [])
    flow_by_row = {int(record["source_row"]): record for record in flow_records}
    if (
        len(flow_by_row) != len(flow_records)
        or len(flow_by_row) != len(network.active_branch_source_rows)
    ):
        raise ProvenanceError("CPU solution must contain each active branch flow once")
    for active_index, source_index in enumerate(network.active_branch_source_rows):
        row = int(source_index) + 1
        set_value(
            int(master.index.flow_by_active_branch[active_index]),
            flow_by_row[row]["flow_mw"],
            label=f"branch {row} flow",
        )
    if not np.all(assigned) or not np.all(np.isfinite(values)):
        missing = np.flatnonzero(~assigned).tolist()
        raise ProvenanceError(f"CPU solution left canonical columns unassigned: {missing[:10]}")
    return values


def canonical_feasibility_audit(
    model: CanonicalMILP, values: FloatArray, *, expected_objective: float
) -> dict[str, Any]:
    objective, column_lower, column_upper, integrality = model.column_arrays()
    row_lower, row_upper = model.row_bound_arrays()
    activity = np.asarray(model.matrix_csr() @ values, dtype=np.float64)
    lower_row = np.where(np.isfinite(row_lower), row_lower - activity, -np.inf)
    upper_row = np.where(np.isfinite(row_upper), activity - row_upper, -np.inf)
    row_violation = np.maximum(lower_row, upper_row)
    row_index = int(np.argmax(row_violation))
    lower_column = np.where(
        np.isfinite(column_lower), column_lower - values, -np.inf
    )
    upper_column = np.where(
        np.isfinite(column_upper), values - column_upper, -np.inf
    )
    column_violation = np.maximum(lower_column, upper_column)
    column_index = int(np.argmax(column_violation))
    integer_columns = np.flatnonzero(integrality)
    integer_violation = (
        np.abs(values[integer_columns] - np.rint(values[integer_columns]))
        if integer_columns.size
        else np.asarray([0.0])
    )
    integer_local_index = int(np.argmax(integer_violation))
    integer_column = (
        int(integer_columns[integer_local_index]) if integer_columns.size else None
    )
    calculated_objective = float(objective @ values)
    return {
        "maximum_row_violation": float(max(row_violation[row_index], 0.0)),
        "maximum_row_violation_name": model.row_names[row_index],
        "maximum_column_bound_violation": float(
            max(column_violation[column_index], 0.0)
        ),
        "maximum_column_bound_violation_name": model.variable_names[column_index],
        "maximum_integrality_violation": float(np.max(integer_violation)),
        "maximum_integrality_violation_name": (
            None if integer_column is None else model.variable_names[integer_column]
        ),
        "objective_recalculated": calculated_objective,
        "objective_expected": float(expected_objective),
        "objective_difference": abs(calculated_objective - expected_objective),
    }


def _solve_summary(result: SolveResult) -> dict[str, Any]:
    return {
        "solver": result.solver,
        "solver_version": result.solver_version,
        "status": result.status,
        "optimal": result.optimal,
        "requested_gap_certified": result.requested_gap_certified,
        "has_incumbent": result.has_incumbent,
        "objective": result.objective,
        "bound": result.bound,
        "mip_gap": result.mip_gap,
        "solve_time_seconds": result.solve_time_seconds,
        "statistics": result.statistics,
    }


def run_seeded_round2_worker(
    config: RunConfig, *, checkpoint_path: Path
) -> dict[str, Any]:
    """Build, audit, and solve the fixed round-2 master exactly once."""

    validate_diagnostic_identity(config)
    guard_runtime_environment(config.root)
    validate_platform(config, "dgx_spark")
    checkpoint = guard_output_path(checkpoint_path)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "case_name": config.case_name,
        "benchmark_id": config.benchmark_id,
        "platform": "dgx_spark",
        "diagnostic": True,
        "one_shot": True,
        "status": "building",
        "solver_launched": False,
        "constraint_generation_enabled": False,
        "fixed_master_source_round": 2,
        "timings_seconds": {},
    }

    def save() -> None:
        payload["worker_elapsed_seconds"] = time.perf_counter() - started
        write_json_atomic(payload, checkpoint)

    with PeakMemorySampler(sample_gpu=True) as memory:
        payload["environment"] = environment_manifest("dgx_spark")
        save()
        stage = time.perf_counter()
        cpu_path, cpu, gpu_path, gpu = validate_reference_evidence(config)
        payload["timings_seconds"]["reference_evidence_validation"] = (
            time.perf_counter() - stage
        )
        payload["reference_evidence"] = {
            "secure_cpu_result": {
                "path": str(cpu_path.relative_to(config.root)),
                "sha256": CPU_RESULT_SHA256,
                "commit": CPU_RESULT_COMMIT,
                "objective": EXPECTED_CPU_OBJECTIVE,
            },
            "gpu_v3_round2_result": {
                "path": str(gpu_path.relative_to(config.root)),
                "sha256": GPU_V3_RESULT_SHA256,
                "commit": GPU_V3_RESULT_COMMIT,
                "security_pair_count": EXPECTED_SECURITY_ROWS,
            },
        }
        stage = time.perf_counter()
        case = read_matpower_case(
            config.case_path,
            expected_sha256=config.raw["raw_inputs"]["case_sha256"],
        )
        contingency_table = read_contingency_table(
            config.contingency_path,
            expected_sha256=config.raw["raw_inputs"]["contingency_sha256"],
        )
        payload["source_manifest"] = build_source_manifest(case, contingency_table)
        payload["timings_seconds"]["raw_input_loading"] = time.perf_counter() - stage
        save()
        stage = time.perf_counter()
        network = build_network(case)
        catalog = build_contingency_catalog(
            case,
            network,
            contingency_table,
            validation_columns=int(config.model["lodf_validation_columns"]),
            validation_tolerance_pu=float(
                config.model["lodf_validation_tolerance_pu"]
            ),
            chunk_columns=int(config.model["lodf_build_chunk_columns"]),
        )
        master = build_master(case, network, segments=int(config.model["pwl_segments"]))
        pair_ids = list(gpu["added_security_pair_ids"])
        pairs = security_pairs_from_ids(pair_ids, network, catalog)
        if master.canonical.num_rows != EXPECTED_BASE_MODEL_ROWS:
            raise ProvenanceError(
                f"Base model row count changed: {master.canonical.num_rows}"
            )
        add_security_pairs(master.canonical, master.index, network, pairs)
        dimensions = {
            "columns": master.canonical.num_columns,
            "rows": master.canonical.num_rows,
            "nonzeros": int(master.canonical.matrix_csr().nnz),
        }
        if dimensions != EXPECTED_MODEL_DIMENSIONS:
            raise ProvenanceError(
                f"Round-2 canonical dimensions changed: {dimensions!r}"
            )
        if master.canonical.row_names[-EXPECTED_SECURITY_ROWS:] != pair_ids:
            raise ProvenanceError("Round-2 security-row ordering changed")
        payload["fixed_master_dimensions"] = dimensions
        payload["fixed_security_pair_ids"] = pair_ids
        payload["contingencies"] = contingency_catalog_report(catalog)
        payload["timings_seconds"]["fixed_round2_master_build"] = (
            time.perf_counter() - stage
        )
        save()
        stage = time.perf_counter()
        raw_values = deserialize_solution_values(cpu["solution"], case, network, master)
        _, column_lower, column_upper, integrality = master.canonical.column_arrays()
        bound_policy = config.raw["diagnostic"].get(
            "mip_start_bound_policy", "preserve_as_serialized"
        )
        project_to_bounds = (
            bound_policy == "project_numerical_excess_to_exact_bound"
        )
        _, integer_normalized_values = prepare_mip_start(
            raw_values,
            expected_shape=(master.canonical.num_columns,),
            integrality=integrality,
            mode=FULL_MIP_START,
        )
        start_columns, start_values = prepare_mip_start(
            raw_values,
            expected_shape=(master.canonical.num_columns,),
            integrality=integrality,
            mode=FULL_MIP_START,
            lower_bounds=column_lower,
            upper_bounds=column_upper,
            clip_to_bounds=project_to_bounds,
        )
        expected_columns = np.arange(master.canonical.num_columns, dtype=np.int64)
        if not np.array_equal(start_columns, expected_columns):
            raise ProvenanceError("Full cuOpt start does not select every canonical column")
        raw_integrality = float(
            np.max(np.abs(raw_values[integrality > 0] - np.rint(raw_values[integrality > 0])))
        )
        bound_projection_delta = np.abs(start_values - integer_normalized_values)
        pre_projection_audit = canonical_feasibility_audit(
            master.canonical,
            integer_normalized_values,
            expected_objective=float(cpu["objective"]),
        )
        normalized_values = start_values
        seed_audit = canonical_feasibility_audit(
            master.canonical,
            normalized_values,
            expected_objective=float(cpu["objective"]),
        )
        seed_solution = serialize_solution(normalized_values, case, network, master)
        seed_payload = {
            "case_name": config.case_name,
            "objective": seed_audit["objective_recalculated"],
            "solution": seed_solution,
        }
        independent_seed_verification = verify_serialized_solution(config, seed_payload)
        canonical_tolerance = float(
            config.raw["diagnostic"]["canonical_feasibility_tolerance"]
        )
        objective_tolerance = float(
            config.raw["diagnostic"]["objective_identity_tolerance"]
        )
        gate_passed = bool(
            pre_projection_audit["maximum_column_bound_violation"]
            <= canonical_tolerance
            and float(np.max(bound_projection_delta)) <= canonical_tolerance
            and seed_audit["maximum_row_violation"] <= canonical_tolerance
            and seed_audit["maximum_column_bound_violation"] <= canonical_tolerance
            and (
                not project_to_bounds
                or seed_audit["maximum_column_bound_violation"] == 0.0
            )
            and seed_audit["maximum_integrality_violation"] <= canonical_tolerance
            and seed_audit["objective_difference"] <= objective_tolerance
            and independent_seed_verification.passed
        )
        payload["seed_audit"] = {
            **seed_audit,
            "raw_cpu_maximum_integrality_violation": raw_integrality,
            "integer_normalization_maximum_delta": float(
                np.max(np.abs(integer_normalized_values - raw_values))
            ),
            "mip_start_bound_policy": bound_policy,
            "bound_projection_count": int(
                np.count_nonzero(bound_projection_delta)
            ),
            "bound_projection_maximum_delta": float(
                np.max(bound_projection_delta)
            ),
            "pre_projection_canonical_audit": pre_projection_audit,
            "selected_mip_start_columns": int(start_columns.size),
            "selected_integer_columns": int(np.count_nonzero(integrality)),
            "mip_start_mode": FULL_MIP_START,
            "canonical_feasibility_tolerance": canonical_tolerance,
            "objective_identity_tolerance": objective_tolerance,
            "independent_exhaustive_verification": (
                independent_seed_verification.as_dict()
            ),
            "gate_passed": gate_passed,
        }
        payload["timings_seconds"]["full_start_reconstruction_and_audit"] = (
            time.perf_counter() - stage
        )
        payload["status"] = "ready_for_exact_round2_cuopt_solve"
        save()
        if not gate_passed:
            payload["status"] = "diagnostic_aborted_before_solver_seed_audit_failed"
            save()
            return payload
        print(
            json.dumps(
                {
                    "diagnostic_event": "full_cpu_start_audit_passed",
                    "canonical_columns": master.canonical.num_columns,
                    "canonical_rows": master.canonical.num_rows,
                    "security_rows": EXPECTED_SECURITY_ROWS,
                    "mip_start_columns": int(start_columns.size),
                    "mip_start_bound_policy": bound_policy,
                    "bound_projection_count": int(
                        np.count_nonzero(bound_projection_delta)
                    ),
                    "maximum_row_violation": seed_audit["maximum_row_violation"],
                    "maximum_security_violation_pu": (
                        independent_seed_verification.maximum_security_violation_pu
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        payload["solver_launched"] = True
        payload["status"] = "exact_round2_cuopt_solve_running"
        save()
        stage = time.perf_counter()
        result = solve_cuopt(
            master.canonical,
            time_limit_seconds=EXPECTED_SOLVER_TIME_LIMIT_SECONDS,
            mip_relative_gap=float(config.model["mip_relative_gap_tolerance"]),
            threads=int(config.raw["platforms"]["dgx_spark"]["solver_threads"]),
            mip_start_values=integer_normalized_values,
            mip_start_mode=FULL_MIP_START,
            clip_mip_start_to_bounds=project_to_bounds,
            log_to_console=True,
            mip_acceptance_policy=str(
                config.raw["platforms"]["dgx_spark"]["mip_acceptance_policy"]
            ),
            mip_certificate_residual_tolerance=float(
                config.raw["platforms"]["dgx_spark"][
                    "mip_certificate_residual_tolerance"
                ]
            ),
        )
        payload["timings_seconds"]["cuopt_exact_round2_solve"] = (
            time.perf_counter() - stage
        )
        payload["solve"] = _solve_summary(result)
        payload.update(
            {
                "solver_status": result.status,
                "objective": result.objective,
                "bound": result.bound,
                "mip_gap": result.mip_gap,
            }
        )
        if result.values is None or not result.has_incumbent:
            if result.statistics.get("native_status") == "Infeasible":
                payload["status"] = (
                    "diagnostic_infeasible_despite_verified_feasible_full_start"
                )
                payload["diagnostic_conclusion"] = (
                    "The exact round-2 canonical master had a verified feasible full CPU "
                    "start before translation, yet cuOpt returned native Infeasible. "
                    "Failure to discover a commitment is excluded; the remaining fault "
                    "domain is cuOpt model translation or native solver behavior."
                )
            else:
                payload["status"] = "diagnostic_no_incumbent_from_verified_full_start"
                payload["diagnostic_conclusion"] = (
                    "cuOpt did not return an incumbent despite receiving a verified feasible "
                    "full start; inspect the native console log and translation statistics."
                )
        else:
            stage = time.perf_counter()
            result_audit = canonical_feasibility_audit(
                master.canonical,
                result.values,
                expected_objective=float(result.objective),
            )
            solution = serialize_solution(result.values, case, network, master)
            payload["solution"] = solution
            screened = ContingencyScreener(
                network,
                catalog,
                backend="cupy",
                chunk_columns=int(config.model["screen_chunk_columns"]),
            ).screen(
                np.asarray(result.values[master.index.flow_by_active_branch]),
                tolerance_pu=float(config.model["security_violation_tolerance_pu"]),
            )
            verification = verify_serialized_solution(config, payload)
            payload["result_audit"] = result_audit
            payload["result_exhaustive_screen"] = {
                "evaluated_sides": screened.evaluated_pairs,
                "violations_above_tolerance": len(screened.violations),
                "maximum_violation_pu": screened.maximum_violation_pu,
                "maximum_pair_id": screened.maximum_pair_id,
            }
            payload["result_independent_verification"] = verification.as_dict()
            payload["timings_seconds"]["result_audit_and_exhaustive_verification"] = (
                time.perf_counter() - stage
            )
            accepted = bool(
                result.requested_gap_certified
                and not screened.violations
                and screened.maximum_violation_pu
                <= float(config.model["security_violation_tolerance_pu"])
                and verification.passed
            )
            payload["status"] = (
                "diagnostic_full_start_accepted_and_verified"
                if accepted
                else "diagnostic_incumbent_returned_but_acceptance_gate_failed"
            )
            payload["diagnostic_conclusion"] = (
                "cuOpt accepted the full CPU state and returned an independently verified "
                "secure incumbent."
                if accepted
                else "cuOpt returned an incumbent, but at least one requested acceptance "
                "gate did not pass."
            )
        save()
    payload["peak_memory"] = {
        "process_rss_bytes": memory.peak_process_rss_bytes,
        "cupy_pool_used_bytes": memory.peak_gpu_pool_used_bytes,
        "cuda_device_memory_delta_bytes": memory.peak_cuda_device_memory_delta_bytes,
    }
    payload["worker_wall_time_seconds"] = time.perf_counter() - started
    return payload


def _registry_path(config: RunConfig) -> Path:
    return guard_output_path(
        config.root
        / "results"
        / "diagnostics"
        / f"{config.benchmark_id}-run-registry.json"
    )


def run_one_shot_seeded_round2_diagnostic(
    config: RunConfig, *, output_path: Path
) -> dict[str, Any]:
    """Register and launch the separately authorized diagnostic exactly once."""

    validate_diagnostic_identity(config)
    validate_platform(config, "dgx_spark")
    validate_reference_evidence(config)
    output = guard_output_path(output_path)
    required_root = (config.root / "results" / "diagnostics").resolve()
    if not output.resolve().is_relative_to(required_root):
        raise ScopfError("Seeded diagnostic output must be under results/diagnostics")
    if output.exists():
        raise ScopfError(f"Seeded diagnostic output already exists: {output}")
    identity = frozen_identity(config)
    checkpoint = guard_output_path(
        config.root
        / "results"
        / "checkpoints"
        / f"{config.benchmark_id}-dgx_spark.json"
    )
    if checkpoint.exists():
        raise ScopfError(f"Seeded diagnostic checkpoint already exists: {checkpoint}")
    registry_path = _registry_path(config)
    registry = (
        _read_json(registry_path)
        if registry_path.exists()
        else {
            "schema_version": "1.0.0",
            "diagnostic_id": config.benchmark_id,
            "runs": {},
        }
    )
    if registry.get("runs", {}).get("dgx_spark") is not None:
        prior = registry["runs"]["dgx_spark"]
        raise ScopfError(
            "Seeded round-2 diagnostic already has a one-shot record with status "
            f"{prior.get('status', 'unknown')}; rerun is forbidden"
        )
    registry["runs"]["dgx_spark"] = {
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "output": str(output),
        "checkpoint": str(checkpoint),
        "frozen_identity": identity,
        "secure_cpu_result_sha256": CPU_RESULT_SHA256,
        "gpu_v3_round2_result_sha256": GPU_V3_RESULT_SHA256,
    }
    write_json_atomic(registry, registry_path)
    command = [
        sys.executable,
        "-m",
        "activsg_scopf.cli",
        "_seeded_round2_worker",
        "--config",
        str(config.path),
        "--output",
        str(output),
        "--checkpoint",
        str(checkpoint),
    ]
    deadline = float(config.runtime["deadline_seconds"])
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=config.root,
            check=False,
            timeout=deadline,
        )
        wall_time = time.perf_counter() - started
        if output.exists():
            result = _read_json(output)
        else:
            result = _read_json(checkpoint) if checkpoint.exists() else {}
            result.update(
                {
                    "status": "diagnostic_worker_exited_without_result",
                    "worker_returncode": completed.returncode,
                }
            )
    except subprocess.TimeoutExpired:
        wall_time = time.perf_counter() - started
        result = _read_json(checkpoint) if checkpoint.exists() else {}
        result.update(
            {
                "status": "diagnostic_hard_deadline_exceeded",
                "worker_timeout_seconds": deadline,
            }
        )
    result.update(
        {
            "diagnostic": True,
            "one_shot": True,
            "platform": "dgx_spark",
            "frozen_identity": identity,
            "deadline_seconds": deadline,
            "total_wall_time_seconds": wall_time,
            "benchmark_boundary": (
                "worker launch through evidence validation, raw loading, exact round-2 "
                "master construction, full-start audit, one console-logged cuOpt solve, "
                "post-solve exhaustive verification when applicable, and serialization"
            ),
        }
    )
    write_json_atomic(result, output)
    registry["runs"]["dgx_spark"].update(
        {
            "status": str(result.get("status", "unknown")),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "total_wall_time_seconds": wall_time,
            "solver_launched": result.get("solver_launched"),
            "solver_status": result.get("solver_status"),
            "objective": result.get("objective"),
            "bound": result.get("bound"),
            "mip_gap": result.get("mip_gap"),
        }
    )
    write_json_atomic(registry, registry_path)
    return result


def run_seeded_round2_worker_serialized(
    config: RunConfig, *, output_path: Path, checkpoint_path: Path
) -> dict[str, Any]:
    """Worker boundary that preserves any failure after the one-shot registration."""

    output = guard_output_path(output_path)
    checkpoint = guard_output_path(checkpoint_path)
    try:
        result = run_seeded_round2_worker(config, checkpoint_path=checkpoint)
    except DeadlineExceeded as exc:
        result = _read_json(checkpoint) if checkpoint.exists() else {}
        result.update(
            {
                "status": "diagnostic_deadline_budget_exhausted",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    except Exception as exc:
        result = _read_json(checkpoint) if checkpoint.exists() else {}
        result.update(
            {
                "status": "diagnostic_failed_exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    write_json_atomic(result, output)
    return result
