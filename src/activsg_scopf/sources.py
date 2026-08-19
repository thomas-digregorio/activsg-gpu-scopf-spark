"""Load registered raw cases and either source or topology-derived contingencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig
from .contingency_mapping import map_reference_branch_contingencies
from .errors import ProvenanceError
from .matpower import (
    ContingencyTable,
    MatpowerCase,
    enumerate_in_service_branch_contingencies,
    read_contingency_table,
    read_matpower_case,
    sha256_file,
)


@dataclass(frozen=True)
class LoadedInputs:
    case: MatpowerCase
    contingencies: ContingencyTable
    source_archive_path: Path | None
    source_archive_sha256: str | None


def load_registered_inputs(config: RunConfig) -> LoadedInputs:
    inputs = config.raw["raw_inputs"]
    case = read_matpower_case(
        config.case_path,
        expected_sha256=inputs["case_sha256"],
    )
    archive_path = config.source_archive_path
    archive_hash: str | None = None
    if archive_path is not None:
        archive_hash = sha256_file(archive_path)
        expected_archive_hash = str(inputs["source_archive_sha256"])
        if archive_hash.casefold() != expected_archive_hash.casefold():
            raise ProvenanceError(
                f"Source archive hash mismatch: expected {expected_archive_hash}, "
                f"observed {archive_hash}"
            )
    if config.contingency_mode == "source_table":
        contingencies = read_contingency_table(
            config.contingency_path,
            expected_sha256=inputs["contingency_sha256"],
        )
    elif config.contingency_mode == "enumerate_in_service_branches":
        contingencies = enumerate_in_service_branch_contingencies(case)
    elif config.contingency_mode == "mapped_reference_branch_table":
        reference_case = read_matpower_case(
            config.reference_case_path,
            expected_sha256=inputs["reference_case_sha256"],
        )
        reference_table = read_contingency_table(
            config.reference_contingency_path,
            expected_sha256=inputs["reference_contingency_sha256"],
        )
        contingencies = map_reference_branch_contingencies(
            reference_case,
            case,
            reference_table,
            method=str(inputs["branch_mapping_method"]),
        )
    else:
        raise ProvenanceError(
            f"Unsupported contingency mode {config.contingency_mode!r}"
        )
    return LoadedInputs(
        case=case,
        contingencies=contingencies,
        source_archive_path=archive_path,
        source_archive_sha256=archive_hash,
    )
