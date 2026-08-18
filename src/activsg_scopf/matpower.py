"""Minimal, non-executing parser for the registered MATPOWER v2 sources."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .cases import infer_registered_case, validate_registered_dimensions
from .errors import ProvenanceError
from .paths import guard_input_path

FloatArray = npt.NDArray[np.float64]

BUS_I = 0
BUS_TYPE = 1
PD = 2
GS = 4
GEN_BUS = 0
GEN_STATUS = 7
PMAX = 8
PMIN = 9
F_BUS = 0
T_BUS = 1
BR_X = 3
RATE_A = 5
TAP = 8
SHIFT = 9
BR_STATUS = 10
ANGMIN = 11
ANGMAX = 12

EXPECTED_CASE_SHA256 = "8ca6d54ea5179eeb03fe29d7b645618e7a86338c172247e81687476660f6dcbe"
EXPECTED_CONTINGENCY_SHA256 = (
    "f6b2e7e38fd1cf5e09e877cf04233b4d0487d6d0e99903070d519eade12b76a9"
)

_MATRIX_START = re.compile(r"^\s*mpc\.(bus|gen|branch|gencost)\s*=\s*\[")
_BASE_MVA = re.compile(r"^\s*mpc\.baseMVA\s*=\s*([^;]+);")
_CONTAB_START = re.compile(r"^\s*chgtab\s*=\s*\[")


@dataclass(frozen=True)
class MatpowerCase:
    case_name: str
    source_path: Path
    sha256: str
    base_mva: float
    bus: FloatArray
    gen: FloatArray
    branch: FloatArray
    gencost: FloatArray


@dataclass(frozen=True)
class ContingencyChange:
    source_row: int
    label: int
    probability: float
    table: str
    element_row: int
    column: str
    change_type: str
    new_value: float


@dataclass(frozen=True)
class ContingencyTable:
    source_path: Path
    sha256: str
    changes: tuple[ContingencyChange, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _without_comment(line: str) -> str:
    return line.split("%", 1)[0].strip()


def _numeric_rows(lines: list[str], start: int, name: str) -> tuple[FloatArray, int]:
    rows: list[list[float]] = []
    for index in range(start, len(lines)):
        clean = _without_comment(lines[index])
        if not clean:
            continue
        if clean == "];":
            if not rows:
                raise ProvenanceError(f"MATPOWER matrix {name} is empty")
            widths = {len(row) for row in rows}
            if len(widths) != 1:
                raise ProvenanceError(f"MATPOWER matrix {name} has ragged rows: {sorted(widths)}")
            return np.asarray(rows, dtype=np.float64), index
        if clean.endswith(";"):
            clean = clean[:-1].strip()
        try:
            rows.append([float(token) for token in clean.replace(",", " ").split()])
        except ValueError as exc:
            message = f"Non-numeric token in mpc.{name} source line {index + 1}"
            raise ProvenanceError(message) from exc
    raise ProvenanceError(f"Unterminated MATPOWER matrix mpc.{name}")


def read_matpower_case(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> MatpowerCase:
    source = guard_input_path(path)
    registration = infer_registered_case(source.name)
    expected_sha256 = expected_sha256 or registration.case_sha256
    observed_hash = sha256_file(source)
    if expected_sha256 and observed_hash.casefold() != expected_sha256.casefold():
        raise ProvenanceError(
            f"{registration.case_name} case hash mismatch: expected {expected_sha256}, "
            f"observed {observed_hash}"
        )
    lines = source.read_text(encoding="utf-8").splitlines()
    base_mva: float | None = None
    matrices: dict[str, FloatArray] = {}
    index = 0
    while index < len(lines):
        clean = _without_comment(lines[index])
        base_match = _BASE_MVA.match(clean)
        if base_match:
            base_mva = float(base_match.group(1))
        matrix_match = _MATRIX_START.match(clean)
        if matrix_match:
            name = matrix_match.group(1)
            matrix, index = _numeric_rows(lines, index + 1, name)
            matrices[name] = matrix
        index += 1
    missing = {"bus", "gen", "branch", "gencost"} - matrices.keys()
    if base_mva is None or not np.isfinite(base_mva) or base_mva <= 0:
        raise ProvenanceError("Missing or invalid positive mpc.baseMVA")
    if missing:
        raise ProvenanceError(f"Missing MATPOWER matrices: {sorted(missing)}")
    case = MatpowerCase(
        case_name=registration.case_name,
        source_path=source,
        sha256=observed_hash,
        base_mva=base_mva,
        bus=matrices["bus"],
        gen=matrices["gen"],
        branch=matrices["branch"],
        gencost=matrices["gencost"],
    )
    validate_case(case)
    if expected_sha256.casefold() == registration.case_sha256.casefold():
        validate_registered_dimensions(
            registration,
            buses=case.bus.shape[0],
            generators=case.gen.shape[0],
            branches=case.branch.shape[0],
        )
    return case


def validate_case(case: MatpowerCase) -> None:
    if case.bus.shape[1] < 13 or case.gen.shape[1] < 10 or case.branch.shape[1] < 13:
        raise ProvenanceError("MATPOWER bus/gen/branch matrix has too few columns")
    if case.gencost.shape[0] != case.gen.shape[0]:
        raise ProvenanceError("Each generator row must have exactly one production-cost row")
    numeric_matrices = (case.bus, case.gen, case.branch, case.gencost)
    if not all(np.all(np.isfinite(matrix)) for matrix in numeric_matrices):
        raise ProvenanceError("Non-finite MATPOWER numeric value is not permitted")
    bus_ids = case.bus[:, BUS_I].astype(np.int64)
    if len(np.unique(bus_ids)) != len(bus_ids):
        raise ProvenanceError("Bus identifiers are not unique")
    if not np.all(case.bus[:, BUS_I] == bus_ids):
        raise ProvenanceError("Bus identifiers must be integers")
    known = set(bus_ids.tolist())
    referenced = np.concatenate((case.gen[:, GEN_BUS], case.branch[:, [F_BUS, T_BUS]].ravel()))
    if any(value != int(value) or int(value) not in known for value in referenced):
        raise ProvenanceError("Generator or branch references an unknown/non-integer bus")
    if np.any(case.gen[:, PMIN] > case.gen[:, PMAX]):
        raise ProvenanceError("A generator has PMIN greater than PMAX")
    if np.any(case.gencost[:, 0] != 2):
        raise ProvenanceError("Version 1 accepts source polynomial production costs only")
    for row, cost in enumerate(case.gencost, start=1):
        coefficient_count = int(cost[3])
        if cost[3] != coefficient_count or coefficient_count < 1:
            raise ProvenanceError(f"Invalid polynomial coefficient count at gencost row {row}")
        if 4 + coefficient_count > cost.size:
            raise ProvenanceError(f"Truncated polynomial at gencost row {row}")


def read_contingency_table(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ContingencyTable:
    source = guard_input_path(path)
    registration = infer_registered_case(source.name, contingency=True)
    expected_sha256 = expected_sha256 or registration.contingency_sha256
    observed_hash = sha256_file(source)
    if expected_sha256 and observed_hash.casefold() != expected_sha256.casefold():
        raise ProvenanceError(
            f"{registration.case_name} contingency hash mismatch: "
            f"expected {expected_sha256}, observed {observed_hash}"
        )
    lines = source.read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if _CONTAB_START.match(_without_comment(line))),
        None,
    )
    if start is None:
        raise ProvenanceError("Missing chgtab matrix")
    changes: list[ContingencyChange] = []
    for line_index in range(start + 1, len(lines)):
        clean = _without_comment(lines[line_index])
        if not clean:
            continue
        if clean == "];":
            break
        if clean.endswith(";"):
            clean = clean[:-1]
        tokens = clean.replace(",", " ").split()
        if len(tokens) != 7:
            raise ProvenanceError(f"Invalid chgtab row at source line {line_index + 1}")
        try:
            change = ContingencyChange(
                source_row=len(changes) + 1,
                label=int(tokens[0]),
                probability=float(tokens[1]),
                table=tokens[2],
                element_row=int(tokens[3]),
                column=tokens[4],
                change_type=tokens[5],
                new_value=float(tokens[6]),
            )
        except ValueError as exc:
            message = f"Invalid numeric chgtab field at source line {line_index + 1}"
            raise ProvenanceError(message) from exc
        changes.append(change)
    else:
        raise ProvenanceError("Unterminated chgtab matrix")
    if not changes:
        raise ProvenanceError("The contingency table is empty")
    return ContingencyTable(source_path=source, sha256=observed_hash, changes=tuple(changes))
