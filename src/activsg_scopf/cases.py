"""Fail-closed registry for the explicitly approved ACTIVSg source cases."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ProvenanceError, ScopeViolation


@dataclass(frozen=True)
class CaseRegistration:
    case_name: str
    case_file: str
    case_sha256: str
    contingency_file: str
    contingency_sha256: str
    expected_buses: int
    expected_generators: int
    expected_branches: int


CASE_REGISTRY: dict[str, CaseRegistration] = {
    "ACTIVSg500": CaseRegistration(
        case_name="ACTIVSg500",
        case_file="case_ACTIVSg500.m",
        case_sha256="8ca6d54ea5179eeb03fe29d7b645618e7a86338c172247e81687476660f6dcbe",
        contingency_file="contab_ACTIVSg500.m",
        contingency_sha256=(
            "f6b2e7e38fd1cf5e09e877cf04233b4d0487d6d0e99903070d519eade12b76a9"
        ),
        expected_buses=500,
        expected_generators=90,
        expected_branches=597,
    ),
    "ACTIVSg10k": CaseRegistration(
        case_name="ACTIVSg10k",
        case_file="case_ACTIVSg10k.m",
        case_sha256="ead10b25fecc4dcc02f88bacdfb3526fe8b8985b81f7e539c95abddb32575590",
        contingency_file="contab_ACTIVSg10k.m",
        contingency_sha256=(
            "7e1681a960b0a2a99d824766e0e94cc291fa36a7ec33b6dee24bf12ac67ddef7"
        ),
        expected_buses=10_000,
        expected_generators=2_485,
        expected_branches=12_706,
    ),
}

_REGISTRY_CASEFOLD = {name.casefold(): item for name, item in CASE_REGISTRY.items()}


def registered_case(case_name: str) -> CaseRegistration:
    try:
        return _REGISTRY_CASEFOLD[case_name.casefold()]
    except KeyError as exc:
        approved = ", ".join(CASE_REGISTRY)
        raise ScopeViolation(
            f"Only the explicitly approved cases are accepted: {approved}; rejected {case_name!r}"
        ) from exc


def infer_registered_case(filename: str, *, contingency: bool = False) -> CaseRegistration:
    attribute = "contingency_file" if contingency else "case_file"
    matches = [
        item
        for item in CASE_REGISTRY.values()
        if getattr(item, attribute).casefold() == filename.casefold()
    ]
    if len(matches) != 1:
        raise ProvenanceError(f"Unregistered ACTIVSg source filename: {filename}")
    return matches[0]


def validate_registered_dimensions(
    registration: CaseRegistration,
    *,
    buses: int,
    generators: int,
    branches: int,
) -> None:
    observed = (buses, generators, branches)
    expected = (
        registration.expected_buses,
        registration.expected_generators,
        registration.expected_branches,
    )
    if observed != expected:
        raise ProvenanceError(
            f"{registration.case_name} dimensions do not match the registry: "
            f"expected bus/gen/branch {expected}, observed {observed}"
        )
