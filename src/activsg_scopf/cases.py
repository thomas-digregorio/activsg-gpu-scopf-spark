"""Fail-closed registry for the explicitly approved ACTIVSg source cases."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ProvenanceError, ScopeViolation


@dataclass(frozen=True)
class CaseRegistration:
    case_name: str
    case_file: str
    case_sha256: str
    contingency_file: str | None
    contingency_sha256: str | None
    expected_buses: int
    expected_generators: int
    expected_branches: int
    contingency_mode: str = "source_table"
    text_encoding: str = "utf-8"
    source_archive_file: str | None = None
    source_archive_sha256: str | None = None


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
    "ACTIVSg2000": CaseRegistration(
        case_name="ACTIVSg2000",
        case_file="case_ACTIVSg2000.m",
        case_sha256="8d00618de8fd10bf35a599f59d2deebfecd0d86e28fcff73219ad7c4ebab860b",
        contingency_file="contab_ACTIVSg2000.m",
        contingency_sha256=(
            "198b39f0381925a4ddacbe2148973cb1d93ddfe220303829cf87b16d45190bba"
        ),
        expected_buses=2_000,
        expected_generators=544,
        expected_branches=3_206,
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
    "Texas2kSeries24Case1": CaseRegistration(
        case_name="Texas2kSeries24Case1",
        case_file="Texas2k_series24_case1_2016summerPeak.m",
        case_sha256="4eb378bb7c5abeba02a1be61752983d90a17dcaac3460ea2bb0c4986d5aa00b2",
        contingency_file=None,
        contingency_sha256=None,
        expected_buses=2_000,
        expected_generators=544,
        expected_branches=3_220,
        contingency_mode="enumerate_in_service_branches",
        text_encoding="cp1252",
        source_archive_file="Texas2k_series24_cases_with_dynamics.zip",
        source_archive_sha256=(
            "954c515a55c186bf1d987efdde0e2873b4a694bc4bf3522c47bb2b1ab58a0fb9"
        ),
    ),
    "Texas2kSeries24Case2": CaseRegistration(
        case_name="Texas2kSeries24Case2",
        case_file="Texas2k_series24_case2_2016lowload.m",
        case_sha256="29e8a640fc7f0e6660746a7784ae1168ac7f17b5e5741dae449247cd9525b819",
        contingency_file=None,
        contingency_sha256=None,
        expected_buses=2_000,
        expected_generators=544,
        expected_branches=3_220,
        contingency_mode="enumerate_in_service_branches",
        text_encoding="cp1252",
        source_archive_file="Texas2k_series24_cases_with_dynamics.zip",
        source_archive_sha256=(
            "954c515a55c186bf1d987efdde0e2873b4a694bc4bf3522c47bb2b1ab58a0fb9"
        ),
    ),
    "Texas2kSeries24Case3": CaseRegistration(
        case_name="Texas2kSeries24Case3",
        case_file="Texas2k_series24_case3_2024summerpeak.m",
        case_sha256="3f2243b14d73bf923ac0b5deee3bf56f9e93aa0f6d6afbfc5618bf9b9d761e46",
        contingency_file=None,
        contingency_sha256=None,
        expected_buses=2_000,
        expected_generators=743,
        expected_branches=3_911,
        contingency_mode="enumerate_in_service_branches",
        text_encoding="cp1252",
        source_archive_file="Texas2k_series24_cases_with_dynamics.zip",
        source_archive_sha256=(
            "954c515a55c186bf1d987efdde0e2873b4a694bc4bf3522c47bb2b1ab58a0fb9"
        ),
    ),
    "Texas2kSeries24Case4": CaseRegistration(
        case_name="Texas2kSeries24Case4",
        case_file="Texas2k_series24_case4_2024lowload.m",
        case_sha256="f69585bcca1b2bc601a257d3ef2c8181d052e7ecc5dd739962c64b32c6ee313f",
        contingency_file=None,
        contingency_sha256=None,
        expected_buses=2_000,
        expected_generators=743,
        expected_branches=3_913,
        contingency_mode="enumerate_in_service_branches",
        text_encoding="cp1252",
        source_archive_file="Texas2k_series24_cases_with_dynamics.zip",
        source_archive_sha256=(
            "954c515a55c186bf1d987efdde0e2873b4a694bc4bf3522c47bb2b1ab58a0fb9"
        ),
    ),
    "Texas2kSeries24Case5": CaseRegistration(
        case_name="Texas2kSeries24Case5",
        case_file="Texas2k_series24_case5_2024highrenewables.m",
        case_sha256="c7d96c0dfe560509eaeecf0de0d9767274cd7b79318cd4b5bf0ebff1468b1d85",
        contingency_file=None,
        contingency_sha256=None,
        expected_buses=2_000,
        expected_generators=743,
        expected_branches=3_663,
        contingency_mode="enumerate_in_service_branches",
        text_encoding="cp1252",
        source_archive_file="Texas2k_series24_cases_with_dynamics.zip",
        source_archive_sha256=(
            "954c515a55c186bf1d987efdde0e2873b4a694bc4bf3522c47bb2b1ab58a0fb9"
        ),
    ),
    "Texas2kSeries24Case6": CaseRegistration(
        case_name="Texas2kSeries24Case6",
        case_file="Texas2k_series24_case6_2024lowloadwithgfm.m",
        case_sha256="8710b40e9eafb38463d9a1135a59e70f14fbad38c99ca90af598928a5d0e7b86",
        contingency_file=None,
        contingency_sha256=None,
        expected_buses=2_000,
        expected_generators=748,
        expected_branches=3_913,
        contingency_mode="enumerate_in_service_branches",
        text_encoding="cp1252",
        source_archive_file="Texas2k_series24_cases_with_dynamics.zip",
        source_archive_sha256=(
            "954c515a55c186bf1d987efdde0e2873b4a694bc4bf3522c47bb2b1ab58a0fb9"
        ),
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
        if getattr(item, attribute) is not None
        and str(getattr(item, attribute)).casefold() == filename.casefold()
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
