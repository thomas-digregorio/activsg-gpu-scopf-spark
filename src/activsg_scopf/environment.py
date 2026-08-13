"""Runtime identity checks for honest system-to-system reporting."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from typing import Any

import psutil

from .config import RunConfig
from .errors import ScopfError


def _cpu_model() -> str:
    if os.name == "nt":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = "/proc/cpuinfo"
    if os.path.isfile(cpuinfo):
        with open(cpuinfo, encoding="utf-8") as stream:
            for line in stream:
                if line.casefold().startswith(("model name", "hardware")):
                    return line.split(":", 1)[-1].strip()
    return platform.processor()


def validate_platform(config: RunConfig, platform_name: str) -> None:
    profile = config.raw["platforms"].get(platform_name)
    if profile is None:
        raise ScopfError(f"Unknown configured platform: {platform_name}")
    machine = platform.machine().casefold()
    if platform_name == "laptop_cpu":
        if os.name != "nt":
            raise ScopfError("laptop_cpu official profile is restricted to the Windows laptop")
        if profile["solver"] != "highs" or profile["screening"] != "numpy":
            raise ScopfError("laptop_cpu must use HiGHS and NumPy")
    elif platform_name == "dgx_spark":
        if machine not in {"aarch64", "arm64"}:
            raise ScopfError("dgx_spark official profile requires an ARM64 runtime")
        if profile["solver"] != "cuopt" or profile["screening"] != "cupy":
            raise ScopfError("dgx_spark must use cuOpt and CuPy")
        try:
            import cuopt
            import cupy as cp
        except ImportError as exc:
            raise ScopfError("DGX Spark profile requires both CuPy and cuOpt") from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise ScopfError("DGX Spark profile cannot see a CUDA device")
        observed_version = str(getattr(cuopt, "__version__", "unknown"))
        if observed_version != profile["cuopt_version"]:
            raise ScopfError(
                f"cuOpt version mismatch: expected {profile['cuopt_version']}, "
                f"observed {observed_version}"
            )
        if profile.get("pricing_solver") == "highs":
            try:
                import highspy
            except ImportError as exc:
                raise ScopfError(
                    "DGX Spark fixed-commitment pricing requires HiGHS"
                ) from exc
            observed_highs = highspy.Highs().version()
            expected_highs = str(profile.get("highspy_version", ""))
            if observed_highs != expected_highs:
                raise ScopfError(
                    "HiGHS pricing version mismatch: "
                    f"expected {expected_highs}, observed {observed_highs}"
                )
        expected_image = profile["container_image"]
        observed_image = os.environ.get("ACTIVSG_CUOPT_IMAGE")
        if observed_image != expected_image:
            raise ScopfError(
                "DGX Spark container identity is not registered: "
                f"expected {expected_image}, observed {observed_image!r}"
            )
    else:
        raise ScopfError(f"Unsupported platform profile: {platform_name}")


def environment_manifest(platform_name: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "platform_profile": platform_name,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": _cpu_model(),
        "python": platform.python_version(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_memory_bytes": int(psutil.virtual_memory().total),
        "gpu_used": platform_name == "dgx_spark",
        "packages": {},
    }
    for package in ("numpy", "scipy", "highspy", "psutil", "cupy-cuda13x", "cuopt"):
        try:
            manifest["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    if platform_name == "dgx_spark":
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode()
        manifest["cuda_device"] = {
            "name": name,
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "total_global_memory_bytes": int(properties["totalGlobalMem"]),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        }
    return manifest
