"""Global deadline accounting and lightweight peak-memory sampling."""

from __future__ import annotations

import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field

import psutil

from .errors import DeadlineExceeded


@dataclass
class Deadline:
    total_seconds: float
    verification_reserve_seconds: float
    serialization_reserve_seconds: float
    started: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed)

    def solver_budget(self) -> float:
        budget = (
            self.remaining
            - self.verification_reserve_seconds
            - self.serialization_reserve_seconds
        )
        if budget <= 0:
            raise DeadlineExceeded("No solver budget remains before verification reserve")
        return budget

    def require(self, stage: str, *, reserve_seconds: float = 0.0) -> None:
        if self.remaining <= reserve_seconds:
            raise DeadlineExceeded(f"Global deadline reached before {stage}")


class PeakMemorySampler:
    def __init__(self, interval_seconds: float = 0.05, *, sample_gpu: bool = False) -> None:
        self.interval_seconds = interval_seconds
        self.sample_gpu = sample_gpu
        self.peak_process_rss_bytes = 0
        self.peak_gpu_pool_used_bytes = 0
        self.peak_cuda_device_memory_delta_bytes = 0
        self._baseline_cuda_free_bytes: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def __enter__(self) -> PeakMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sample()

    def _sample(self) -> None:
        process = psutil.Process()
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                rss += child.memory_info().rss
        self.peak_process_rss_bytes = max(self.peak_process_rss_bytes, int(rss))
        if self.sample_gpu:
            try:
                import cupy as cp

                used = int(cp.get_default_memory_pool().used_bytes())
                self.peak_gpu_pool_used_bytes = max(self.peak_gpu_pool_used_bytes, used)
                free_bytes, _ = cp.cuda.runtime.memGetInfo()
                free_bytes = int(free_bytes)
                if self._baseline_cuda_free_bytes is None:
                    self._baseline_cuda_free_bytes = free_bytes
                delta = max(0, self._baseline_cuda_free_bytes - free_bytes)
                self.peak_cuda_device_memory_delta_bytes = max(
                    self.peak_cuda_device_memory_delta_bytes, delta
                )
            except (ImportError, RuntimeError):
                pass

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()
