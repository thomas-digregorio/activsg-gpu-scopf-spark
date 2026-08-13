"""Durable JSONL diagnostics that survive worker termination."""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import guard_output_path


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class DiagnosticEventWriter:
    """Append flushed JSON records to one new, guarded local file."""

    def __init__(self, path: Path, *, started: float) -> None:
        self.path = guard_output_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = started
        self._lock = threading.Lock()
        self._stream = self.path.open("x", encoding="utf-8", buffering=1)
        self._closed = False

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "schema_version": "1.0.0",
            "event": event,
            "elapsed_seconds": time.perf_counter() - self.started,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            **fields,
        }
        line = json.dumps(_json_value(record), sort_keys=True, allow_nan=False)
        with self._lock:
            if self._closed:
                return
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._stream.flush()
            self._stream.close()
            self._closed = True

    def __enter__(self) -> DiagnosticEventWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
