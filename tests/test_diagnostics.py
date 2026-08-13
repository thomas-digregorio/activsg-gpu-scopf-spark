import json
import time
from pathlib import Path

from activsg_scopf.diagnostics import DiagnosticEventWriter


def test_diagnostic_event_writer_flushes_jsonl_and_normalizes_infinity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results" / "diagnostics" / "events.jsonl"
    writer = DiagnosticEventWriter(path, started=time.perf_counter())
    writer.emit("mip_progress", primal_bound=float("inf"), node_count=3)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["event"] == "mip_progress"
    assert record["primal_bound"] is None
    assert record["node_count"] == 3
    assert record["elapsed_seconds"] >= 0.0
    writer.close()
