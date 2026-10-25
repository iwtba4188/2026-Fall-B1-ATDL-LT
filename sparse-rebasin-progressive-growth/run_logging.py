"""Small local run logger using only Python's standard library."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_logger = logging.getLogger("sparse_rebasin")
_metrics_file: Path | None = None


def start_run(name: str, config: dict[str, Any], run_dir: str | Path) -> None:
    """Start a local run and write its metadata to disk."""
    global _metrics_file

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _metrics_file = output_dir / f"{name}_metrics.jsonl"

    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.info("Started run %s", name)

    log_metrics({"event": "run_started", "run_name": name, "config": config})


def log_metrics(metrics: dict[str, Any]) -> None:
    """Append one JSON-serializable metrics record to the active run."""
    if _metrics_file is None:
        raise RuntimeError("A run must be started before logging metrics")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    with _metrics_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")


def finish_run() -> None:
    """Record the end of the active run."""
    if _metrics_file is not None:
        log_metrics({"event": "run_finished"})
        _logger.info("Finished run")
