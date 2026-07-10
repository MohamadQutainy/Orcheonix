"""
Orcheonix — Structured Logging

Configures logging for both console (human-readable) and file
(machine-readable JSON) output.

Every agent logs: start time, query, tokens used, duration,
confidence score, and errors with full traceback.
"""

import os
import json
import logging
import traceback
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

AGENT_LOG_FILE = os.path.join(LOG_DIR, "agent_runs.jsonl")


class JSONLHandler(logging.Handler):
    """Logging handler that writes structured JSON lines to a file."""

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def emit(self, record: logging.LogRecord):
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            # Attach structured extras if present
            for key in ("agent", "query", "tokens_used", "duration_sec",
                        "confidence", "error", "traceback"):
                if hasattr(record, key):
                    entry[key] = getattr(record, key)
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for console + JSONL file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    # Console handler (human-readable)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    ))
    logger.addHandler(console)

    # JSONL file handler (machine-readable)
    jsonl = JSONLHandler(AGENT_LOG_FILE)
    logger.addHandler(jsonl)

    return logger


def log_agent_run(
    logger: logging.Logger,
    agent: str,
    query: str,
    duration_sec: float,
    confidence: float = 0.0,
    tokens_used: int = 0,
    error: str | None = None,
):
    """Log a structured agent run entry."""
    extra = {
        "agent": agent,
        "query": query[:200],
        "duration_sec": round(duration_sec, 2),
        "confidence": confidence,
        "tokens_used": tokens_used,
    }
    if error:
        extra["error"] = error
        extra["traceback"] = traceback.format_exc()
        logger.error(f"Agent [{agent}] failed in {duration_sec:.1f}s", extra=extra)
    else:
        logger.info(
            f"Agent [{agent}] completed in {duration_sec:.1f}s "
            f"(confidence={confidence:.2f})",
            extra=extra,
        )
