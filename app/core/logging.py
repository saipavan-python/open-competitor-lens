"""Structured JSON logging with secret redaction.

Log records are emitted as one JSON object per line so a run can be reconstructed
from logs alone. Anything that looks like a credential is redacted before it
reaches a handler.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

from app.core.config import get_settings

#: Fields promoted out of ``extra`` onto every record.
CONTEXT_FIELDS = (
    "run_id", "company", "competitor", "agent", "node", "provider", "model",
    "duration_ms", "status", "error", "step", "company_name",
)

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"tvly-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"fc-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|password|secret)"
               r"[\"'\s:=]+[A-Za-z0-9_\-\.]{8,}"),
)


def redact(text: str) -> str:
    """Replace anything credential-shaped in ``text`` with ``[redacted]``."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        for field in CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        bits = [f"{self.formatTime(record, '%H:%M:%S')}", f"{record.levelname:<7}",
                record.name, redact(record.getMessage())]
        context = " ".join(
            f"{field}={getattr(record, field)}"
            for field in CONTEXT_FIELDS
            if getattr(record, field, None) is not None
        )
        if context:
            bits.append(f"| {context}")
        return " ".join(bits)


_configured = False


def configure_logging(*, force: bool = False) -> None:
    """Install the application log handler. Safe to call repeatedly."""
    global _configured
    if _configured and not force:
        return
    settings = get_settings()
    root = logging.getLogger("app")
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else TextFormatter())
    root.addHandler(handler)
    root.setLevel(settings.log_level)
    root.propagate = False
    _configured = True


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter:
    """Return a logger that stamps ``context`` onto every record.

    ``get_logger(__name__, run_id=run_id)`` produces records that can be
    filtered by ``run_id`` without string matching on the message.
    """
    return logging.LoggerAdapter(logging.getLogger(name), context)
