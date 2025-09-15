from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

__all__ = [
    "setup_logger",
    "bind",
    "JsonFormatter",
    "ContextAdapter",
    "add_file_handler",
]

class JsonFormatter(logging.Formatter):
    """
    Minimal JSON formatter with structured fields + extras.
    """
    _STD = {
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process"
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.filename,
            "func": record.funcName,
            "line": record.lineno,
        }
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in self._STD and not k.startswith("_")
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class ContextAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that ensures persistent context fields appear in every log entry.
    """
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.update(self.extra or {})
        kwargs["extra"] = extra
        return msg, kwargs


class _EnsureContextFilter(logging.Filter):
    """
    Adds default values for context keys so format strings never KeyError.
    """
    def __init__(self, defaults: Mapping[str, Any] | None = None):
        super().__init__()
        self.defaults = dict(defaults or {})

    def filter(self, record: logging.LogRecord) -> bool:
        for k, v in self.defaults.items():
            if not hasattr(record, k):
                setattr(record, k, v)
        return True


def setup_logger(
    name: str = "wisent",
    level: int = logging.INFO,
    *,
    json_logs: bool = False,
    stream = sys.stderr,
) -> logging.Logger:
    """
    Create or return a named logger with a single stream handler.
    Safe to call multiple times; won’t duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        if json_logs:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s "
                    "[file=%(filename)s func=%(funcName)s line=%(lineno)d] "
                    "%(task_name)s%(subtask)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            ))
        # ensure context placeholders always exist
        handler.addFilter(_EnsureContextFilter({"task_name": "", "subtask": ""}))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def add_file_handler(
    logger: logging.Logger,
    filepath: str,
    *,
    level: int | None = None,
    json_logs: bool = False,
) -> None:
    """
    Optionally add a file handler (e.g., for long-running CLI jobs).
    """
    fh = logging.FileHandler(filepath, encoding="utf-8")
    fh.setLevel(level or logger.level)
    if json_logs:
        fh.setFormatter(JsonFormatter())
    else:
        fh.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s "
                "[file=%(filename)s func=%(funcName)s line=%(lineno)d] "
                "%(task_name)s%(subtask)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        ))
    fh.addFilter(_EnsureContextFilter({"task_name": "", "subtask": ""}))
    logger.addHandler(fh)


def bind(
    logger: logging.Logger | ContextAdapter,
    **extra: Any
) -> ContextAdapter:
    """
    Return a ContextAdapter with merged extras.
    Works whether you pass a raw Logger or an existing ContextAdapter.
    """
    if isinstance(logger, ContextAdapter):
        merged = {**logger.extra, **extra}
        return ContextAdapter(logger.logger, merged)
    return ContextAdapter(logger, extra)
