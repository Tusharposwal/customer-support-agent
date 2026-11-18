"""Logging for the support_agent package.

Only the `support_agent` logger is configured (one handler, on stderr like
Uvicorn's own logs). The root logger and Uvicorn's loggers are left alone, so
Uvicorn keeps its startup and access logs and nothing is printed twice.

Every line has a timestamp, level, logger name and the current request ID
("-" outside a request):

    2026-11-17 10:15:02,123 INFO support_agent.api [3f2a...] chat_request status=200 ...

Settings:
    LOG_LEVEL   DEBUG, INFO, WARNING, ERROR or CRITICAL (default INFO; any case)
"""

import contextvars
import logging
import os
import sys
from collections.abc import Mapping

LOG_LEVEL_ENV = "LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"
PACKAGE_LOGGER = "support_agent"

# The ID of the request being handled. Set by the request middleware in
# api.py; visible in the worker thread that runs /chat (Starlette copies the
# context into it), so agent and Cohere log lines carry it too.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def log_level_from_env(environ: Mapping[str, str] | None = None) -> int:
    """Read LOG_LEVEL. Blank or unset means INFO. Raises ValueError (naming the
    setting, never its value) for anything else."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(LOG_LEVEL_ENV) or "").strip().upper() or DEFAULT_LOG_LEVEL
    if raw not in LEVELS:
        raise ValueError(f"{LOG_LEVEL_ENV} must be one of {', '.join(LEVELS)}")
    return getattr(logging, raw)


class RequestIdFilter(logging.Filter):
    """Adds `request_id` to every record the handler formats."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: int) -> None:
    """Give the support_agent logger its level and one formatted handler.

    Safe to call more than once (each server start): the handler is replaced,
    not added again. `propagate = False` keeps records away from the root
    logger, so a root handler added by a host or tool can't print them twice.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        if getattr(handler, "_support_agent_handler", False):
            logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler._support_agent_handler = True
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(RequestIdFilter())

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
