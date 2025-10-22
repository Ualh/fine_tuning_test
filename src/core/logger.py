"""Logging utilities with Rich console output and file handlers."""

from __future__ import annotations

import io
import logging
import re
import sys
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console  # type: ignore[import-untyped]
from rich.logging import RichHandler  # type: ignore[import-untyped]
from rich.traceback import Traceback  # type: ignore[import-untyped]

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;?]*[ -/]*[@-~]")


def _sanitize_for_log(data: str) -> str:
    """Return text with ANSI sequences stripped and carriage returns normalised."""

    if not data:
        return data
    cleaned = ANSI_ESCAPE_RE.sub("", data)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned


def configure_logging(
    name: str,
    log_dir: Path,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    enable_rich: bool = True,
) -> logging.Logger:
    """Configure a logger with both console and rotating file outputs."""

    logger = logging.getLogger(name)
    if logger.handlers:  # Reconfigure existing logger
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if enable_rich:
        console = Console(force_terminal=True)
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=False,
            show_time=False,
            show_path=False,
        )
        rich_handler.setLevel(console_level)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(rich_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(console_level)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("trl").setLevel(logging.INFO)

    logger.propagate = False
    logger.debug("Logger initialised; writing to %s", log_file)
    return logger


class _StreamTee(io.TextIOBase):
    def __init__(
        self,
        stream: io.TextIOBase,
        mirror: io.TextIOBase,
        transform: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._stream = stream
        self._mirror = mirror
        self._transform = transform

    def write(self, data: str) -> int:  # type: ignore[override]
        if not data:
            return 0
        written = self._stream.write(data)
        mirror_payload = self._transform(data) if self._transform else data
        self._mirror.write(mirror_payload)
        return written

    def flush(self) -> None:  # type: ignore[override]
        self._stream.flush()
        self._mirror.flush()

    @property
    def encoding(self) -> Optional[str]:  # type: ignore[override]
        return getattr(self._stream, "encoding", None)

    def isatty(self) -> bool:  # type: ignore[override]
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def fileno(self) -> int:  # type: ignore[override]
        if hasattr(self._stream, "fileno"):
            return self._stream.fileno()
        raise OSError("Underlying stream does not expose fileno")


@contextmanager
def tee_std_streams(log_file: Path, strip_ansi: bool = True):
    """Mirror stdout and stderr into ``log_file`` while preserving console output."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    transform = _sanitize_for_log if strip_ansi else None
    with log_file.open("a", encoding="utf-8", buffering=1) as mirror:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        tee_stdout = _StreamTee(original_stdout, mirror, transform=transform)
        tee_stderr = _StreamTee(original_stderr, mirror, transform=transform)
        sys.stdout = tee_stdout  # type: ignore[assignment]
        sys.stderr = tee_stderr  # type: ignore[assignment]
        try:
            yield
        finally:
            mirror.flush()
            sys.stdout = original_stdout  # type: ignore[assignment]
            sys.stderr = original_stderr  # type: ignore[assignment]


def finalize_logger(logger: logging.Logger) -> None:
    """Flush and close all handlers associated with ``logger``."""

    for handler in list(logger.handlers):
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)


def format_exception_with_locals(exc: BaseException, width: int = 120) -> str:
    """Render an exception with locals captured via Rich and return plain text."""

    traceback = Traceback.from_exception(type(exc), exc, exc.__traceback__, show_locals=True)
    console = Console(width=width, record=True, force_terminal=False, color_system=None)
    console.print(traceback)
    return console.export_text(clear=True)


def log_exception_with_locals(logger: logging.Logger, message: str, exc: BaseException) -> None:
    """Log a failure message followed by a Rich-formatted traceback with locals."""

    formatted = format_exception_with_locals(exc)
    logger.error("%s\n%s", message, formatted)