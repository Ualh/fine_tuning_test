"""Logging utilities with Rich console output and file handlers."""

from __future__ import annotations

import io
import warnings
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
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    enable_rich: bool = True,
) -> logging.Logger:
    """Configure a logger with both console and rotating file outputs."""

    # Create or reconfigure the named pipeline logger. We attach a console
    # handler to this logger (so the interactive console shows only WARNING+),
    # and attach a file handler to the root logger so all third-party and
    # library logs propagate into the same run.log file.
    logger = logging.getLogger(name)
    # remove existing console-like handlers from the named logger so we can
    # reconfigure predictably
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)

    # Ensure log directory and file exist early
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    # File handler lives on the root logger so we capture messages from
    # third-party libraries that propagate up the hierarchy. This is the
    # canonical place where DEBUG+ messages are stored for the run.
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger = logging.getLogger()
    # Remove any existing file handlers at the root that look similar to avoid
    # duplicate file writes when reconfiguring.
    for h in list(root_logger.handlers):
        if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == str(log_file):
            root_logger.removeHandler(h)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)

    # Capture Python warnings (DeprecationWarning, etc.) so they land in the
    # run.log file via the root handler instead of printing directly.
    logging.captureWarnings(True)

    # Strip stray console handlers that popular libraries sometimes attach;
    # we keep their log level permissive so INFO/DEBUG still reach run.log.
    for library in ("transformers", "datasets", "accelerate", "peft", "trl"):
        lib_logger = logging.getLogger(library)
        for handler in list(lib_logger.handlers):
            if isinstance(handler, logging.StreamHandler):
                lib_logger.removeHandler(handler)
        lib_logger.setLevel(logging.DEBUG)

    # Console handler remains attached to the named logger; set to WARNING by
    # default so only operational messages hit the interactive console.
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

    # Keep library loggers at their defaults so their INFO/DEBUG entries still
    # propagate to the root file handler. We intentionally do not force levels
    # here because the file handler collects everything and the console is
    # controlled via the named logger's handler above.

    # Allow propagation so messages from this named logger reach the root
    # (file handler) while the console handler handles interactive output.
    logger.propagate = True
    logger.debug("Logger initialised; writing to %s", log_file)

    # Wrap warnings.warn so that calls to warnings.warn are also forwarded
    # into the logging system immediately (this helps tests that use
    # warnings.catch_warnings(record=True) while still expecting the
    # message to be recorded in run.log). We store the original on the
    # logger so it can be restored by `finalize_logger`.
    try:
        orig_warn = warnings.warn

        def _warn_and_log(message, category=None, stacklevel=1, source=None):
            # Log to the py.warnings logger (which propagates to root file handler)
            try:
                logging.getLogger("py.warnings").warning(str(message))
            except Exception:
                pass
            # Delegate to the original warn implementation so existing
            # showwarning/catch_warnings behaviours still operate.
            return orig_warn(message, category=category, stacklevel=stacklevel, source=source)

        warnings.warn = _warn_and_log  # type: ignore[assignment]
        setattr(logger, "_orig_warnings_warn", orig_warn)
    except Exception:
        # If anything goes wrong here we don't want configure_logging to fail.
        pass
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
        # Sanitize payload for the mirror (run.log) first.
        mirror_payload = self._transform(data) if self._transform else data

        # Suppress noisy metric/dict lines from appearing on the interactive
        # console while still writing them to the run.log. We detect common
        # patterns: explicit TRAIN/EVAL metric markers and bare Python dict
        # reprs that some libraries print (these appear between tqdm bars).
        try:
            suppress = False
            if isinstance(mirror_payload, str):
                lp = mirror_payload.strip()
                if (
                    "TRAIN_METRICS" in lp
                    or "FINAL_TRAIN_METRICS" in lp
                    or "FINAL_EVAL_METRICS" in lp
                    or "EVAL_METRICS" in lp
                ):
                    suppress = True
                else:
                    # Bare Python dict reprs typically start with a '{' followed
                    # by a quote or word char; catch those to avoid raw dicts
                    # interrupting progress bars.
                    if lp.startswith("{") and len(lp) > 2:
                        suppress = True
        except Exception:
            suppress = False

        # Write to the original console only when not suppressed.
        written = 0
        if not suppress:
            written = self._stream.write(data)

        # Always write the sanitized payload to the mirror file.
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

    # Restore any patched warnings.warn the logger installed.
    try:
        orig = getattr(logger, "_orig_warnings_warn", None)
        if orig is not None:
            import warnings as _warnings

            _warnings.warn = orig  # type: ignore[assignment]
            delattr(logger, "_orig_warnings_warn")
    except Exception:
        pass

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