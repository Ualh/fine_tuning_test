"""Utility helpers for interacting with the filesystem in a safe manner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

from .errors import PipelineConfigError


def ensure_dir(path: Path) -> Path:
    """Ensure that *path* exists on disk and return the absolute variant."""

    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def atomic_write_json(payload: Dict[str, Any], destination: Path) -> None:
    """Atomically write *payload* to *destination* as JSON.

    Supports serializing pathlib.Path objects by converting them to str.
    """

    def _json_default(obj: Any):  # minimal, explicit fallback
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    tmp_path.replace(destination)


def write_text(content: str, destination: Path) -> None:
    """Write *content* to *destination* using UTF-8 encoding."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON data from *path* and return the parsed dictionary."""

    if not path.exists():  # pragma: no cover - guard
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expand_path(path: str | Path) -> Path:
    """Expand environment variables and user markers in *path*."""

    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    return Path(expanded).resolve()


def parse_comma_separated(raw: str | Iterable[str]) -> Iterable[str]:
    """Split a comma separated string into distinct trimmed entries."""

    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        parts = list(raw)
    if not parts:
        raise PipelineConfigError("At least one entry is required for the provided list")
    return parts
