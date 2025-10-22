"""Run directory orchestration and metadata utilities."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

from .io_utils import ensure_dir, write_text


class RunManager:
    """Create deterministic log directories with auto-incremented versions."""

    def __init__(self, logs_root: Path, latest_pointer: Path) -> None:
        self.logs_root = ensure_dir(logs_root)
        self.latest_pointer = latest_pointer
        self.latest_pointer.parent.mkdir(parents=True, exist_ok=True)
        self._current_run_root: Optional[Path] = None
        self._current_version: Optional[int] = None

    def _next_version(self) -> int:
        existing = [p for p in self.logs_root.iterdir() if p.is_dir() and p.name.startswith("log_v")]
        versions = []
        for candidate in existing:
            name = candidate.name
            if not name.startswith("log_v"):
                continue
            numeric_token = name[len("log_v"):].split("_", 1)[0]
            try:
                versions.append(int(numeric_token))
            except ValueError:
                continue
        highest = max(versions, default=1)
        return highest + 1

    def create_run_dir(self, stage_name: str) -> Path:
        if self._current_run_root is None or not self._current_run_root.exists():
            version = self._next_version()
            timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            self._current_version = version
            self._current_run_root = ensure_dir(self.logs_root / f"log_v{version:02d}_{timestamp}")
        run_root = self._current_run_root
        stage_dir = ensure_dir(run_root / stage_name)
        # Update the pointer so users can quickly locate the most recent run.
        write_text(str(stage_dir.resolve()), self.latest_pointer)
        return stage_dir

    def touch_summary(self, stage_dir: Path, content: str) -> None:
        write_text(content, stage_dir / "SUMMARY.txt")

    def latest(self) -> Optional[Path]:
        if not self.latest_pointer.exists():
            return None
        value = self.latest_pointer.read_text(encoding="utf-8").strip()
        if not value:
            return None
        candidate = Path(value)
        return candidate if candidate.exists() else None

    def start_new_run(self) -> None:
        """Reset the current run root so the next stage call creates a new version."""

        self._current_run_root = None
        self._current_version = None
