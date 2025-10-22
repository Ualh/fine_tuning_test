"""Checkpoint management utilities for the fine-tuning stage."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.io_utils import ensure_dir


class Checkpointer:
    """Create and resolve directories used by the trainer."""

    def __init__(self, output_dir: Path, logger) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.logger = logger

    @property
    def adapter_dir(self) -> Path:
        return ensure_dir(self.output_dir / "adapter")

    @property
    def trainer_state_dir(self) -> Path:
        return ensure_dir(self.output_dir / "trainer_state")

    def resume_path(self, resume_from: Optional[str]) -> Optional[Path]:
        if not resume_from:
            return None
        path = Path(resume_from)
        if not path.is_absolute():
            path = (self.output_dir.parent / resume_from).resolve()
        if path.exists():
            self.logger.info("Resuming trainer from %s", path)
            return path
        self.logger.warning("Requested resume checkpoint not found: %s", path)
        return None
