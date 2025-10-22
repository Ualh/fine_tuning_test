"""Helpers for dealing with resume checkpoints across pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .errors import ResumeNotFoundError


CHECKPOINT_FILENAMES = {
    "pytorch_model.bin",
    "adapter_model.bin",
    "adapter_model.safetensors",
    "model.safetensors",
    "pytorch_model.safetensors",
}


class ResumeManager:
    """Resolve resume checkpoints while providing friendly diagnostics."""

    def __init__(self, logger) -> None:
        self.logger = logger

    def resolve(self, requested: Optional[str], default_path: Path) -> Optional[Path]:
        if requested in (None, "", "null"):
            return None
        normalized = "".join(
            "/" if ord(ch) < 32 else ch for ch in requested.replace("\\", "/")
        )
        path = Path(normalized)
        if requested == "latest":
            checkpoint = self._locate_checkpoint(default_path)
            if checkpoint is not None:
                self.logger.info("Resuming from latest at %s", checkpoint)
                return checkpoint
            raise ResumeNotFoundError(f"No checkpoints available under: {default_path}")
        candidates: list[Path] = []

        def add_candidate(candidate: Path) -> None:
            try:
                resolved = candidate.resolve()
            except FileNotFoundError:
                resolved = candidate
            if resolved not in candidates:
                candidates.append(resolved)

        if path.is_absolute():
            add_candidate(path)
        else:
            add_candidate(Path.cwd() / path)
            parts = path.parts
            parent_name = default_path.parent.name if default_path.parent.name else None
            first = parts[0] if parts else None
            if parent_name and first == parent_name:
                suffix = Path(*parts[1:]) if len(parts) > 1 else Path()
                add_candidate(default_path.parent / suffix)
            elif first == default_path.name:
                add_candidate(default_path.parent / path)
            else:
                add_candidate(default_path / path)

        for candidate in candidates:
            checkpoint = self._locate_checkpoint(candidate)
            if checkpoint is not None:
                self.logger.info("Resuming from %s", checkpoint)
                return checkpoint
            if candidate.exists():
                self.logger.warning("Resume candidate has no checkpoint artefacts: %s", candidate)

        checked = ", ".join(str(item) for item in candidates) if candidates else normalized
        self.logger.error("Resume directory not found for %s; checked: %s", requested, checked)
        last = candidates[-1] if candidates else Path(normalized)
        raise ResumeNotFoundError(
            "Resume directory not found: {}".format(last)
        )

    def _locate_checkpoint(self, path: Path) -> Optional[Path]:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            resolved = path
        if not resolved.exists():
            return None
        if resolved.is_file():
            return resolved
        if self._is_checkpoint_dir(resolved):
            return resolved

        checkpoints = sorted(
            (
                candidate
                for candidate in resolved.glob("checkpoint-*")
                if candidate.is_dir() and self._is_checkpoint_dir(candidate)
            ),
            key=self._checkpoint_step,
            reverse=True,
        )
        if checkpoints:
            return checkpoints[0]

        trainer_state = resolved / "trainer_state.json"
        if trainer_state.exists():
            return resolved

        return None

    @staticmethod
    def _is_checkpoint_dir(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        return any((path / name).exists() for name in CHECKPOINT_FILENAMES)

    @staticmethod
    def _checkpoint_step(path: Path) -> int:
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1
