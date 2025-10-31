"""Run directory orchestration and metadata utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .config import PipelineConfig
from .io_utils import ensure_dir, write_text
from .run_naming import RunNameResult, build_run_name


class RunManager:
    """Create per-run log directories named after the canonical run slug."""

    def __init__(
        self,
        cfg: PipelineConfig,
        *,
        run_info: Optional[RunNameResult] = None,
    ) -> None:
        self.cfg = cfg
        self.logs_root = ensure_dir(cfg.paths.logs_dir)
        self.latest_pointer = cfg.paths.run_metadata_file
        self.latest_pointer.parent.mkdir(parents=True, exist_ok=True)

        self._run_info: Optional[RunNameResult] = run_info
        self._run_root: Optional[Path] = None
        self._explicit = run_info is not None

    def _resolve_run_info(self) -> RunNameResult:
        if self._run_info is None:
            self._run_info = build_run_name(
                self.cfg,
                outputs_root=self.cfg.paths.outputs_dir,
                logs_root=self.cfg.paths.logs_dir,
            )
        return self._run_info

    def create_run_dir(self, stage_name: str) -> Tuple[RunNameResult, Path]:
        run_info = self._resolve_run_info()
        self._run_root = ensure_dir(self.logs_root / run_info.name)
        stage_dir = ensure_dir(self._run_root / stage_name)

        # Track the most recent run for convenience tools (see logs/latest.txt).
        write_text(str(self._run_root.resolve()), self.latest_pointer)

        return run_info, stage_dir

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
        """Prepare the manager to allocate a fresh run slug on the next stage."""

        if self._explicit:
            return
        self._run_info = None
        self._run_root = None
