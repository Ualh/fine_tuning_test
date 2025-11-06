"""AWQ conversion helpers shared by Typer and batch entry points.

Stage: post-processing — launches the AWQ conversion runner and writes logs under logs/<run>/convert-awq.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import typer

from ..core.config import ConfigLoader, PipelineConfig
from ..core.io_utils import ensure_dir, write_text
from ..core.logger import configure_logging, finalize_logger, tee_std_streams
from ..core.run_manager import RunManager
from ..core.run_naming import RunNameResult, build_run_name
from ..core.ssl import disable_ssl_verification


def run_convert_awq(
    config_path: Path,
    *,
    merged_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
    run_name: Optional[str] = None,
    run_index: Optional[int] = None,
) -> None:
    """Execute the AWQ conversion stage using the configured environment.

    Args:
        config_path: Path to the YAML configuration file.
        merged_dir: Optional override for the merged model directory.
        output_dir: Optional override for the AWQ output directory.
        force: Whether to overwrite existing output directories.
        run_name: Explicit run slug to reuse.
        run_index: Explicit run index when reserving a new slug.

    Raises:
        typer.BadParameter: Raised when the run context cannot be resolved.
        typer.Exit: Propagated when the underlying AWQ runner fails.
    """

    disable_ssl_verification()
    cfg = _load_config(config_path)

    merged_hint = merged_dir if merged_dir is not None else None
    output_hint = output_dir if output_dir is not None else None
    run_name_override = run_name or _infer_run_name_from_outputs(cfg, [merged_hint, output_hint])
    if run_name_override is None:
        run_name_override = _read_latest_run_name(cfg)
    if run_name_override is None:
        raise typer.BadParameter(
            "Unable to determine run context. Provide --run-name, --merged-dir or execute finetune-sft first."
        )

    run_info = _compute_run_info(cfg, run_name=run_name_override, run_index=run_index)
    _require_existing_outputs(cfg, run_info.name)
    outputs_root, _ = _ensure_run_dirs(cfg, run_info)

    run_manager = RunManager(cfg, run_info=run_info)
    _, run_dir = run_manager.create_run_dir("convert-awq")
    logger = configure_logging(
        name="convert-awq",
        log_dir=run_dir,
        console_level=_level_to_int(cfg.logging.console_level),
        file_level=_level_to_int(cfg.logging.file_level),
    )

    logger.info("Starting AWQ conversion for run %s", run_info.name)
    logger.debug("Runtime resolved outputs dir: %s", outputs_root)

    merged_resolved = Path(merged_dir or (outputs_root / "merged"))
    out_resolved = Path(output_dir or (outputs_root / "merged_awq"))

    logger.info("Merged directory: %s", merged_resolved)
    logger.info("Target AWQ output directory: %s", out_resolved)

    awq_cfg = getattr(cfg, "awq_conversion", None)
    enabled = True
    gpu_enabled = False
    if awq_cfg:
        enabled = bool(getattr(awq_cfg, "enabled", True))
        gpu_enabled = bool(getattr(awq_cfg, "gpu_enabled", False))

    if not enabled and not force:
        logger.info("AWQ conversion disabled in config and --force not provided; skipping conversion.")
        finalize_logger(logger)
        return

    console_log = run_dir / "console.log"
    try:
        with tee_std_streams(console_log):
            start_time = time.perf_counter()
            docker_exe = shutil.which("docker")
            if docker_exe:
                logger.info("Docker binary detected: %s", docker_exe)
            else:
                logger.info("Docker binary not found on PATH; will attempt local fallback")
            if docker_exe:
                cfg_rel = _rel_to_project(Path(config_path), cfg.project_root, label="config")
                merged_rel = _rel_to_project(merged_resolved, cfg.project_root, label="Merged directory")
                out_rel = _rel_to_project(out_resolved, cfg.project_root, label="Output directory")

                runner_cmd = (
                    "python3 -m src.training.awq_runner "
                    f"--config '/workspace/{cfg_rel.as_posix()}' "
                    f"--merged '/workspace/{merged_rel.as_posix()}' --out '/workspace/{out_rel.as_posix()}'"
                )
                if force:
                    runner_cmd += " --force"
                if gpu_enabled:
                    runner_cmd += " --gpu"

                docker_cmd = [
                    "docker",
                    "compose",
                    "run",
                    "--rm",
                    "--no-deps",
                    "-T",
                    "awq-runner",
                    "bash",
                    "-lc",
                    runner_cmd,
                ]
                logger.info("Invoking awq-runner sidecar: %s", " ".join(docker_cmd))
                proc = subprocess.run(docker_cmd)
                if proc.returncode != 0:
                    logger.error("awq-runner sidecar failed (rc=%d)", proc.returncode)
                    raise typer.Exit(code=proc.returncode)
                elapsed = time.perf_counter() - start_time
                logger.info("awq-runner sidecar finished successfully in %.1fs", elapsed)
            else:
                logger.info("Attempting local AWQ runner invocation (Docker unavailable).")
                from src.training.awq_runner import main as awq_main  # imported lazily to defer heavy deps

                args = [
                    "--config",
                    str(config_path),
                    "--merged",
                    str(merged_resolved),
                    "--out",
                    str(out_resolved),
                ]
                if force:
                    args.append("--force")
                if gpu_enabled:
                    args.append("--gpu")
                logger.info("Calling inline awq_runner with args: %s", " ".join(args))
                rc = awq_main(args)
                if rc != 0:
                    logger.error("Local awq runner failed (rc=%d)", rc)
                    raise typer.Exit(code=rc)
                elapsed = time.perf_counter() - start_time
                logger.info("Local awq runner finished successfully in %.1fs", elapsed)
    finally:
        finalize_logger(logger)


def _load_config(config_path: Path) -> PipelineConfig:
    loader = ConfigLoader(config_path)
    return loader.load()


def _level_to_int(level: str) -> int:
    value = getattr(logging, level.upper(), None)
    if value is None:
        raise typer.BadParameter(f"Unknown logging level: {level}")
    return value


def _compute_run_info(
    cfg: PipelineConfig,
    *,
    run_name: Optional[str] = None,
    run_index: Optional[int] = None,
) -> RunNameResult:
    env_overrides = dict(os.environ)
    if run_name:
        env_overrides["RUN_NAME"] = run_name
    if run_index is not None:
        env_overrides["FORCE_RUN_INDEX"] = str(run_index)

    run_info = build_run_name(cfg, env=env_overrides)
    os.environ.update(run_info.to_env())
    return run_info


def _default_outputs_root(cfg: PipelineConfig, run_info: RunNameResult) -> Tuple[Path, Path]:
    outputs_root = ensure_dir(cfg.paths.outputs_dir / run_info.name)
    logs_root = ensure_dir(cfg.paths.logs_dir / run_info.name)
    try:
        write_text(str(outputs_root), cfg.paths.outputs_dir / "latest.txt")
        write_text(str(logs_root), cfg.paths.run_metadata_file)
    except Exception:
        pass
    return outputs_root, logs_root


def _ensure_run_dirs(cfg: PipelineConfig, run_info: RunNameResult) -> Tuple[Path, Path]:
    return _default_outputs_root(cfg, run_info)


def _read_latest_run_name(cfg: PipelineConfig) -> Optional[str]:
    pointer = cfg.paths.run_metadata_file
    try:
        raw = pointer.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if not raw:
        return None
    return Path(raw).name


def _infer_run_name_from_outputs(
    cfg: PipelineConfig,
    artifacts: Sequence[Optional[Path]],
) -> Optional[str]:
    outputs_root = cfg.paths.outputs_dir
    try:
        outputs_root_resolved = outputs_root.resolve(strict=False)
    except RuntimeError:
        outputs_root_resolved = outputs_root

    for artifact in artifacts:
        if artifact is None:
            continue
        expanded = Path(artifact).expanduser()
        try:
            resolved = expanded.resolve(strict=False)
        except RuntimeError:
            resolved = expanded
        try:
            relative = resolved.relative_to(outputs_root_resolved)
        except ValueError:
            continue
        parts = relative.parts
        if parts:
            return parts[0]
    return None


def _require_existing_outputs(cfg: PipelineConfig, run_name: str) -> None:
    outputs_root = cfg.paths.outputs_dir / run_name
    if outputs_root.exists():
        return
    raise typer.BadParameter(
        "Cannot locate outputs for run '{run_name}'. Provide --run-name or pass explicit paths to reuse an existing run.".format(
            run_name=run_name
        )
    )


def _rel_to_project(path: Path, project_root: Path, *, label: str) -> Path:
    try:
        return path.relative_to(project_root)
    except ValueError as exc:
        raise typer.BadParameter(f"{label} must live within the project root ({project_root}).") from exc
