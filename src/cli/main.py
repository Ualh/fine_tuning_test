"""Command line entry-points for the fine-tuning pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import typer

from ..core.config import ConfigLoader, PipelineConfig
from ..core.io_utils import ensure_dir, write_text
from ..core.logger import (
    configure_logging,
    finalize_logger,
    log_exception_with_locals,
    tee_std_streams,
)
from ..core.run_manager import RunManager
from ..core.run_naming import RunNameResult, build_run_name
from ..core.resume import ResumeManager
from ..core.ssl import disable_ssl_verification
from ..data.preprocess import DataPreprocessor
from ..training.sft_trainer import SFTTrainerRunner
from ..training.lora_merge import LoraMerger
 
from ..eval.evaluator import Evaluator
from ..serve.vllm_client import VLLMClient

POST_FINETUNE_TARGETS = {"export-merged", "convert-awq", "eval-sft", "serve-vllm"}

app = typer.Typer(add_completion=False, help="Fine-tuning pipeline for Qwen2.5 0.5B")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> PipelineConfig:
    loader = ConfigLoader(config_path)
    return loader.load()


def _level_to_int(level: str) -> int:
    value = getattr(logging, level.upper(), None)
    if value is None:
        raise typer.BadParameter(f"Unknown logging level: {level}")
    return value



def _default_preprocess_dir(cfg: PipelineConfig) -> Path:
    dataset_stub = cfg.preprocess.dataset_name.split("/")[-1] if cfg.preprocess.dataset_name else "dataset"
    size_tag = "full" if (cfg.preprocess.sample_size in (None, 0)) else str(cfg.preprocess.sample_size)
    return cfg.paths.prepared_dir / f"{dataset_stub}_{size_tag}".replace("-", "_")


def _default_output_dir(cfg: PipelineConfig, run_info: RunNameResult) -> Path:
    return cfg.paths.outputs_dir / run_info.name


def _ensure_run_dirs(cfg: PipelineConfig, run_info: RunNameResult) -> tuple[Path, Path]:
    outputs_root = ensure_dir(cfg.paths.outputs_dir / run_info.name)
    logs_root = ensure_dir(cfg.paths.logs_dir / run_info.name)
    try:
        write_text(str(outputs_root), cfg.paths.outputs_dir / "latest.txt")
        write_text(str(logs_root), cfg.paths.run_metadata_file)
    except Exception:
        pass
    return Path(outputs_root), Path(logs_root)


def _compute_run_info(
    cfg: PipelineConfig,
    *,
    run_name: Optional[str] = None,
    run_index: Optional[int] = None,
    update_env: bool = True,
) -> RunNameResult:
    env_overrides = dict(os.environ)
    if run_name:
        env_overrides["RUN_NAME"] = run_name
    if run_index is not None:
        env_overrides["FORCE_RUN_INDEX"] = str(run_index)

    run_info = build_run_name(cfg, env=env_overrides)
    if update_env:
        os.environ.update(run_info.to_env())
    return run_info


def _normalize_stage_name(stage: Optional[str]) -> str:
    return (stage or "").strip().lower()


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


def _infer_run_name_from_outputs(cfg: PipelineConfig, artifacts: Sequence[Optional[Path]]) -> Optional[str]:
    outputs_root = cfg.paths.outputs_dir
    try:
        outputs_root_resolved = outputs_root.resolve(strict=False)
    except RuntimeError:  # pragma: no cover - defensive
        outputs_root_resolved = outputs_root

    for artifact in artifacts:
        if artifact is None:
            continue
        expanded = Path(artifact).expanduser()
        try:
            resolved = expanded.resolve(strict=False)
        except RuntimeError:  # pragma: no cover - defensive
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


@app.command("run-name-preview")
def run_name_preview(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config YAML."),
    run_name: Optional[str] = typer.Option(None, help="Override run naming with an explicit slug."),
    run_index: Optional[int] = typer.Option(None, help="Force the run counter (runX)."),
    format: str = typer.Option("name", "--format", "-f", help="Output format: name, json, or env."),
) -> None:
    """Preview the resolved run name without reserving directories."""

    disable_ssl_verification()
    cfg = _load_config(config)
    run_info = _compute_run_info(
        cfg,
        run_name=run_name,
        run_index=run_index,
        update_env=False,
    )

    fmt = format.lower()
    if fmt == "name":
        typer.echo(run_info.name)
        return
    if fmt == "json":
        payload = {
            "run_name": run_info.name,
            "run_prefix": run_info.prefix,
            "run_number": run_info.index,
            "model_slug": run_info.model_slug,
            "dataset_slug": run_info.dataset_slug,
            "size_slug": run_info.size_slug,
            "legacy": run_info.legacy,
        }
        typer.echo(json.dumps(payload, indent=2))
        return
    if fmt == "env":
        for key, value in run_info.to_env().items():
            typer.echo(f"{key}={value}")
        return
    raise typer.BadParameter("Unsupported format. Use 'name', 'json', or 'env'.", param_hint="format")


@app.command("convert-awq")
def convert_awq(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config YAML."),
    merged_dir: Optional[Path] = typer.Option(None, help="Merged model directory (defaults to outputs/*/merged)."),
    output_dir: Optional[Path] = typer.Option(None, help="Where to store compressed model output."),
    force: bool = typer.Option(False, help="Overwrite existing output if present."),
    run_name: Optional[str] = typer.Option(None, help="Override run naming with an explicit slug."),
    run_index: Optional[int] = typer.Option(None, help="Force the run counter (runX)."),
) -> None:
    """Run AWQ conversion using the isolated `awq-runner` sidecar (preferred) or locally.

    The command will shell out to `docker compose run --rm --no-deps -T awq-runner ...`
    when Docker is available. If Docker is not available but the environment has
    the Python dependencies installed, it will call the in-repo runner
    implementation (`src.training.awq_runner`) directly.
    """
    disable_ssl_verification()
    cfg = _load_config(config)

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

    # Check config AWQ block (loader supports legacy `llm_compressor` key)
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
            # Prefer Docker sidecar if docker available
            if docker_exe:
                # Build container-invocation command using workspace-relative paths
                cfg_rel = _rel_to_project(Path(config), cfg.project_root, label="config")
                merged_rel = _rel_to_project(merged_resolved, cfg.project_root, label="Merged directory")
                out_rel = _rel_to_project(out_resolved, cfg.project_root, label="Output directory")

                runner_cmd = (
                    f"python3 /workspace/scripts/awq_runner.py --config '/workspace/{cfg_rel.as_posix()}' "
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
                # No Docker: try to run the runner locally
                logger.info("Attempting local AWQ runner invocation (Docker unavailable).")
                try:
                    from src.training.awq_runner import main as awq_main

                    args = [
                        f"--config",
                        str(config),
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
                except Exception as exc:
                    log_exception_with_locals(logger, "Local awq runner failed", exc)
                    raise
    finally:
        finalize_logger(logger)



_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(value: str, fallback: str, max_length: int = 24) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    if not slug:
        slug = fallback
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug or fallback


def _rel_to_project(path: Path, project_root: Path, *, label: str) -> Path:
    try:
        return path.relative_to(project_root)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise typer.BadParameter(
            f"{label} must live within the project root ({project_root})."
        ) from exc


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("preprocess-sft")
def preprocess_sft(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config YAML."),
    dataset_name: Optional[str] = typer.Option(None, help="Override dataset name."),
    sample_size: Optional[int] = typer.Option(None, help="Number of samples to keep."),
    filter_langs: Optional[str] = typer.Option(None, help="Comma separated list of languages."),
    test_size: Optional[float] = typer.Option(None, help="Validation split ratio."),
    cutoff_len: Optional[int] = typer.Option(None, help="Max sequence length."),
    seed: Optional[int] = typer.Option(None, help="Random seed."),
    max_workers: Optional[int] = typer.Option(None, help="Parallel workers for map."),
    pack_sequences: Optional[bool] = typer.Option(None, help="Enable sequence packing."),
    output: Optional[Path] = typer.Option(None, help="Output directory for prepared splits."),
    resume_from: Optional[Path] = typer.Option(None, help="Reuse an existing prepared directory."),
    run_name: Optional[str] = typer.Option(None, help="Override run naming with an explicit slug."),
    run_index: Optional[int] = typer.Option(None, help="Force the run counter (runX)."),
) -> None:
    # Debug: bypass SSL verification to unblock HF downloads in restricted envs
    disable_ssl_verification()
    cfg = _load_config(config)
    if dataset_name:
        cfg.preprocess.dataset_name = dataset_name
    if sample_size is not None:
        cfg.preprocess.sample_size = sample_size
    if filter_langs:
        cfg.preprocess.filter_langs = [item.strip() for item in filter_langs.split(",") if item.strip()]
    if test_size is not None:
        cfg.preprocess.test_size = test_size
    if cutoff_len is not None:
        cfg.preprocess.cutoff_len = cutoff_len
    if seed is not None:
        cfg.preprocess.seed = seed
    if max_workers is not None:
        cfg.preprocess.max_workers = max_workers
    if pack_sequences is not None:
        cfg.preprocess.pack_sequences = pack_sequences

    run_info = _compute_run_info(cfg, run_name=run_name, run_index=run_index)
    _ensure_run_dirs(cfg, run_info)

    run_manager = RunManager(cfg, run_info=run_info)
    _, run_dir = run_manager.create_run_dir("preprocess")
    logger = configure_logging(
        name="preprocess",
        log_dir=run_dir,
        console_level=_level_to_int(cfg.logging.console_level),
        file_level=_level_to_int(cfg.logging.file_level),
    )

    output_dir = Path(output or _default_preprocess_dir(cfg))
    resume_mgr = ResumeManager(logger)
    resume_dir = resume_mgr.resolve(str(resume_from) if resume_from else None, output_dir)

    preprocessor = DataPreprocessor(cfg.preprocess, cfg.train, cfg.paths, logger)
    console_log = run_dir / "console.log"
    try:
        with tee_std_streams(console_log):
            try:
                logger.info(
                    "Starting preprocess stage | dataset=%s | sample_size=%s | output=%s",
                    cfg.preprocess.dataset_name,
                    cfg.preprocess.sample_size,
                    output_dir,
                )
                summary = preprocessor.run(output_dir=output_dir, resume_dir=resume_dir)
                logger.info("Preprocessing completed: %s", summary)
            except Exception as exc:
                log_exception_with_locals(logger, "Preprocess stage failed", exc)
                raise
    finally:
        finalize_logger(logger)


@app.command("finetune-sft")
def finetune_sft(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config YAML."),
    data_dir: Optional[Path] = typer.Option(None, help="Directory containing train/val jsonl."),
    output_dir: Optional[Path] = typer.Option(None, help="Directory to store adapter checkpoints."),
    base_model: Optional[str] = typer.Option(None, help="Base model to load."),
    cutoff_len: Optional[int] = typer.Option(None, help="Max sequence length."),
    batch_size: Optional[int] = typer.Option(None),
    gradient_accumulation: Optional[int] = typer.Option(None),
    epochs: Optional[int] = typer.Option(None),
    learning_rate: Optional[float] = typer.Option(None),
    min_learning_rate: Optional[float] = typer.Option(None),
    weight_decay: Optional[float] = typer.Option(None),
    warmup_ratio: Optional[float] = typer.Option(None),
    lr_scheduler: Optional[str] = typer.Option(None),
    lora_r: Optional[int] = typer.Option(None),
    lora_alpha: Optional[int] = typer.Option(None),
    lora_dropout: Optional[float] = typer.Option(None),
    lora_targets: Optional[str] = typer.Option(None),
    gradient_checkpointing: Optional[bool] = typer.Option(None),
    bf16: Optional[bool] = typer.Option(None),
    fp16: Optional[bool] = typer.Option(None),
    logging_steps: Optional[int] = typer.Option(None),
    eval_steps: Optional[int] = typer.Option(None),
    resume_from: Optional[Path] = typer.Option(None, help="Resume trainer state."),
    run_name: Optional[str] = typer.Option(None, help="Override run naming with an explicit slug."),
    run_index: Optional[int] = typer.Option(None, help="Force the run counter (runX)."),
) -> None:
    # Debug: bypass SSL verification to unblock HF downloads in restricted envs
    disable_ssl_verification()
    cfg = _load_config(config)
    if base_model:
        cfg.train.base_model = base_model
    if cutoff_len is not None:
        cfg.train.cutoff_len = cutoff_len
    if batch_size is not None:
        cfg.train.batch_size = batch_size
    if gradient_accumulation is not None:
        cfg.train.gradient_accumulation = gradient_accumulation
    if epochs is not None:
        cfg.train.epochs = epochs
    if learning_rate is not None:
        cfg.train.learning_rate = learning_rate
    if min_learning_rate is not None:
        cfg.train.min_learning_rate = min_learning_rate
    if weight_decay is not None:
        cfg.train.weight_decay = weight_decay
    if warmup_ratio is not None:
        cfg.train.warmup_ratio = warmup_ratio
    if lr_scheduler is not None:
        cfg.train.lr_scheduler = lr_scheduler
    if lora_r is not None:
        cfg.train.lora_r = lora_r
    if lora_alpha is not None:
        cfg.train.lora_alpha = lora_alpha
    if lora_dropout is not None:
        cfg.train.lora_dropout = lora_dropout
    if lora_targets:
        cfg.train.lora_target_modules = [item.strip() for item in lora_targets.split(",") if item.strip()]
    if gradient_checkpointing is not None:
        cfg.train.gradient_checkpointing = gradient_checkpointing
    if bf16 is not None:
        cfg.train.bf16 = bf16
    if fp16 is not None:
        cfg.train.fp16 = fp16
    if logging_steps is not None:
        cfg.train.logging_steps = logging_steps
    if eval_steps is not None:
        cfg.train.eval_steps = eval_steps

    run_info = _compute_run_info(cfg, run_name=run_name, run_index=run_index)
    outputs_root, _ = _ensure_run_dirs(cfg, run_info)

    run_manager = RunManager(cfg, run_info=run_info)
    _, run_dir = run_manager.create_run_dir("train")
    logger = configure_logging(
        name="train",
        log_dir=run_dir,
        console_level=_level_to_int(cfg.logging.console_level),
        file_level=_level_to_int(cfg.logging.file_level),
    )

    output_resolved = Path(output_dir or outputs_root)
    data_resolved = Path(data_dir or _default_preprocess_dir(cfg))
    resume_mgr = ResumeManager(logger)
    resume_path = resume_mgr.resolve(str(resume_from) if resume_from else None, output_resolved)

    trainer = SFTTrainerRunner(cfg.train, cfg.preprocess, logger, run_dir)
    console_log = run_dir / "console.log"
    try:
        with tee_std_streams(console_log):
            try:
                logger.info(
                    "Starting finetune stage | base_model=%s | data_dir=%s | output=%s",
                    cfg.train.base_model,
                    data_resolved,
                    output_resolved,
                )
                summary = trainer.run(data_dir=data_resolved, output_dir=output_resolved, resume_path=resume_path)
                logger.info("Training summary: %s", summary)
            except Exception as exc:
                log_exception_with_locals(logger, "Finetune stage failed", exc)
                raise
    finally:
        finalize_logger(logger)


@app.command("export-merged")
def export_merged(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config YAML."),
    adapter_dir: Optional[Path] = typer.Option(None, help="Directory with LoRA adapter."),
    output_dir: Optional[Path] = typer.Option(None, help="Where to place merged model."),
    base_model: Optional[str] = typer.Option(None, help="Base model name."),
    run_name: Optional[str] = typer.Option(None, help="Override run naming with an explicit slug."),
    run_index: Optional[int] = typer.Option(None, help="Force the run counter (runX)."),
    use_latest_adapter: bool = typer.Option(False, help="If adapter is missing, automatically pick the most recent adapter from outputs/*/adapter."),
) -> None:
    # Debug: bypass SSL verification to unblock HF downloads in restricted envs
    disable_ssl_verification()
    cfg = _load_config(config)
    if base_model:
        cfg.train.base_model = base_model

    adapter_hint = adapter_dir if adapter_dir is not None else None
    output_hint = output_dir if output_dir is not None else None

    run_name_override = run_name or _infer_run_name_from_outputs(cfg, [adapter_hint, output_hint])
    if run_name_override is None:
        run_name_override = _read_latest_run_name(cfg)
    if run_name_override is None:
        raise typer.BadParameter(
            "Unable to determine run context. Provide --run-name, --adapter-dir, or execute finetune-sft first."
        )

    run_info = _compute_run_info(cfg, run_name=run_name_override, run_index=run_index)
    _require_existing_outputs(cfg, run_info.name)
    outputs_root, _ = _ensure_run_dirs(cfg, run_info)

    run_manager = RunManager(cfg, run_info=run_info)
    _, run_dir = run_manager.create_run_dir("export")
    logger = configure_logging(
        name="export",
        log_dir=run_dir,
        console_level=_level_to_int(cfg.logging.console_level),
        file_level=_level_to_int(cfg.logging.file_level),
    )

    adapter_resolved = Path(adapter_dir or (outputs_root / "adapter"))
    output_resolved = Path(output_dir or (outputs_root / "merged"))
    merger = LoraMerger(cfg.train.base_model, logger)
    console_log = run_dir / "console.log"
    try:
        with tee_std_streams(console_log):
            try:
                logger.info(
                    "Starting export stage | base_model=%s | adapter_dir=%s | output=%s",
                    cfg.train.base_model,
                    adapter_resolved,
                    output_resolved,
                )
                # If the adapter doesn't exist at the expected location, offer
                # a helpful fallback: optionally pick the most-recent adapter
                # found under `outputs/*/adapter` when the user passed
                # --use-latest-adapter. Otherwise raise a clear error listing
                # available candidates.
                if not adapter_resolved.exists():
                    candidates = list(cfg.paths.outputs_dir.glob("*/adapter"))
                    if not candidates:
                        raise FileNotFoundError(
                            f"Adapter directory not found: {adapter_resolved}\nNo adapters found under {cfg.paths.outputs_dir}.\nRun finetune-sft first or pass --adapter-dir to point to an existing adapter."
                        )
                    if not use_latest_adapter:
                        # List candidates to help the user pick the right one
                        pretty = "\n".join([str(p.parent) for p in candidates])
                        raise FileNotFoundError(
                            f"Adapter directory not found: {adapter_resolved}\nFound the following adapter candidates:\n{pretty}\n\nTo automatically use the most-recent adapter, re-run with --use-latest-adapter.\nOr pass --adapter-dir to explicitly select one."
                        )
                    # use the most recently modified adapter
                    candidates_sorted = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
                    chosen = candidates_sorted[0]
                    logger.info("--use-latest-adapter: selecting adapter from %s", chosen.parent)
                    adapter_resolved = chosen
                    # default output_resolved to the chosen run's merged dir if not explicitly passed
                    if output_dir is None:
                        output_resolved = chosen.parent / "merged"

                summary = merger.run(adapter_dir=adapter_resolved, output_dir=output_resolved)
                logger.info("Merge summary: %s", summary)
            except Exception as exc:
                log_exception_with_locals(logger, "Export stage failed", exc)
                raise
    finally:
        finalize_logger(logger)





@app.command("eval-sft")
def eval_sft(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config YAML."),
    model_dir: Optional[Path] = typer.Option(None, help="Merged model directory."),
    output_dir: Optional[Path] = typer.Option(None, help="Where to store evaluation outputs."),
    cutoff_len: Optional[int] = typer.Option(None, help="Override sequence length for context."),
    val_path: Optional[Path] = typer.Option(None, help="Optional validation jsonl for perplexity."),
    run_name: Optional[str] = typer.Option(None, help="Override run naming with an explicit slug."),
    run_index: Optional[int] = typer.Option(None, help="Force the run counter (runX)."),
) -> None:
    # Debug: bypass SSL verification to unblock HF downloads in restricted envs
    disable_ssl_verification()
    cfg = _load_config(config)
    if cutoff_len is not None:
        cfg.eval.cutoff_len = cutoff_len

    model_hint = model_dir if model_dir is not None else None
    output_hint = output_dir if output_dir is not None else None
    run_name_override = run_name or _infer_run_name_from_outputs(cfg, [model_hint, output_hint])
    if run_name_override is None:
        run_name_override = _read_latest_run_name(cfg)
    if run_name_override is None:
        raise typer.BadParameter(
            "Unable to determine run context. Provide --run-name, --model-dir or execute finetune-sft first."
        )

    run_info = _compute_run_info(cfg, run_name=run_name_override, run_index=run_index)
    _require_existing_outputs(cfg, run_info.name)
    outputs_root, _ = _ensure_run_dirs(cfg, run_info)

    run_manager = RunManager(cfg, run_info=run_info)
    _, run_dir = run_manager.create_run_dir("eval")
    logger = configure_logging(
        name="eval",
        log_dir=run_dir,
        console_level=_level_to_int(cfg.logging.console_level),
        file_level=_level_to_int(cfg.logging.file_level),
    )

    model_resolved = Path(model_dir or (outputs_root / "merged"))
    output_resolved = Path(output_dir or (outputs_root / "eval"))
    evaluator = Evaluator(cfg.eval, logger)
    console_log = run_dir / "console.log"
    try:
        with tee_std_streams(console_log):
            try:
                logger.info(
                    "Starting eval stage | model_dir=%s | output=%s",
                    model_resolved,
                    output_resolved,
                )
                summary = evaluator.run(model_dir=model_resolved, output_dir=output_resolved, val_path=val_path)
                logger.info("Evaluation summary: %s", summary)
            except Exception as exc:
                log_exception_with_locals(logger, "Eval stage failed", exc)
                raise
    finally:
        finalize_logger(logger)





def _build_runtime_metadata(cfg: PipelineConfig, *, stage: Optional[str] = None) -> Dict[str, str]:
    if not cfg.preprocess.dataset_name:
        raise typer.BadParameter("preprocess.dataset_name must be set in config.yaml.")
    sample_size = cfg.preprocess.sample_size
    # None means 'full' dataset
    if sample_size is not None and int(sample_size) <= 0:
        raise typer.BadParameter("preprocess.sample_size must be a positive integer or 'full' in config.yaml.")
    if not cfg.train.base_model:
        raise typer.BadParameter("train.base_model must be set in config.yaml.")
    if not cfg.serve.served_model_name:
        raise typer.BadParameter("serve.served_model_name must be set in config.yaml.")

    project_root = cfg.project_root
    stage_normalized = _normalize_stage_name(stage)

    explicit_env_run = os.environ.get("RUN_NAME") or os.environ.get("FORCE_RUN_NAME")
    run_name_hint: Optional[str] = None
    require_existing = stage_normalized in POST_FINETUNE_TARGETS

    if explicit_env_run:
        run_name_hint = explicit_env_run
    elif require_existing:
        run_name_hint = _read_latest_run_name(cfg)
        if run_name_hint is None:
            raise typer.BadParameter(
                "No existing run detected. Provide --run-name/--run-index or run finetune-sft before executing '{stage}'.".format(
                    stage=stage_normalized or "this stage",
                )
            )

    run_info = _compute_run_info(cfg, run_name=run_name_hint)

    if require_existing:
        _require_existing_outputs(cfg, run_info.name)

    outputs_root, logs_root = _ensure_run_dirs(cfg, run_info)

    # Use a stable compose project name so docker compose containers/images are
    # not recreated per run. Default to "SFT" but allow callers to override via
    # HOST_COMPOSE_PROJECT, COMPOSE_PROJECT, or COMPOSE_PROJECT_NAME.
    compose_env = os.environ.get("COMPOSE_PROJECT") or os.environ.get("COMPOSE_PROJECT_NAME")
    host_compose = os.environ.get("HOST_COMPOSE_PROJECT")
    compose_project = (host_compose or compose_env or "sft").strip()
    if not compose_project:
        compose_project = "sft"
    compose_project = re.sub(r"[^A-Za-z0-9_-]+", "-", compose_project).lower()
    compose_project = compose_project[:62].rstrip("-") or "sft"
    if not compose_project[0].isalnum():
        compose_project = f"sft-{compose_project}"
        compose_project = compose_project[:62].rstrip("-") or "sft"

    preprocess_dir = _default_preprocess_dir(cfg)
    adapter_dir = outputs_root / "adapter"
    merged_dir = outputs_root / "merged"
    eval_dir = outputs_root / "eval"

    preprocess_rel = _rel_to_project(preprocess_dir, project_root, label="Preprocess output directory")
    output_rel = _rel_to_project(outputs_root, project_root, label="Training output directory")
    adapter_rel = _rel_to_project(adapter_dir, project_root, label="Adapter directory")
    merged_rel = _rel_to_project(merged_dir, project_root, label="Merged directory")
    eval_rel = _rel_to_project(eval_dir, project_root, label="Evaluation directory")
    logs_rel = _rel_to_project(logs_root, project_root, label="Logs directory")

    # Determine the served model relative path. If the config value is falsy
    # or explicitly set to the string "none" (case-insensitive), use the
    # merged model directory instead. Otherwise normalize backslashes to
    # forward slashes and strip any leading/trailing slashes.
    awq_suffix = "awq"
    if cfg.awq_conversion and cfg.awq_conversion.output_suffix:
        awq_suffix = str(cfg.awq_conversion.output_suffix)
    awq_dir = merged_dir.with_name(f"{merged_dir.name}_{awq_suffix}")

    raw_served = cfg.serve.served_model_relpath
    prefer_awq = bool(getattr(cfg.serve, "prefer_awq", True))

    served_dir = merged_dir
    served_source = "merged"

    if raw_served and not (isinstance(raw_served, str) and raw_served.strip().lower() == "none"):
        override_str = str(raw_served).replace("\\", "/").strip("/")
        override_path = Path(override_str)
        if override_path.is_absolute():
            served_dir = override_path
        else:
            parts = [part for part in override_path.parts if part and part not in {".", ".."}]
            if parts and parts[0].lower() == "outputs":
                parts = parts[1:]
            served_dir = cfg.paths.outputs_dir.joinpath(*parts) if parts else cfg.paths.outputs_dir
        served_source = "override"
    else:
        if prefer_awq and awq_dir.exists():
            served_dir = awq_dir
            served_source = "awq"
        else:
            served_dir = merged_dir
            served_source = "merged"

    try:
        served_rel_candidate = served_dir.relative_to(cfg.paths.outputs_dir).as_posix()
        served_rel = served_rel_candidate.strip("/") or ""
    except ValueError:
        try:
            served_rel = _rel_to_project(served_dir, project_root, label="Served model directory").as_posix()
        except typer.BadParameter:
            served_rel = served_dir.as_posix().replace("\\", "/")

    served_name = getattr(cfg.serve, "model_name", None) or cfg.serve.served_model_name

    runtime: Dict[str, str] = {
        "PROJECT_ROOT": str(project_root),
        "COMPOSE_PROJECT": compose_project,
        "DATASET_NAME": cfg.preprocess.dataset_name,
        "DATASET_SAMPLE_SIZE": "full" if sample_size is None else str(sample_size),
        "BASE_MODEL_NAME": cfg.train.base_model,
        "PREPROCESS_DIR": preprocess_rel.as_posix(),
        "PREPROCESS_DIR_CONTAINER": f"/app/{preprocess_rel.as_posix()}",
        "OUTPUT_DIR": output_rel.as_posix(),
        "OUTPUT_DIR_CONTAINER": f"/app/{output_rel.as_posix()}",
        "ADAPTER_DIR": adapter_rel.as_posix(),
        "MERGED_DIR": merged_rel.as_posix(),
        "MERGED_DIR_CONTAINER": f"/app/{merged_rel.as_posix()}",
        "EVAL_DIR": eval_rel.as_posix(),
        "EVAL_DIR_CONTAINER": f"/app/{eval_rel.as_posix()}",
    "SERVED_MODEL_RELPATH": served_rel,
    "SERVED_MODEL_PATH": f"/models/{served_rel}" if served_rel else "/models",
    "SERVED_MODEL_NAME": served_name,
        "SERVED_MODEL_MAX_LEN": str(cfg.serve.max_model_len),
        "SERVE_PORT": str(cfg.serve.port),
        "SERVE_HOST": cfg.serve.host,
        "DEBUG_PIPELINE": "1" if cfg.logging.debug_pipeline else "0",
        "TENSORBOARD_LOGDIR": logs_rel.as_posix(),
        "TENSORBOARD_LOGDIR_CONTAINER": f"/app/{logs_rel.as_posix()}",
    # Indicate whether training reports to tensorboard (so wrapper can
    # automatically start the tensorboard service when requested).
    "TENSORBOARD_ENABLED": "1" if ("tensorboard" in (cfg.train.report_to or [])) else "0",
        # AWQ / llm-compressor runtime hints for wrapper scripts
        "AWQ_ENABLED": "1" if (cfg.awq_conversion and cfg.awq_conversion.enabled) else "0",
        "AWQ_GPU_ENABLED": "1" if (cfg.awq_conversion and cfg.awq_conversion.gpu_enabled) else "0",
        "AWQ_OUTPUT_SUFFIX": (cfg.awq_conversion.output_suffix if (cfg.awq_conversion and cfg.awq_conversion.output_suffix) else "awq"),
        "RUN_OUTPUTS_DIR": output_rel.as_posix(),
        "RUN_OUTPUTS_DIR_CONTAINER": f"/app/{output_rel.as_posix()}",
        "RUN_LOGS_DIR": logs_rel.as_posix(),
        "RUN_LOGS_DIR_CONTAINER": f"/app/{logs_rel.as_posix()}",
        "RUN_DIR_NAME": run_info.name,
        "POST_FINETUNE_TARGETS": " ".join(cfg.orchestration.post_finetune or []),
    }
    # Host path for the served model directory so docker-compose can mount it
    # directly into the vLLM container at the expected `/models/<served_rel>`.
    # Use forward-slash notation to keep compose interpolation portable.
    try:
        served_host_path = served_dir.resolve()
    except Exception:
        served_host_path = served_dir
    runtime["SERVED_MODEL_PATH_HOST"] = served_host_path.as_posix()
    runtime["SERVED_MODEL_SOURCE"] = served_source
    runtime.update(run_info.to_env())
    return runtime


@app.command("print-runtime")
def print_runtime(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config YAML."),
    format: str = typer.Option("env", "--format", "-f", help="Output format: env or json."),
    stage: Optional[str] = typer.Option(None, "--stage", help="Pipeline stage requesting metadata."),
) -> None:
    cfg = _load_config(config)
    runtime = _build_runtime_metadata(cfg, stage=stage)

    if format.lower() == "env":
        for key, value in runtime.items():
            typer.echo(f"{key}={value}")
        return

    if format.lower() == "json":
        typer.echo(json.dumps(runtime, indent=2))
        return

    raise typer.BadParameter("Unsupported format. Use 'env' or 'json'.", param_hint="format")


@app.command("smoke-test")
def smoke_test(
    endpoint: str = typer.Option("http://localhost:8080", help="vLLM endpoint base URL."),
    model: str = typer.Option("Qwen2.5-0.5B-SFT", help="Served model name."),
    prompt: str = typer.Option("Say hello in French.", help="Prompt to query."),
) -> None:
    client = VLLMClient(endpoint=endpoint, model=model)
    if not client.healthcheck():
        raise typer.Exit(code=2)
    response = client.generate(prompt)
    typer.echo(json.dumps(response, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
