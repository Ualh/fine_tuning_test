"""Command line entry-points for the fine-tuning pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from ..core.config import ConfigLoader, PipelineConfig
from ..core.logger import (
    configure_logging,
    finalize_logger,
    log_exception_with_locals,
    tee_std_streams,
)
from ..core.run_manager import RunManager
from ..core.resume import ResumeManager
from ..core.io_utils import ensure_dir
from ..core.ssl import disable_ssl_verification
from ..data.preprocess import DataPreprocessor
from ..training.sft_trainer import SFTTrainerRunner
from ..training.lora_merge import LoraMerger
from ..eval.evaluator import Evaluator
from ..serve.vllm_client import VLLMClient

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
    return cfg.paths.prepared_dir / f"{dataset_stub}_{cfg.preprocess.sample_size}".replace("-", "_")


def _default_output_dir(cfg: PipelineConfig) -> Path:
    dataset_stub = cfg.preprocess.dataset_name.split("/")[-1] if cfg.preprocess.dataset_name else "dataset"
    model_stub = cfg.train.base_model.split("/")[-1] if cfg.train.base_model else "model"
    return cfg.paths.outputs_dir / f"{dataset_stub}_{model_stub}"


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

    run_manager = RunManager(cfg.paths.logs_dir, cfg.paths.run_metadata_file)
    run_dir = run_manager.create_run_dir("preprocess")
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
    data_dir: Path = typer.Option(..., help="Directory containing train/val jsonl."),
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

    run_manager = RunManager(cfg.paths.logs_dir, cfg.paths.run_metadata_file)
    run_dir = run_manager.create_run_dir("train")
    logger = configure_logging(
        name="train",
        log_dir=run_dir,
        console_level=_level_to_int(cfg.logging.console_level),
        file_level=_level_to_int(cfg.logging.file_level),
    )

    output_resolved = Path(output_dir or _default_output_dir(cfg))
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
                    data_dir,
                    output_resolved,
                )
                summary = trainer.run(data_dir=Path(data_dir), output_dir=output_resolved, resume_path=resume_path)
                logger.info("Training summary: %s", summary)
            except Exception as exc:
                log_exception_with_locals(logger, "Finetune stage failed", exc)
                raise
    finally:
        finalize_logger(logger)


@app.command("export-merged")
def export_merged(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config YAML."),
    adapter_dir: Path = typer.Option(..., help="Directory with LoRA adapter."),
    output_dir: Optional[Path] = typer.Option(None, help="Where to place merged model."),
    base_model: Optional[str] = typer.Option(None, help="Base model name."),
) -> None:
    # Debug: bypass SSL verification to unblock HF downloads in restricted envs
    disable_ssl_verification()
    cfg = _load_config(config)
    if base_model:
        cfg.train.base_model = base_model

    run_manager = RunManager(cfg.paths.logs_dir, cfg.paths.run_metadata_file)
    run_dir = run_manager.create_run_dir("export")
    logger = configure_logging(
        name="export",
        log_dir=run_dir,
        console_level=_level_to_int(cfg.logging.console_level),
        file_level=_level_to_int(cfg.logging.file_level),
    )

    output_resolved = Path(output_dir or (cfg.paths.outputs_dir / "merged"))
    merger = LoraMerger(cfg.train.base_model, logger)
    console_log = run_dir / "console.log"
    try:
        with tee_std_streams(console_log):
            try:
                logger.info(
                    "Starting export stage | base_model=%s | adapter_dir=%s | output=%s",
                    cfg.train.base_model,
                    adapter_dir,
                    output_resolved,
                )
                summary = merger.run(adapter_dir=adapter_dir, output_dir=output_resolved)
                logger.info("Merge summary: %s", summary)
            except Exception as exc:
                log_exception_with_locals(logger, "Export stage failed", exc)
                raise
    finally:
        finalize_logger(logger)


@app.command("eval-sft")
def eval_sft(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config YAML."),
    model_dir: Path = typer.Option(..., help="Merged model directory."),
    output_dir: Optional[Path] = typer.Option(None, help="Where to store evaluation outputs."),
    cutoff_len: Optional[int] = typer.Option(None, help="Override sequence length for context."),
    val_path: Optional[Path] = typer.Option(None, help="Optional validation jsonl for perplexity."),
) -> None:
    # Debug: bypass SSL verification to unblock HF downloads in restricted envs
    disable_ssl_verification()
    cfg = _load_config(config)
    if cutoff_len is not None:
        cfg.eval.cutoff_len = cutoff_len

    run_manager = RunManager(cfg.paths.logs_dir, cfg.paths.run_metadata_file)
    run_dir = run_manager.create_run_dir("eval")
    logger = configure_logging(
        name="eval",
        log_dir=run_dir,
        console_level=_level_to_int(cfg.logging.console_level),
        file_level=_level_to_int(cfg.logging.file_level),
    )

    output_resolved = Path(output_dir or (cfg.paths.outputs_dir / "eval"))
    evaluator = Evaluator(cfg.eval, logger)
    console_log = run_dir / "console.log"
    try:
        with tee_std_streams(console_log):
            try:
                logger.info(
                    "Starting eval stage | model_dir=%s | output=%s",
                    model_dir,
                    output_resolved,
                )
                summary = evaluator.run(model_dir=Path(model_dir), output_dir=output_resolved, val_path=val_path)
                logger.info("Evaluation summary: %s", summary)
            except Exception as exc:
                log_exception_with_locals(logger, "Eval stage failed", exc)
                raise
    finally:
        finalize_logger(logger)


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
