"""Regression tests for failure logging and traceback capture."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from src.cli.main import app
from src.training.sft_trainer import SFTTrainerRunner


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


def _write_failure_config(tmp_path: Path) -> Path:
    config = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "prepared_dir": str(tmp_path / "prepared"),
            "outputs_dir": str(tmp_path / "outputs"),
            "logs_dir": str(tmp_path / "logs"),
            "huggingface_cache": str(tmp_path / "hf_cache"),
            "models_mount": str(tmp_path / "models"),
            "run_metadata_file": str(tmp_path / "logs" / "latest.txt"),
        },
        "preprocess": {
            "dataset_name": "acme/dummy",
            "sample_size": 2,
            "filter_langs": ["en"],
            "test_size": 0.5,
            "cutoff_len": 64,
            "seed": 123,
            "save_splits": True,
            "max_workers": 1,
            "pack_sequences": False,
        },
        "train": {
            "base_model": "acme/dummy",
            "cutoff_len": 64,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "epochs": 1,
            "learning_rate": 2e-5,
            "min_learning_rate": 5e-6,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "lr_scheduler": "linear",
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj"],
            "gradient_checkpointing": False,
            "bf16": False,
            "fp16": False,
            "logging_steps": 1,
            "eval_steps": 1,
            "max_steps": 1,
            "resume_from": None,
        },
        "export": {"include_adapter_weights": True, "resume_from": None},
        "eval": {
            "cutoff_len": 64,
            "max_new_tokens": 16,
            "temperature": 0.7,
            "top_p": 0.9,
            "prompts": [],
            "resume_from": None,
        },
        "serve": {
            "host": "127.0.0.1",
            "port": 8080,
            "max_model_len": 512,
            "served_model_name": "dummy",
            "resume_from": None,
        },
        "logging": {
            "console_level": "INFO",
            "file_level": "DEBUG",
            "tqdm_refresh_rate": 1.0,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_finetune_failure_writes_traceback(tmp_path: Path, monkeypatch, runner: CliRunner) -> None:
    config_path = _write_failure_config(tmp_path)
    data_dir = tmp_path / "prepared"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (data_dir / "val.jsonl").write_text("{}\n", encoding="utf-8")

    def _failing_run(*args, **kwargs):
        resume_from_checkpoint = "/app/outputs/run/trainer_state"  # noqa: F841 - exercise locals capture
        adapter_weights_file = "/app/outputs/run/trainer_state/adapter_model.safetensors"  # noqa: F841
        raise ValueError("Can't find a valid checkpoint at /app/outputs/run/trainer_state")

    monkeypatch.setattr(SFTTrainerRunner, "run", _failing_run)

    result = runner.invoke(
        app,
        [
            "finetune-sft",
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "train_out"),
        ],
        catch_exceptions=True,
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)

    latest_pointer = tmp_path / "logs" / "latest.txt"
    assert latest_pointer.exists()
    stage_dir = Path(latest_pointer.read_text(encoding="utf-8").strip())
    assert stage_dir.exists()

    run_log = stage_dir / "run.log"
    console_log = stage_dir / "console.log"
    assert run_log.exists()
    assert console_log.exists()

    run_text = run_log.read_text(encoding="utf-8")
    console_text = console_log.read_text(encoding="utf-8")

    assert "ValueError: Can't find a valid checkpoint" in run_text
    assert "resume_from_checkpoint" in run_text
    assert "ValueError: Can't find a valid checkpoint" in console_text
    assert "\x1b[" not in console_text