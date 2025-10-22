"""Lightweight orchestration tests that exercise the Typer CLI without heavy downloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from src.cli.main import app
from src.data.preprocess import DataPreprocessor, PreprocessSummary
from src.training.sft_trainer import SFTTrainerRunner, TrainingSummary
class _DummyTokenizer:
    def __init__(self) -> None:
        self.chat_template = "{{ messages }}"
        self.eos_token = "<eos>"
        self.pad_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(str(m.get("content", "")) for m in messages)

    def save_pretrained(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)

    @property
    def vocab_size(self) -> int:
        return 1

    def get_vocab(self) -> dict[str, int]:
        return {"<eos>": 0}


class _DummyAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _DummyTokenizer()



@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


def _write_temp_config(tmp_path: Path) -> Path:
    base = {
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
            "sample_size": 4,
            "filter_langs": ["en"],
            "test_size": 0.25,
            "cutoff_len": 64,
            "seed": 42,
            "save_splits": True,
            "max_workers": 1,
            "pack_sequences": False,
        },
        "train": {
            "base_model": "acme/dummy-model",
            "cutoff_len": 64,
            "batch_size": 2,
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
            "logging_steps": 5,
            "eval_steps": 5,
            "max_steps": 1,
            "resume_from": None,
        },
        "export": {
            "include_adapter_weights": True,
            "resume_from": None,
        },
        "eval": {
            "cutoff_len": 64,
            "max_new_tokens": 64,
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return config_path


def _stub_preprocess_run(self, output_dir: Path, resume_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.jsonl").write_text('{"text": "hello"}\n', encoding="utf-8")
    (output_dir / "val.jsonl").write_text('{"text": "world"}\n', encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps({"preprocess": {"sample_size": 2}}), encoding="utf-8")
    return PreprocessSummary(
        output_dir=output_dir,
        sample_size=2,
        kept_languages=["en"],
        discarded=0,
        seed=self.config.seed,
    )


def _stub_trainer_run(self, data_dir: Path, output_dir: Path, resume_path=None):
    data_dir = Path(data_dir)
    assert (data_dir / "train.jsonl").exists(), "preprocess output missing train.jsonl"
    assert (data_dir / "val.jsonl").exists(), "preprocess output missing val.jsonl"

    output_dir = Path(output_dir)
    (output_dir / "adapter").mkdir(parents=True, exist_ok=True)
    (output_dir / "trainer_state").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps({"train": {"train_loss": 0.0, "eval_loss": 0.0}}),
        encoding="utf-8",
    )
    return TrainingSummary(
        output_dir=output_dir,
        train_loss=0.0,
        eval_loss=0.0,
        epochs=0.1,
        total_tokens=32,
    )


@pytest.mark.usefixtures("runner")
def test_cli_preprocess_and_finetune_smoke(tmp_path, monkeypatch, runner: CliRunner):
    config_path = _write_temp_config(tmp_path)
    monkeypatch.setattr("src.data.preprocess.AutoTokenizer", _DummyAutoTokenizer)
    monkeypatch.setattr("src.training.sft_trainer.AutoTokenizer", _DummyAutoTokenizer)
    monkeypatch.setattr(DataPreprocessor, "run", _stub_preprocess_run)
    monkeypatch.setattr(SFTTrainerRunner, "run", _stub_trainer_run)

    prepared_output = tmp_path / "prepared_out"
    train_output = tmp_path / "train_out"

    prep_result = runner.invoke(
        app,
        [
            "preprocess-sft",
            "--config",
            str(config_path),
            "--output",
            str(prepared_output),
        ],
        catch_exceptions=False,
    )
    assert prep_result.exit_code == 0, prep_result.stdout
    assert (prepared_output / "train.jsonl").exists()
    assert (prepared_output / "val.jsonl").exists()

    train_result = runner.invoke(
        app,
        [
            "finetune-sft",
            "--config",
            str(config_path),
            "--data-dir",
            str(prepared_output),
            "--output-dir",
            str(train_output),
        ],
        catch_exceptions=False,
    )
    assert train_result.exit_code == 0, train_result.stdout
    assert (train_output / "adapter").exists()
    assert (train_output / "metadata.json").exists()

    latest_pointer = Path(tmp_path / "logs" / "latest.txt")
    assert latest_pointer.exists()
    latest_path = Path(latest_pointer.read_text(encoding="utf-8").strip())
    assert latest_path.exists()
    assert latest_path.name in {"train", "preprocess"}

    run_dirs = list((tmp_path / "logs").glob("log_v*/"))
    assert run_dirs, "run directories were not created"
    for run_dir in run_dirs:
        for stage_dir in [p for p in run_dir.iterdir() if p.is_dir()]:
            run_log = stage_dir / "run.log"
            console_log = stage_dir / "console.log"
            assert run_log.exists(), f"run.log missing in {stage_dir}"
            assert run_log.stat().st_size > 0, f"run.log empty in {stage_dir}"
            assert console_log.exists(), f"console.log missing in {stage_dir}"
            assert console_log.stat().st_size > 0, f"console.log empty in {stage_dir}"