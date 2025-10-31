"""TensorBoard integration smoke test."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from typer.testing import CliRunner

from src.cli.main import app
from src.data.preprocess import DataPreprocessor, PreprocessSummary
from src.training.sft_trainer import SFTTrainerRunner


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
    def vocab_size(self) -> int:  # pragma: no cover - unused but mirrors interface
        return 1

    def get_vocab(self) -> dict[str, int]:  # pragma: no cover - unused but mirrors interface
        return {"<eos>": 0}


class _StubTrainer:
    def __init__(self, *_, **kwargs) -> None:
        self.args = kwargs.get("args")
        self.callbacks = list(kwargs.get("callbacks", []))

    def remove_callback(self, _callback_type) -> None:
        pass

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def train(self, resume_from_checkpoint=None):  # pylint: disable=unused-argument
        logdir = Path(self.args.logging_dir)
        logdir.mkdir(parents=True, exist_ok=True)
        (logdir / "events.out.tfevents.mock").write_text("", encoding="utf-8")
        return SimpleNamespace(metrics={"train_loss": 0.0, "epoch": 0.1})

    def evaluate(self):
        return {"eval_loss": 0.0}

    def save_model(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)


@pytest.fixture()
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
            "logging_steps": 1,
            "eval_steps": 1,
            "report_to": ["tensorboard"],
            "logging_dir": None,
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
    config_path.write_text(json.dumps(base), encoding="utf-8")
    return config_path


def _stub_preprocess(self, output_dir: Path, resume_dir=None):  # pylint: disable=unused-argument
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.jsonl").write_text('{"text": "hello"}\n', encoding="utf-8")
    (output_dir / "val.jsonl").write_text('{"text": "world"}\n', encoding="utf-8")
    return PreprocessSummary(
        output_dir=output_dir,
        sample_size=2,
        kept_languages=["en"],
        discarded=0,
        seed=42,
    )


def _stub_load_splits(_self, _data_dir: Path):
    return {"train": ["train"], "validation": ["val"]}


def test_tensorboard_events_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    config_path = _write_temp_config(tmp_path)

    monkeypatch.setattr("src.data.preprocess.AutoTokenizer", lambda *args, **kwargs: _DummyTokenizer())
    monkeypatch.setattr("src.training.sft_trainer.AutoTokenizer", lambda *args, **kwargs: _DummyTokenizer())
    monkeypatch.setattr("src.training.sft_trainer.AutoModelForCausalLM", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.training.sft_trainer.ensure_chat_template", lambda tokenizer, logger=None: tokenizer)
    monkeypatch.setattr("src.training.sft_trainer.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(DataPreprocessor, "run", _stub_preprocess)
    monkeypatch.setattr(SFTTrainerRunner, "_load_jsonl_splits", _stub_load_splits)
    monkeypatch.setattr("src.training.sft_trainer.SFTTrainer", _StubTrainer)

    prep_dir = tmp_path / "prepared"
    train_dir = tmp_path / "outputs"

    result_prep = runner.invoke(
        app,
        [
            "preprocess-sft",
            "--config",
            str(config_path),
            "--output",
            str(prep_dir),
        ],
        catch_exceptions=False,
    )
    assert result_prep.exit_code == 0, result_prep.stdout

    result_train = runner.invoke(
        app,
        [
            "finetune-sft",
            "--config",
            str(config_path),
            "--data-dir",
            str(prep_dir),
            "--output-dir",
            str(train_dir),
        ],
        catch_exceptions=False,
    )
    assert result_train.exit_code == 0, result_train.stdout

    latest_pointer = Path(tmp_path / "logs" / "latest.txt")
    assert latest_pointer.exists()
    train_stage_dir = Path(latest_pointer.read_text(encoding="utf-8").strip())
    tensorboard_dir = train_stage_dir / "tensorboard"
    assert tensorboard_dir.exists(), "TensorBoard directory was not created"
    event_files = list(tensorboard_dir.glob("events.out.tfevents*"))
    assert event_files, "TensorBoard event files missing"
"""TensorBoard integration smoke test."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest
"""TensorBoard integration smoke test."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from typer.testing import CliRunner

from src.cli.main import app
from src.data.preprocess import DataPreprocessor, PreprocessSummary
from src.training.sft_trainer import SFTTrainerRunner


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
    def vocab_size(self) -> int:  # pragma: no cover - unused but mirrors interface
        return 1

    def get_vocab(self) -> dict[str, int]:  # pragma: no cover - unused but mirrors interface
        return {"<eos>": 0}


class _StubTrainer:
    def __init__(self, *_, **kwargs) -> None:
        self.args = kwargs.get("args")
        self.callbacks = list(kwargs.get("callbacks", []))

    def remove_callback(self, _callback_type) -> None:
        pass

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def train(self, resume_from_checkpoint=None):  # pylint: disable=unused-argument
        logdir = Path(self.args.logging_dir)
        logdir.mkdir(parents=True, exist_ok=True)
        (logdir / "events.out.tfevents.mock").write_text("", encoding="utf-8")
        return SimpleNamespace(metrics={"train_loss": 0.0, "epoch": 0.1})

    def evaluate(self):
        return {"eval_loss": 0.0}

    def save_model(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)


@pytest.fixture()
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
            "logging_steps": 1,
            "eval_steps": 1,
            "report_to": ["tensorboard"],
            "logging_dir": None,
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
    config_path.write_text(json.dumps(base), encoding="utf-8")
    return config_path


def _stub_preprocess(self, output_dir: Path, resume_dir=None):  # pylint: disable=unused-argument
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.jsonl").write_text('{"text": "hello"}\n', encoding="utf-8")
    (output_dir / "val.jsonl").write_text('{"text": "world"}\n', encoding="utf-8")
    return PreprocessSummary(
        output_dir=output_dir,
        sample_size=2,
        kept_languages=["en"],
        discarded=0,
        seed=42,
    )


def _stub_load_splits(_self, _data_dir: Path):
    return {"train": ["train"], "validation": ["val"]}


def test_tensorboard_events_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    config_path = _write_temp_config(tmp_path)

    monkeypatch.setattr("src.data.preprocess.AutoTokenizer", lambda *args, **kwargs: _DummyTokenizer())
    monkeypatch.setattr("src.training.sft_trainer.AutoTokenizer", lambda *args, **kwargs: _DummyTokenizer())
    monkeypatch.setattr("src.training.sft_trainer.AutoModelForCausalLM", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.training.sft_trainer.ensure_chat_template", lambda tokenizer, logger=None: tokenizer)
    monkeypatch.setattr("src.training.sft_trainer.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(DataPreprocessor, "run", _stub_preprocess)
    monkeypatch.setattr(SFTTrainerRunner, "_load_jsonl_splits", _stub_load_splits)
    monkeypatch.setattr("src.training.sft_trainer.SFTTrainer", _StubTrainer)

    prep_dir = tmp_path / "prepared"
    train_dir = tmp_path / "outputs"

    result_prep = runner.invoke(
        app,
        [
            "preprocess-sft",
            "--config",
            str(config_path),
            "--output",
            str(prep_dir),
        ],
        catch_exceptions=False,
    )
    assert result_prep.exit_code == 0, result_prep.stdout

    result_train = runner.invoke(
        app,
        [
            "finetune-sft",
            "--config",
            str(config_path),
            "--data-dir",
            str(prep_dir),
            "--output-dir",
            str(train_dir),
        ],
        catch_exceptions=False,
    )
    assert result_train.exit_code == 0, result_train.stdout

    latest_pointer = Path(tmp_path / "logs" / "latest.txt")
    assert latest_pointer.exists()
    train_stage_dir = Path(latest_pointer.read_text(encoding="utf-8").strip())
    tensorboard_dir = train_stage_dir / "tensorboard"
    assert tensorboard_dir.exists(), "TensorBoard directory was not created"
    event_files = list(tensorboard_dir.glob("events.out.tfevents*"))
    assert event_files, "TensorBoard event files missing"
