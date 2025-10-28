import logging
import warnings
from pathlib import Path
from typer.testing import CliRunner
import yaml
import json

from src.core.logger import configure_logging
from src.cli.main import app
from src.data.preprocess import DataPreprocessor, PreprocessSummary
from src.training.sft_trainer import SFTTrainerRunner, TrainingSummary, StageProgressCallback


class _DummyTokenizer:
    def __init__(self) -> None:
        self.chat_template = "{{ messages }}"
        self.eos_token = "<eos>"
        self.pad_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(str(m.get("content", "")) for m in messages)

    def save_pretrained(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)


class _DummyAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _DummyTokenizer()


def test_info_logs_go_to_run_log_and_not_stdout(tmp_path, capsys):
    run_dir = tmp_path / "run"
    # configure logger: console WARNING, file DEBUG
    logger = configure_logging(name="test-routing", log_dir=run_dir, console_level=logging.WARNING, file_level=logging.DEBUG)

    # Emit an INFO message which should go to the run.log (file) but not to stdout
    logger.info("TEST_METRIC: %s", {"loss": 0.123})

    # Also emit a WARNING which should appear on console (captured by capsys)
    logger.warning("TEST_WARNING: this should be on console")

    # read run.log
    log_file = run_dir / "run.log"
    assert log_file.exists(), "run.log should be created"
    content = log_file.read_text(encoding="utf-8")
    assert "TEST_METRIC" in content
    assert "TEST_WARNING" in content

    captured = capsys.readouterr()
    # INFO-level metric should not be on stdout
    assert "TEST_METRIC" not in captured.out
    # WARNING should be on stdout
    assert "TEST_WARNING" in captured.out


def test_warnings_redirected_to_run_log(tmp_path, capsys):
    run_dir = tmp_path / "run"
    logger = configure_logging(name="test-warnings", log_dir=run_dir, console_level=logging.WARNING, file_level=logging.DEBUG)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warnings.warn("deprecated api", DeprecationWarning)

    assert recorded, "warnings should still be emitted for introspection"

    content = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "deprecated api" in content

    captured = capsys.readouterr()
    assert "deprecated api" not in captured.out


def test_stage_progress_callback_labels_bars():
    callback = StageProgressCallback()

    class _DummyBar:
        def __init__(self) -> None:
            self.desc = None

        def set_description(self, value, refresh=True):
            self.desc = value

    dummy_args = object()
    dummy_state = object()
    dummy_control = object()

    # Monkeypatch progress bar factory so we can inspect the description.
    bars = {}

    def _fake_init_progress_bar(*_, **__):
        bar = _DummyBar()
        bars.setdefault("calls", []).append(bar)
        return bar

    callback._init_progress_bar = _fake_init_progress_bar  # type: ignore[assignment]

    callback.on_train_begin(dummy_args, dummy_state, dummy_control)
    assert bars["calls"][0].desc == "TRAIN"

    callback.on_eval_begin(dummy_args, dummy_state, dummy_control)
    assert bars["calls"][1].desc == "EVAL"

    callback.on_predict_begin(dummy_args, dummy_state, dummy_control)
    assert bars["calls"][2].desc == "EVAL"


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
        "export": {"include_adapter_weights": True, "resume_from": None},
        "eval": {"cutoff_len": 64, "max_new_tokens": 64, "temperature": 0.7, "top_p": 0.9, "prompts": [], "resume_from": None},
        "serve": {"host": "127.0.0.1", "port": 8080, "max_model_len": 512, "served_model_name": "dummy", "resume_from": None},
        "logging": {"console_level": "WARNING", "file_level": "DEBUG", "tqdm_refresh_rate": 1.0},
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
    return PreprocessSummary(output_dir=output_dir, sample_size=2, kept_languages=["en"], discarded=0, seed=42)


def _stub_trainer_run(self, data_dir: Path, output_dir: Path, resume_path=None):
    # Simulate trainer emitting metrics via logger.info (these should go to run.log)
    self.logger.info("TRAIN_METRICS: %s", {"loss": 0.321, "epoch": 0.1})
    output_dir = Path(output_dir)
    (output_dir / "adapter").mkdir(parents=True, exist_ok=True)
    (output_dir / "trainer_state").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps({"train": {"train_loss": 0.321}}), encoding="utf-8")
    return TrainingSummary(output_dir=output_dir, train_loss=0.321, eval_loss=0.0, epochs=0.1, total_tokens=10)


def test_cli_smoke_writes_metrics_to_run_log_and_not_console(tmp_path, monkeypatch):
    # Prepare config and monkeypatch heavy parts
    config_path = _write_temp_config(tmp_path)
    monkeypatch.setattr("src.data.preprocess.AutoTokenizer", _DummyAutoTokenizer)
    monkeypatch.setattr("src.training.sft_trainer.AutoTokenizer", _DummyAutoTokenizer)
    monkeypatch.setattr(DataPreprocessor, "run", _stub_preprocess_run)
    monkeypatch.setattr(SFTTrainerRunner, "run", _stub_trainer_run)

    runner = CliRunner()

    # Run preprocess then finetune via CLI (the CLI writes latest.txt pointing to train stage)
    prep_result = runner.invoke(app, ["preprocess-sft", "--config", str(config_path), "--output", str(tmp_path / "prepared")], catch_exceptions=False)
    assert prep_result.exit_code == 0, prep_result.stdout

    train_result = runner.invoke(app, ["finetune-sft", "--config", str(config_path), "--data-dir", str(tmp_path / "prepared"), "--output-dir", str(tmp_path / "train_out")], catch_exceptions=False)
    assert train_result.exit_code == 0, train_result.stdout

    latest_pointer = tmp_path / "logs" / "latest.txt"
    assert latest_pointer.exists()
    latest_path = Path(latest_pointer.read_text(encoding="utf-8").strip())
    # Read run.log from the train stage
    run_log = latest_path / "run.log"
    assert run_log.exists(), "run.log should exist in the latest stage dir"
    content = run_log.read_text(encoding="utf-8")
    assert "TRAIN_METRICS" in content or "FINAL_TRAIN_METRICS" in content

    # Ensure CLI stdout did not include the metric line
    assert "TRAIN_METRICS" not in train_result.stdout
