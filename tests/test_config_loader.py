from pathlib import Path

import pytest
import yaml

from src.core.config import ConfigLoader
from src.core.errors import PipelineConfigError


def test_config_loader_reads_defaults():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    loader = ConfigLoader(config_path)
    cfg = loader.load()

    assert cfg.train.base_model == "Qwen/Qwen2-7B"
    assert cfg.preprocess.sample_size is None  # 'full' maps to None internally
    assert cfg.paths.logs_dir.name == "logs"
    assert cfg.logging.console_level == "INFO"
    assert cfg.logging.debug_pipeline is False
    assert cfg.serve.served_model_relpath is None
    assert cfg.serve.prefer_awq is True
    assert cfg.serve.model_name is None
    assert cfg.project_root == config_path.parent


def test_real_mode_requires_oracle_credentials(tmp_path: Path) -> None:
    config_payload = {
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
            "mode": "real_drg",
            "dataset_name": "acme/dummy",
            "sample_size": 4,
            "filter_langs": ["en"],
            "test_size": 0.1,
            "cutoff_len": 64,
            "seed": 1,
            "save_splits": True,
            "max_workers": 1,
            "pack_sequences": False,
            "real_data": {
                "oracle": {
                    "enabled": False,
                }
            },
        },
        "train": {
            "base_model": "acme/model",
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
            "logging_steps": 5,
            "eval_steps": 5,
            "max_steps": 1,
            "resume_from": None,
            "report_to": ["none"],
            "logging_dir": None,
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
            "host": "0.0.0.0",
            "port": 8080,
            "max_model_len": 256,
            "served_model_name": "dummy",
            "prefer_awq": False,
        },
        "logging": {
            "console_level": "INFO",
            "file_level": "DEBUG",
            "tqdm_refresh_rate": 1.0,
        },
    }

    config_path = tmp_path / "missing_oracle.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    loader = ConfigLoader(config_path)
    with pytest.raises(PipelineConfigError) as exc:
        loader.load()

    assert "preprocess.real_data.oracle" in str(exc.value)


def test_real_mode_stub_mode_allows_offline_run(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()

    config_payload = {
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
            "mode": "real_drg",
            "dataset_name": "acme/dummy",
            "sample_size": 4,
            "filter_langs": ["en"],
            "test_size": 0.1,
            "cutoff_len": 64,
            "seed": 1,
            "save_splits": True,
            "max_workers": 1,
            "pack_sequences": False,
            "real_data": {
                "sample_dir": str(sample_dir),
                "use_sample_data": True,
                "oracle": {
                    "enabled": False,
                    "stub_mode": True,
                    "stub_data_path": None,
                },
            },
        },
        "train": {
            "base_model": "acme/model",
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
            "logging_steps": 5,
            "eval_steps": 5,
            "max_steps": 1,
            "resume_from": None,
            "report_to": ["none"],
            "logging_dir": None,
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
            "host": "0.0.0.0",
            "port": 8080,
            "max_model_len": 256,
            "served_model_name": "dummy",
            "prefer_awq": False,
        },
        "logging": {
            "console_level": "INFO",
            "file_level": "DEBUG",
            "tqdm_refresh_rate": 1.0,
        },
    }

    config_path = tmp_path / "stub_oracle.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    loader = ConfigLoader(config_path)
    cfg = loader.load()

    assert cfg.preprocess.real_data.oracle.stub_mode is True
    assert cfg.preprocess.real_data.oracle.enabled is False
