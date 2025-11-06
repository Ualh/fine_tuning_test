# -*- coding: utf-8 -*-
"""Tests for the DRG classification trainer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from src.core.config import ConfigLoader
from src.training.drg_classifier import DRGClassificationTrainerRunner


@pytest.mark.slow
def test_drg_classifier_runner_trains_tiny_model(tmp_path: Path) -> None:
    loader = ConfigLoader(Path("debug_config.yaml"))
    cfg = loader.load()

    cfg.train.base_model = "hf-internal-testing/tiny-random-bert"
    cfg.train.batch_size = 1
    cfg.train.gradient_accumulation = 1
    cfg.train.epochs = 1
    cfg.train.max_steps = 1
    cfg.train.eval_steps = 1
    cfg.train.logging_steps = 1
    cfg.train.gradient_checkpointing = False
    cfg.train.bf16 = False
    cfg.train.fp16 = False
    cfg.train.report_to = ["none"]
    cfg.train.lora_target_modules = []

    cfg.preprocess.cutoff_len = 32

    data_dir = tmp_path / "prepared"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_records = [
        {"text": "Patient with DRG F62A", "drg": "F62A", "label": 0, "stay_id": "1", "path": "a", "n_tokens": 10},
        {"text": "Patient with DRG E77B", "drg": "E77B", "label": 1, "stay_id": "2", "path": "b", "n_tokens": 12},
    ]
    val_records = [
        {"text": "Another DRG F62A case", "drg": "F62A", "label": 0, "stay_id": "3", "path": "c", "n_tokens": 9},
    ]

    pd.DataFrame(train_records).to_parquet(data_dir / "train.parquet", index=False)
    pd.DataFrame(val_records).to_parquet(data_dir / "val.parquet", index=False)
    (data_dir / "label2id.json").write_text(json.dumps({"F62A": 0, "E77B": 1}), encoding="utf-8")

    output_dir = tmp_path / "outputs"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("test_drg_classifier")
    logger.setLevel(logging.INFO)
    trainer = DRGClassificationTrainerRunner(cfg.train, cfg.preprocess, logger=logger, run_dir=run_dir)

    summary = trainer.run(data_dir=data_dir, output_dir=output_dir)

    adapter_dir = output_dir / "adapter"
    assert adapter_dir.exists()
    assert summary.output_dir == output_dir
    metadata_path = output_dir / "metadata.json"
    assert metadata_path.exists()
*** End of File