# -*- coding: utf-8 -*-
"""Tests for the real-data preprocessing pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.core.config import ConfigLoader
from src.preprocessing.drg_pipeline import RealDatasetPreprocessor


@pytest.fixture()
def sample_meta_df(tmp_path: Path) -> pd.DataFrame:
    """Build a metadata DataFrame based on the bundled sample JSON letters."""

    sample_dir = Path("data") / "sample_real_data"
    rows: List[dict[str, str]] = []
    for json_path in sample_dir.glob("*.json"):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "eds_key": str(payload.get("eds_id", "")),
                "doc_key": str(payload.get("doc_id", "")),
                "update_begin_date_iso": payload.get("update_begin_date_iso", ""),
                "update_end_date_iso": payload.get("update_end_date_iso", ""),
                "update_begin_date_raw": payload.get("update_begin_date_raw", ""),
                "update_end_date_raw": payload.get("update_end_date_raw", ""),
            }
        )
    return pd.DataFrame(rows)


def _oracle_stub(eds_ids, *_args, **_kwargs) -> pd.DataFrame:
    records = []
    for eds in eds_ids:
        eds_int = int(str(eds))
        records.append(
            {
                "NOPTN": None,
                "EDS": eds_int,
                "DRG_OPA_CODE": "F62A",
                "DRG_OPA_LIB": "Test DRG",
                "SERMED_ID_LST": "",
                "SERMED_MNM_LST": "",
                "DT_DEB_SEJ": pd.Timestamp("2022-04-17"),
                "DT_FIN_SEJ": pd.Timestamp("2022-05-24"),
                "AGE_ANNEE": 75,
                "PRE_DRG_LIST": "F62A|E77B",
                "PRE_DRG_DATE_LIST": "2022-06-15|2022-05-01",
            }
        )
    return pd.DataFrame(records)


def test_real_dataset_preprocessor_builds_parquet(tmp_path: Path, sample_meta_df: pd.DataFrame) -> None:
    loader = ConfigLoader(Path("debug_config.yaml"))
    cfg = loader.load()
    cfg.preprocess.real_data.use_sample_data = True
    cfg.preprocess.real_data.min_count = 1
    cfg.preprocess.real_data.copy_splits = False

    logger = logging.getLogger("test_real_preprocess")
    logger.setLevel(logging.INFO)

    preprocessor = RealDatasetPreprocessor(
        cfg.preprocess,
        cfg.train,
        cfg.paths,
        logger,
        project_root=cfg.project_root,
        oracle_fetcher=_oracle_stub,
        metadata_loader=lambda _path: sample_meta_df,
    )

    output_dir = tmp_path / "prepared_real_sample"
    summary = preprocessor.run(output_dir=output_dir)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    label_map = output_dir / "label2id.json"
    metadata_path = output_dir / "metadata.json"

    assert train_path.exists()
    assert val_path.exists()
    assert label_map.exists()
    assert metadata_path.exists()

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    assert not train_df.empty
    assert set(train_df.columns) == {"text", "drg", "label", "stay_id", "path", "n_tokens"}
    assert summary.dataset.total_examples == len(train_df) + len(val_df)
    assert summary.enrichment.formatted_files >= summary.dataset.total_examples