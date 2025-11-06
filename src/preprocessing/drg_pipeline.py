# -*- coding: utf-8 -*-
"""
preprocessing.drg_pipeline
Purpose: Bridge raw hospital discharge letters to classification-ready splits using mandatory Oracle enrichment.
Stage: preprocessing — enrich JSON notes via Oracle, then build train/val parquet with labels.
Example: invoked automatically by `preprocess-sft` when `preprocess.mode` is ``real_drg`` (see src/cli/main.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from ..core.config import PathsConfig, PreprocessConfig, TrainConfig
from ..core.io_utils import atomic_write_json, ensure_dir
from .fill_notes import (
    collect_eds_ids,
    fetch_oracle_data,
    load_metadata_csv,
    update_json_files,
)

CAND_LABEL_KEYS: Sequence[str] = ("predrg_max", "drg_target")


@dataclass
class EnrichmentSummary:
    """Report produced by the Oracle enrichment pass."""

    input_dir: Path
    formatted_dir: Path
    log_csv: Optional[Path]
    quarantined_dir: Path
    total_raw_files: int
    formatted_files: int
    quarantined_files: int


@dataclass
class DatasetSummary:
    """Report describing the built classification dataset."""

    train_examples: int
    val_examples: int
    total_examples: int
    n_labels: int
    label2id_path: Path
    stats_path: Path
    meta_path: Path


@dataclass
class RealPreprocessSummary:
    """Payload returned to the CLI summarising the real-data preprocessing."""

    output_dir: Path
    enrichment: EnrichmentSummary
    dataset: DatasetSummary


def choose_drg(payload: Dict[str, object], truncate_label_to: Optional[int]) -> Optional[str]:
    """Select the preferred DRG label from enriched payload fields."""

    for key in CAND_LABEL_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            cleaned = value.strip()
            if truncate_label_to and len(cleaned) >= truncate_label_to:
                return cleaned[:truncate_label_to]
            return cleaned
    return None


def _iter_json_files(root: Path) -> Iterable[Path]:
    excluded = {"no_drg_eds_files", "train_split", "val_split", "logs"}
    for candidate in root.rglob("*.json"):
        if any(part in excluded for part in candidate.parts):
            continue
        yield candidate


class ClassificationDatasetBuilder:
    """Replicate the legacy classification dataset builder on top of enriched JSONs."""

    def __init__(
        self,
        preprocess_cfg: PreprocessConfig,
        train_cfg: TrainConfig,
        logger: logging.Logger,
        *,
        hf_token: Optional[str] = None,
    ) -> None:
        self.preprocess_cfg = preprocess_cfg
        self.real_cfg = preprocess_cfg.real_data
        self.train_cfg = train_cfg
        self.logger = logger
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._tokenizer: Optional[AutoTokenizer] = None

    # ------------------------------------------------------------------
    def build(self, formatted_dir: Path, output_dir: Path) -> DatasetSummary:
        formatted_dir = Path(formatted_dir)
        output_dir = ensure_dir(Path(output_dir))

        df_all = self._load_json_dir(formatted_dir)
        if df_all.empty:
            raise ValueError(f"No enriched JSON files found under {formatted_dir}.")

        df_all = self._filter_by_prefix(df_all)
        if df_all.empty:
            raise ValueError("All samples were filtered out by keep_prefixes settings.")

        df_all["n_tokens"] = self._tokenize_lengths(df_all["text"].tolist())
        before_len = len(df_all)
        df_all = df_all[df_all["n_tokens"] >= self.real_cfg.min_tokens].copy()
        removed = before_len - len(df_all)
        self.logger.info("Filtered %s samples below min_tokens=%s", removed, self.real_cfg.min_tokens)
        if df_all.empty:
            raise ValueError("No samples remain after enforcing min_tokens.")

        df_train, df_val = self._split_train_val(df_all)
        label_stats = self._map_rare_labels(df_train, df_val)
        df_train, df_val, frequent_json = label_stats

        label2id = self._build_label2id(df_train)
        df_train["label"] = df_train["drg"].map(label2id).astype(int)
        df_val = df_val[df_val["drg"].isin(label2id)].copy()
        df_val["label"] = df_val["drg"].map(label2id).astype(int)

        dataset_summary = self._write_outputs(df_train, df_val, label2id, frequent_json, output_dir)
        return dataset_summary

    # ------------------------------------------------------------------
    def _load_json_dir(self, dir_path: Path) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        files = sorted(_iter_json_files(dir_path))
        total = len(files)
        empty_text = 0
        missing_drg = 0
        for json_path in tqdm(files, desc="LOAD", leave=False):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover - malformed file
                continue
            text = payload.get("text") or payload.get("note") or ""
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            if not text.strip():
                empty_text += 1
                continue
            drg = choose_drg(payload, self.real_cfg.truncate_label_to)
            if drg is None:
                missing_drg += 1
                continue
            stay_id = payload.get("eds_id") or payload.get("stay_id") or json_path.stem
            rows.append(
                {
                    "text": text,
                    "drg": drg.strip(),
                    "stay_id": str(stay_id),
                    "path": str(json_path),
                }
            )
        self.logger.info(
            "Loaded %s JSON files (%s usable, %s missing text, %s missing DRG)",
            total,
            len(rows),
            empty_text,
            missing_drg,
        )
        return pd.DataFrame(rows)

    def _filter_by_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        keep_prefixes = self.real_cfg.keep_prefixes
        if not keep_prefixes:
            return df
        pattern = re.compile(r"^(%s)" % "|".join(map(re.escape, keep_prefixes)))
        mask = df["drg"].astype(str).str.match(pattern)
        if self.real_cfg.outside_to_other:
            df = df.copy()
            df.loc[~mask, "drg"] = self.real_cfg.outside_label
            self.logger.info(
                "Mapped %s samples outside prefixes %s to %s",
                (~mask).sum(),
                keep_prefixes,
                self.real_cfg.outside_label,
            )
            return df
        filtered = df[mask].copy()
        self.logger.info(
            "Filtered dataset by prefixes %s -> kept %s / %s",
            keep_prefixes,
            len(filtered),
            len(df),
        )
        return filtered

    def _tokenize_lengths(self, texts: List[str]) -> List[int]:
        if self._tokenizer is None:
            self.logger.info(
                "Loading tokenizer %s (cutoff_len=%s)",
                self.real_cfg.tokenizer,
                self.preprocess_cfg.cutoff_len,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.real_cfg.tokenizer,
                model_max_length=self.preprocess_cfg.cutoff_len,
                token=self.hf_token,
            )
        lengths: List[int] = []
        batch_size = 32
        for idx in range(0, len(texts), batch_size):
            chunk = texts[idx : idx + batch_size]
            encoded = self._tokenizer(
                chunk,
                truncation=False,
                add_special_tokens=False,
            )
            chunk_lengths = [len(ids) for ids in encoded["input_ids"]]
            lengths.extend(chunk_lengths)
        return lengths

    def _split_train_val(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        test_size = self.real_cfg.test_size_override or self.preprocess_cfg.test_size
        rng = self.preprocess_cfg.seed
        counts = df["drg"].value_counts()
        rare_labels = counts[counts < 2].index.tolist()
        df_common = df[~df["drg"].isin(rare_labels)].copy()
        df_rare = df[df["drg"].isin(rare_labels)].copy()
        if df_common.empty:
            indices = np.arange(len(df))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=rng,
                shuffle=True,
            )
            return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
        train_common, val_common = train_test_split(
            df_common,
            test_size=test_size,
            random_state=rng,
            stratify=df_common["drg"],
        )
        df_train = pd.concat([train_common, df_rare], ignore_index=True)
        df_val = val_common.copy()
        self.logger.info(
            "Split dataset -> train=%s, val=%s (test_size=%.3f)",
            len(df_train),
            len(df_val),
            test_size,
        )
        return df_train, df_val

    def _map_rare_labels(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        min_count = self.real_cfg.min_count
        if min_count <= 0:
            return df_train, df_val, {}
        counts = df_train["drg"].value_counts()
        frequent = set(counts[counts >= min_count].index.tolist())
        other_label = self.real_cfg.outside_label

        def _map(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.loc[~df["drg"].isin(frequent), "drg"] = other_label
            return df

        df_train = _map(df_train)
        df_val = _map(df_val)
        stats = {label: int(counts.get(label, 0)) for label in sorted(frequent)}
        self.logger.info(
            "Grouped rare labels (<%s occurrences) into %s. Frequent label count: %s",
            min_count,
            other_label,
            len(stats),
        )
        return df_train, df_val, stats

    def _build_label2id(self, df_train: pd.DataFrame) -> Dict[str, int]:
        unique = sorted(df_train["drg"].unique().tolist())
        label2id = {label: idx for idx, label in enumerate(unique)}
        self.logger.info("label2id contains %s labels", len(label2id))
        return label2id

    def _write_outputs(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        label2id: Dict[str, int],
        frequent_json: Dict[str, int],
        output_dir: Path,
    ) -> DatasetSummary:
        output_dir = ensure_dir(output_dir)
        df_train_out = df_train[["text", "drg", "label", "stay_id", "path", "n_tokens"]].reset_index(drop=True)
        df_val_out = df_val[["text", "drg", "label", "stay_id", "path", "n_tokens"]].reset_index(drop=True)

        train_path = output_dir / "train.parquet"
        val_path = output_dir / "val.parquet"
        df_train_out.to_parquet(train_path, index=False)
        df_val_out.to_parquet(val_path, index=False)

        label2id_path = output_dir / "label2id.json"
        with label2id_path.open("w", encoding="utf-8") as handle:
            json.dump(label2id, handle, ensure_ascii=False, indent=2)

        if frequent_json:
            frequent_path = output_dir / "frequent_labels.json"
            with frequent_path.open("w", encoding="utf-8") as handle:
                json.dump(frequent_json, handle, ensure_ascii=False, indent=2)

        stats_payload = {
            "mode": "real_drg",
            "test_size": self.real_cfg.test_size_override or self.preprocess_cfg.test_size,
            "min_tokens": self.real_cfg.min_tokens,
            "min_count": self.real_cfg.min_count,
            "tokenizer": self.real_cfg.tokenizer,
            "cutoff_len": self.preprocess_cfg.cutoff_len,
            "total_examples": int(len(df_train_out) + len(df_val_out)),
            "train_examples": int(len(df_train_out)),
            "val_examples": int(len(df_val_out)),
            "n_labels": int(len(label2id)),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        }
        stats_path = output_dir / "stats.json"
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats_payload, handle, ensure_ascii=False, indent=2)

        meta_payload = {
            "seed": self.preprocess_cfg.seed,
            "min_tokens": self.real_cfg.min_tokens,
            "min_count": self.real_cfg.min_count,
            "keep_prefixes": self.real_cfg.keep_prefixes,
            "outside_to_other": self.real_cfg.outside_to_other,
            "outside_label": self.real_cfg.outside_label,
            "truncate_label_to": self.real_cfg.truncate_label_to,
            "tokenizer": self.real_cfg.tokenizer,
            "test_size": self.real_cfg.test_size_override or self.preprocess_cfg.test_size,
        }
        meta_path = output_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta_payload, handle, ensure_ascii=False, indent=2)

        if self.real_cfg.copy_splits:
            copied_train = self._copy_split_files(df_train_out, output_dir / "train_split")
            copied_val = self._copy_split_files(df_val_out, output_dir / "val_split")
            self.logger.info(
                "Copied original JSON files for splits -> train=%s, val=%s",
                copied_train,
                copied_val,
            )

        return DatasetSummary(
            train_examples=int(len(df_train_out)),
            val_examples=int(len(df_val_out)),
            total_examples=int(len(df_train_out) + len(df_val_out)),
            n_labels=int(len(label2id)),
            label2id_path=label2id_path,
            stats_path=stats_path,
            meta_path=meta_path,
        )

    def _copy_split_files(self, df: pd.DataFrame, dst_dir: Path) -> int:
        dst_dir = ensure_dir(dst_dir)
        copied = 0
        for src in df["path"]:
            try:
                src_path = Path(src)
                if not src_path.exists():
                    continue
                target = dst_dir / src_path.name
                shutil.copy2(src_path, target)
                copied += 1
            except Exception:  # pragma: no cover - best effort copy
                continue
        return copied


class RealDatasetPreprocessor:
    """Coordinate enrichment (Oracle) and dataset building for the real letters pipeline."""

    def __init__(
        self,
        preprocess_cfg: PreprocessConfig,
        train_cfg: TrainConfig,
        paths_cfg: PathsConfig,
        logger: logging.Logger,
        *,
        project_root: Optional[Path] = None,
        oracle_fetcher: Optional[Callable[..., pd.DataFrame]] = None,
        metadata_loader: Optional[Callable[[str], pd.DataFrame]] = None,
    ) -> None:
        self.preprocess_cfg = preprocess_cfg
        self.train_cfg = train_cfg
        self.paths_cfg = paths_cfg
        self.logger = logger
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.oracle_fetcher = oracle_fetcher or fetch_oracle_data
        self.metadata_loader = metadata_loader or load_metadata_csv

    # ------------------------------------------------------------------
    def run(self, output_dir: Path, resume_dir: Optional[Path] = None) -> RealPreprocessSummary:
        output_dir = ensure_dir(Path(resume_dir) if resume_dir else Path(output_dir))
        if resume_dir:
            self.logger.info("Reusing existing prepared directory: %s", resume_dir)

        real_cfg = self.preprocess_cfg.real_data
        input_dir = self._resolve_input_dir(real_cfg)
        formatted_dir = ensure_dir(output_dir / real_cfg.formatted_subdir)
        logs_dir = ensure_dir(output_dir / "logs")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_csv_path = logs_dir / f"fill_notes_{ts}.csv"
        sql_path = (
            Path(real_cfg.oracle.debug_sql_dir)
            if real_cfg.oracle.debug_sql_dir
            else logs_dir / f"oracle_query_{ts}.sql"
        )

        enrichment = self._enrich_notes(
            input_dir=input_dir,
            formatted_dir=formatted_dir,
            log_csv_path=log_csv_path,
            sql_path=sql_path,
        )

        builder = ClassificationDatasetBuilder(self.preprocess_cfg, self.train_cfg, self.logger)
        dataset_summary = builder.build(formatted_dir, output_dir)

        metadata_payload = {
            "mode": "real_drg",
            "output_dir": str(output_dir),
            "formatted_dir": str(formatted_dir),
            "train_examples": dataset_summary.train_examples,
            "val_examples": dataset_summary.val_examples,
            "n_labels": dataset_summary.n_labels,
            "enrichment": asdict(enrichment),
            "dataset": asdict(dataset_summary),
        }
        atomic_write_json(metadata_payload, output_dir / "metadata.json")

        return RealPreprocessSummary(output_dir=output_dir, enrichment=enrichment, dataset=dataset_summary)

    # ------------------------------------------------------------------
    def _resolve_input_dir(self, real_cfg) -> Path:
        if real_cfg.use_sample_data:
            if not real_cfg.sample_dir:
                raise ValueError("real_data.sample_dir must be configured when use_sample_data is true.")
            input_dir = real_cfg.sample_dir
        else:
            if not real_cfg.raw_dir:
                raise ValueError("real_data.raw_dir must be configured when use_sample_data is false.")
            input_dir = real_cfg.raw_dir
        if not Path(input_dir).exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        return Path(input_dir)

    def _enrich_notes(
        self,
        *,
        input_dir: Path,
        formatted_dir: Path,
        log_csv_path: Path,
        sql_path: Path,
    ) -> EnrichmentSummary:
        real_cfg = self.preprocess_cfg.real_data
        eds_ids, json_files = collect_eds_ids(str(input_dir))
        self.logger.info("Collected %s JSON files for enrichment", len(json_files))

        if not (real_cfg.oracle.enabled or real_cfg.oracle.stub_mode):
            raise RuntimeError(
                "Oracle enrichment is mandatory for DRG preprocessing. Set preprocess.real_data.oracle.enabled=true."
            )

        if real_cfg.oracle.stub_mode:
            self.logger.warning(
                "Oracle stub mode enabled. Using synthetic enrichment data; install cx_Oracle/python-oracledb for production runs."
            )
            oracle_df = self._build_stub_oracle_df(json_files, real_cfg.oracle.stub_data_path)
        else:
            oracle_password = self._resolve_oracle_password()
            oracle_user = real_cfg.oracle.user
            oracle_dsn = real_cfg.oracle.dsn or real_cfg.oracle.network_alias

            if not oracle_user or not oracle_dsn:
                raise ValueError(
                    "Oracle configuration incomplete (expected user and dsn/network_alias to be provided via config or environment)."
                )

            if real_cfg.oracle.tns_admin:
                os.environ.setdefault("TNS_ADMIN", str(real_cfg.oracle.tns_admin))

            if real_cfg.oracle.instant_client_dir:
                os.environ.setdefault("ORACLE_HOME", str(real_cfg.oracle.instant_client_dir))

            oracle_df = self.oracle_fetcher(
                [item[2] for item in json_files],
                oracle_dsn,
                oracle_user,
                oracle_password,
                batch_size=real_cfg.oracle.batch_size,
                debug_print_sql=real_cfg.oracle.debug_print_sql,
                debug_sql_path=str(sql_path)
                if real_cfg.oracle.debug_sql_dir or real_cfg.oracle.debug_print_sql
                else None,
                instant_client_dir=str(real_cfg.oracle.instant_client_dir) if real_cfg.oracle.instant_client_dir else None,
            )

        meta_df = pd.DataFrame()
        if real_cfg.metadata_csv:
            meta_df = self.metadata_loader(str(real_cfg.metadata_csv))

        formatted_dir.mkdir(parents=True, exist_ok=True)
        update_json_files(json_files, oracle_df, meta_df, str(formatted_dir), str(log_csv_path))

        formatted_files = len(list(_iter_json_files(formatted_dir)))
        quarantined_dir = formatted_dir / "no_drg_eds_files"
        quarantined_files = len(list(quarantined_dir.glob("*.json"))) if quarantined_dir.exists() else 0

        return EnrichmentSummary(
            input_dir=input_dir,
            formatted_dir=formatted_dir,
            log_csv=log_csv_path if log_csv_path.exists() else None,
            quarantined_dir=quarantined_dir,
            total_raw_files=len(json_files),
            formatted_files=formatted_files,
            quarantined_files=quarantined_files,
        )

    def _build_stub_oracle_df(
        self,
        json_files: Sequence[tuple[str, str, str, str]],
        stub_data_path: Optional[Path],
    ) -> pd.DataFrame:
        if stub_data_path:
            stub_path = Path(stub_data_path)
            if not stub_path.exists():
                raise FileNotFoundError(f"Oracle stub data not found: {stub_path}")
            if stub_path.suffix.lower() == ".csv":
                df = pd.read_csv(stub_path)
            elif stub_path.suffix.lower() in {".jsonl", ".json"}:
                df = pd.read_json(stub_path, lines=stub_path.suffix.lower() == ".jsonl")
            else:
                raise ValueError(
                    f"Unsupported oracle.stub_data_path format: {stub_path.suffix}. Use .csv, .json, or .jsonl"
                )
            self.logger.info("Loaded Oracle stub data from %s with %s rows", stub_path, len(df))
            df.attrs["stub_mode"] = True
            return df

        rows = []
        for _, _, eds_id, _ in json_files:
            try:
                eds_value = int(str(eds_id))
            except Exception:
                eds_value = 0
            rows.append(
                {
                    "NOPTN": None,
                    "EDS": eds_value,
                    "DRG_OPA_CODE": "F62A",
                    "DRG_OPA_LIB": "Stub DRG",
                    "SERMED_ID_LST": "",
                    "SERMED_MNM_LST": "",
                    "DT_DEB_SEJ": pd.Timestamp("2022-01-01"),
                    "DT_FIN_SEJ": pd.Timestamp("2022-01-07"),
                    "AGE_ANNEE": 65,
                    "PRE_DRG_LIST": "F62A|E77B",
                    "PRE_DRG_DATE_LIST": "2022-01-01|2022-01-03",
                }
            )

        df = pd.DataFrame(rows)
        df.attrs["stub_mode"] = True
        self.logger.info(
            "Generated %s synthetic Oracle rows for stub mode (unique EDS=%s)",
            len(df),
            df["EDS"].nunique() if not df.empty else 0,
        )
        return df

    def _resolve_oracle_password(self) -> str:
        real_cfg = self.preprocess_cfg.real_data
        if real_cfg.oracle.password:
            return real_cfg.oracle.password
        if real_cfg.oracle.password_env:
            value = os.environ.get(real_cfg.oracle.password_env)
            if value:
                return value
        raise ValueError(
            "Oracle password not supplied. Define preprocess.real_data.oracle.password or set the "
            f"{real_cfg.oracle.password_env or 'ORACLE_PASSWORD'} environment variable."
        )
