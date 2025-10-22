"""Configuration primitives for the fine-tuning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .errors import PipelineConfigError
from .io_utils import expand_path, parse_comma_separated


@dataclass
class PathsConfig:
    data_root: Path
    prepared_dir: Path
    outputs_dir: Path
    logs_dir: Path
    huggingface_cache: Path
    models_mount: Path
    run_metadata_file: Path


@dataclass
class PreprocessConfig:
    dataset_name: str
    sample_size: int
    filter_langs: List[str]
    test_size: float
    cutoff_len: int
    seed: int
    save_splits: bool
    max_workers: int
    pack_sequences: bool


@dataclass
class TrainConfig:
    base_model: str
    cutoff_len: int
    batch_size: int
    gradient_accumulation: int
    epochs: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool
    logging_steps: int
    eval_steps: int
    max_steps: Optional[int]
    resume_from: Optional[str]


@dataclass
class ExportConfig:
    include_adapter_weights: bool
    resume_from: Optional[str]


@dataclass
class EvalPrompt:
    instruction: str
    language: str
    max_sentences: Optional[int] = None


@dataclass
class EvalConfig:
    cutoff_len: int
    max_new_tokens: int
    temperature: float
    top_p: float
    prompts: List[EvalPrompt]
    resume_from: Optional[str]


@dataclass
class ServeConfig:
    host: str
    port: int
    max_model_len: int
    served_model_name: str
    resume_from: Optional[str]


@dataclass
class LoggingConfig:
    console_level: str
    file_level: str
    tqdm_refresh_rate: float


@dataclass
class PipelineConfig:
    paths: PathsConfig
    preprocess: PreprocessConfig
    train: TrainConfig
    export: ExportConfig
    eval: EvalConfig
    serve: ServeConfig
    logging: LoggingConfig


class ConfigLoader:
    """Load configuration from YAML and provide dataclass accessors."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise PipelineConfigError(f"Config file not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            try:
                self.raw: Dict[str, Any] = yaml.safe_load(handle) or {}
            except yaml.YAMLError as exc:  # pragma: no cover - YAML parse errors
                raise PipelineConfigError(f"Cannot parse config: {exc}") from exc
        self.project_root = self.config_path.parent

    def load(self) -> PipelineConfig:
        return PipelineConfig(
            paths=self._parse_paths(),
            preprocess=self._parse_preprocess(),
            train=self._parse_train(),
            export=self._parse_export(),
            eval=self._parse_eval(),
            serve=self._parse_serve(),
            logging=self._parse_logging(),
        )

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------
    def _parse_paths(self) -> PathsConfig:
        data = self.raw.get("paths", {})
        huggingface_cache = data.get("huggingface_cache")
        if not huggingface_cache:
            huggingface_cache = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE") or "${USERPROFILE}/.cache/huggingface"

        return PathsConfig(
            data_root=self._resolve_path(data.get("data_root", "data")),
            prepared_dir=self._resolve_path(data.get("prepared_dir", "prepared")),
            outputs_dir=self._resolve_path(data.get("outputs_dir", "outputs")),
            logs_dir=self._resolve_path(data.get("logs_dir", "logs")),
            huggingface_cache=self._resolve_path(huggingface_cache),
            models_mount=self._resolve_path(data.get("models_mount", "${USERPROFILE}/models")),
            run_metadata_file=self._resolve_path(data.get("run_metadata_file", "logs/latest.txt")),
        )

    def _parse_preprocess(self) -> PreprocessConfig:
        data = self.raw.get("preprocess", {})
        filter_langs = data.get("filter_langs", ["en"]) or []
        return PreprocessConfig(
            dataset_name=data.get("dataset_name", ""),
            sample_size=int(data.get("sample_size", 1000)),
            filter_langs=list(parse_comma_separated(filter_langs)),
            test_size=float(data.get("test_size", 0.1)),
            cutoff_len=int(data.get("cutoff_len", 2048)),
            seed=int(data.get("seed", 42)),
            save_splits=bool(self._to_bool(data.get("save_splits", True))),
            max_workers=int(data.get("max_workers", os.cpu_count() or 4)),
            pack_sequences=bool(self._to_bool(data.get("pack_sequences", True))),
        )

    def _parse_train(self) -> TrainConfig:
        data = self.raw.get("train", {})
        return TrainConfig(
            base_model=data.get("base_model", ""),
            cutoff_len=int(data.get("cutoff_len", 2048)),
            batch_size=int(data.get("batch_size", 1)),
            gradient_accumulation=int(data.get("gradient_accumulation", 1)),
            epochs=int(data.get("epochs", 1)),
            learning_rate=float(data.get("learning_rate", 2e-5)),
            min_learning_rate=float(data.get("min_learning_rate", 5e-6)),
            weight_decay=float(data.get("weight_decay", 0.0)),
            warmup_ratio=float(data.get("warmup_ratio", 0.0)),
            lr_scheduler=str(data.get("lr_scheduler", "linear")),
            lora_r=int(data.get("lora_r", 8)),
            lora_alpha=int(data.get("lora_alpha", 16)),
            lora_dropout=float(data.get("lora_dropout", 0.05)),
            lora_target_modules=list(parse_comma_separated(data.get("lora_target_modules", ["q_proj", "v_proj"]))),
            gradient_checkpointing=bool(self._to_bool(data.get("gradient_checkpointing", True))),
            bf16=bool(self._to_bool(data.get("bf16", True))),
            fp16=bool(self._to_bool(data.get("fp16", True))),
            logging_steps=int(data.get("logging_steps", 20)),
            eval_steps=int(data.get("eval_steps", 100)),
            max_steps=self._to_optional_int(data.get("max_steps")),
            resume_from=self._to_optional_str(data.get("resume_from")),
        )

    def _parse_export(self) -> ExportConfig:
        data = self.raw.get("export", {})
        return ExportConfig(
            include_adapter_weights=bool(self._to_bool(data.get("include_adapter_weights", True))),
            resume_from=self._to_optional_str(data.get("resume_from")),
        )

    def _parse_eval(self) -> EvalConfig:
        data = self.raw.get("eval", {})
        prompt_entries = data.get("prompts", []) or []
        prompts = [
            EvalPrompt(
                instruction=item.get("instruction", ""),
                language=item.get("language", "en"),
                max_sentences=item.get("max_sentences"),
            )
            for item in prompt_entries
        ]
        return EvalConfig(
            cutoff_len=int(data.get("cutoff_len", 2048)),
            max_new_tokens=int(data.get("max_new_tokens", 256)),
            temperature=float(data.get("temperature", 0.7)),
            top_p=float(data.get("top_p", 0.9)),
            prompts=prompts,
            resume_from=self._to_optional_str(data.get("resume_from")),
        )

    def _parse_serve(self) -> ServeConfig:
        data = self.raw.get("serve", {})
        return ServeConfig(
            host=data.get("host", "0.0.0.0"),
            port=int(data.get("port", 8080)),
            max_model_len=int(data.get("max_model_len", 2048)),
            served_model_name=data.get("served_model_name", "model"),
            resume_from=self._to_optional_str(data.get("resume_from")),
        )

    def _parse_logging(self) -> LoggingConfig:
        data = self.raw.get("logging", {})
        return LoggingConfig(
            console_level=str(data.get("console_level", "INFO")),
            file_level=str(data.get("file_level", "DEBUG")),
            tqdm_refresh_rate=float(data.get("tqdm_refresh_rate", 1.0)),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, raw: str) -> Path:
        path = Path(os.path.expandvars(str(raw)))
        if not path.is_absolute():
            path = self.project_root / path
        return expand_path(path)

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        value_str = str(value).strip().lower()
        if value_str in {"1", "true", "yes", "y"}:
            return True
        if value_str in {"0", "false", "no", "n"}:
            return False
        raise PipelineConfigError(f"Cannot coerce value to bool: {value}")

    @staticmethod
    def _to_optional_int(value: Any) -> Optional[int]:
        return None if value in (None, "", "null") else int(value)

    @staticmethod
    def _to_optional_str(value: Any) -> Optional[str]:
        return None if value in (None, "", "null") else str(value)
