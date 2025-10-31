"""Helpers to derive canonical run names for outputs, logs, and metadata."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Set

from .config import NamingConfig, PipelineConfig
from .errors import PipelineConfigError


_RUN_TOKEN_PATTERN = "run"
_ALLOWED_CHARS_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class RunNameResult:
    """Carry the computed run name and useful components."""

    name: str
    prefix: str
    index: int
    separator: str
    model_slug: str
    dataset_slug: str
    size_slug: str
    legacy: bool = False

    def to_env(self) -> dict[str, str]:
        """Return environment variables representing this run."""

        env = {
            "RUN_NAME": self.name,
            "RUN_NAME_PREFIX": self.prefix,
            "RUN_NUMBER": str(self.index),
            "RUN_MODEL_SLUG": self.model_slug,
            "RUN_DATASET_SLUG": self.dataset_slug,
            "RUN_SIZE_SLUG": self.size_slug,
            "RUN_SEPARATOR": self.separator,
            "RUN_LEGACY": "1" if self.legacy else "0",
        }
        return env


class LegacyNamingRequested(PipelineConfigError):
    """Raised when legacy naming is requested but disallowed."""


def build_run_name(
    cfg: PipelineConfig,
    *,
    outputs_root: Optional[Path] = None,
    logs_root: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> RunNameResult:
    """Return the canonical run name for the given configuration.

    The function honours runtime overrides from environment variables and the
    `naming` config block. It inspects existing runs to select the next
    available run counter (`runX`).
    """

    env = env or os.environ
    naming_cfg = cfg.naming
    outputs_root = outputs_root or cfg.paths.outputs_dir
    logs_root = logs_root or cfg.paths.logs_dir
    outputs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    use_legacy = _env_flag(env, "USE_LEGACY_NAMING") or naming_cfg.use_legacy
    if use_legacy and not naming_cfg.legacy_allowed:
        raise LegacyNamingRequested("Legacy naming requested but disallowed by configuration")

    legacy_name_override = env.get("LEGACY_RUN_NAME")
    if use_legacy:
        legacy_name = legacy_name_override or _legacy_run_slug(cfg)
        return RunNameResult(
            name=legacy_name,
            prefix=legacy_name,
            index=-1,
            separator="-",
            model_slug=_legacy_model_slug(cfg),
            dataset_slug=_legacy_dataset_slug(cfg),
            size_slug=_legacy_size_slug(cfg),
            legacy=True,
        )

    explicit_name = env.get("RUN_NAME") or env.get("FORCE_RUN_NAME")
    separator = naming_cfg.separator
    if explicit_name:
        explicit_name = explicit_name.strip()
        pattern = _compiled_run_pattern(separator)
        match = pattern.match(explicit_name)
        if not match:
            raise PipelineConfigError(
                f"Explicit run name '{explicit_name}' does not match '<prefix>{separator}run<index>'"
            )
        index = int(match.group("index"))
        prefix = match.group("prefix")
        components = _derive_components(cfg, naming_cfg)
        return RunNameResult(
            name=explicit_name,
            prefix=prefix,
            index=index,
            separator=separator,
            model_slug=components.model_slug,
            dataset_slug=components.dataset_slug,
            size_slug=components.size_slug,
            legacy=False,
        )

    components = _derive_components(cfg, naming_cfg)

    full_prefix = _join_non_empty(
        separator,
        [components.run_prefix, components.model_slug, components.dataset_slug, components.size_slug],
    )
    model_prefix = _join_non_empty(separator, [components.run_prefix, components.model_slug])
    dataset_prefix = _join_non_empty(separator, [components.run_prefix, components.model_slug, components.dataset_slug])

    existing_indices = _collect_indices(
        roots=(outputs_root, logs_root),
        separator=separator,
        scope=naming_cfg.run_counter_scope,
        full_prefix=full_prefix,
        dataset_prefix=dataset_prefix,
        model_prefix=model_prefix,
    )

    forced_index_raw = env.get("FORCE_RUN_INDEX")
    index: Optional[int] = None
    if forced_index_raw:
        try:
            forced_index = int(forced_index_raw)
        except ValueError as exc:  # pragma: no cover - defensive
            raise PipelineConfigError(f"FORCE_RUN_INDEX must be an integer, got '{forced_index_raw}'") from exc
        if forced_index <= 0:
            raise PipelineConfigError("FORCE_RUN_INDEX must be positive")
        if forced_index in existing_indices:
            raise PipelineConfigError(
                f"Run index {forced_index} already exists for prefix '{full_prefix}'. Use a different index."
            )
        index = forced_index

    counter = 1
    safety_limit = 10000
    while index is None and counter <= safety_limit:
        if counter not in existing_indices:
            index = counter
            break
        counter += 1

    if index is None:
        # Fall back to timestamp suffix when too many runs exist.
        index = _timestamp_fallback(existing_indices)

    prefix_max_len = naming_cfg.max_name_length - len(f"{separator}{_RUN_TOKEN_PATTERN}{index}")
    if prefix_max_len <= 0:
        raise PipelineConfigError("naming.max_name_length too small to accommodate run index")
    truncated_prefix = _truncate_prefix(full_prefix, prefix_max_len)

    if not truncated_prefix:
        truncated_prefix = "run"

    run_name = f"{truncated_prefix}{separator}{_RUN_TOKEN_PATTERN}{index}"
    # Ensure uniqueness even if truncation caused collision
    while _run_exists(run_name, (outputs_root, logs_root)):
        if forced_index_raw:
            raise PipelineConfigError(
                f"Run name '{run_name}' already exists and FORCE_RUN_INDEX prevented auto-increment."
            )
        index += 1
        prefix_max_len = naming_cfg.max_name_length - len(f"{separator}{_RUN_TOKEN_PATTERN}{index}")
        truncated_prefix = _truncate_prefix(full_prefix, prefix_max_len)
        if not truncated_prefix:
            truncated_prefix = "run"
        run_name = f"{truncated_prefix}{separator}{_RUN_TOKEN_PATTERN}{index}"

    return RunNameResult(
        name=run_name,
        prefix=truncated_prefix,
        index=index,
        separator=separator,
        model_slug=components.model_slug,
        dataset_slug=components.dataset_slug,
        size_slug=components.size_slug,
        legacy=False,
    )


@dataclass(frozen=True)
class _RunComponents:
    run_prefix: str
    model_slug: str
    dataset_slug: str
    size_slug: str


def _derive_components(cfg: PipelineConfig, naming_cfg: NamingConfig) -> _RunComponents:
    run_prefix = _sanitize(naming_cfg.run_prefix or "", fallback="run", naming_cfg=naming_cfg, allow_empty=True)

    model_raw = cfg.train.base_model.split("/")[-1] if cfg.train.base_model else "model"
    dataset_raw = cfg.preprocess.dataset_name.split("/")[-1] if cfg.preprocess.dataset_name else "dataset"

    if cfg.preprocess.sample_size in (None, 0, "full", "Full"):
        size_raw = "full"
    else:
        size_raw = f"n{cfg.preprocess.sample_size}"

    model_slug = _sanitize(model_raw, fallback="model", naming_cfg=naming_cfg)
    dataset_slug = _sanitize(dataset_raw, fallback="dataset", naming_cfg=naming_cfg)
    size_slug = _sanitize(size_raw, fallback="full", naming_cfg=naming_cfg)

    return _RunComponents(
        run_prefix=run_prefix,
        model_slug=model_slug,
        dataset_slug=dataset_slug,
        size_slug=size_slug,
    )


def _join_non_empty(separator: str, parts: Sequence[str]) -> str:
    values = [part for part in parts if part]
    return separator.join(values)


def _sanitize(
    value: str,
    *,
    fallback: str,
    naming_cfg: NamingConfig,
    allow_empty: bool = False,
) -> str:
    ascii_value = value.encode("ascii", "ignore").decode("ascii") if value else ""
    text = ascii_value.strip()
    if not text:
        text = fallback
    # Always lower-case to keep docker-compatible values.
    text = text.lower()
    slug = _ALLOWED_CHARS_RE.sub(naming_cfg.separator, text)
    slug = slug.strip(naming_cfg.separator)
    slug = re.sub(rf"{re.escape(naming_cfg.separator)}+", naming_cfg.separator, slug)
    if not slug and not allow_empty:
        slug = fallback
    slug = _truncate_prefix(slug, naming_cfg.max_segment_length)
    return slug


def _truncate_prefix(value: str, max_length: int) -> str:
    if not value:
        return value
    if len(value) <= max_length:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:4]
    slice_len = max(1, max_length - 5)
    candidate = value[:slice_len].rstrip("-")
    if not candidate:
        candidate = digest
    return f"{candidate}-{digest}"


def _collect_indices(
    *,
    roots: Iterable[Path],
    separator: str,
    scope: str,
    full_prefix: str,
    dataset_prefix: str,
    model_prefix: str,
) -> Set[int]:
    pattern = _compiled_run_pattern(separator)
    indices: Set[int] = set()
    for root in roots:
        if root is None or not root.exists():
            continue
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            match = pattern.match(entry.name)
            if not match:
                continue
            prefix = match.group("prefix")
            index = int(match.group("index"))
            if scope == "global":
                indices.add(index)
            elif scope == "model":
                if prefix.startswith(model_prefix) or model_prefix.startswith(prefix):
                    indices.add(index)
            else:  # model_dataset (default)
                if prefix.startswith(full_prefix) or full_prefix.startswith(prefix):
                    indices.add(index)
    return indices


def _run_exists(run_name: str, roots: Iterable[Path]) -> bool:
    for root in roots:
        if root is None:
            continue
        candidate = root / run_name
        if candidate.exists():
            return True
    return False


def _compiled_run_pattern(separator: str) -> re.Pattern[str]:
    sep = re.escape(separator)
    return re.compile(rf"^(?P<prefix>[a-z0-9{sep}]+){sep}{_RUN_TOKEN_PATTERN}(?P<index>\d+)$")


def _env_flag(env: Mapping[str, str], key: str) -> bool:
    value = env.get(key)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _timestamp_fallback(existing: Set[int]) -> int:
    base = 10000
    idx = base
    while idx in existing:
        idx += 1
    return idx


def _legacy_run_slug(cfg: PipelineConfig) -> str:
    dataset = _legacy_dataset_slug(cfg)
    sample = _legacy_size_slug(cfg)
    model = _legacy_model_slug(cfg)
    return f"{dataset}_{sample}_{model}"


def _legacy_dataset_slug(cfg: PipelineConfig) -> str:
    dataset_name = cfg.preprocess.dataset_name.split("/")[-1] if cfg.preprocess.dataset_name else "dataset"
    return dataset_name.replace("-", "_")


def _legacy_model_slug(cfg: PipelineConfig) -> str:
    value = cfg.train.base_model.split("/")[-1] if cfg.train.base_model else "model"
    return value.replace("-", "_")


def _legacy_size_slug(cfg: PipelineConfig) -> str:
    sample = cfg.preprocess.sample_size
    if sample in (None, 0, "full"):
        return "full"
    return f"n{sample}"
