# Run Naming Specification

This document tracks the canonical naming convention for all pipeline artifacts, logs, Docker entities, and runtime metadata. It will be refined alongside implementation.

## Naming Pattern

```
<model_name>-<dataset_name>-<dataset_size>-runX
```

- `model_name`: sanitized slug built from the configured base model identifier. Primary source is `config.training.model_name_or_path`, falling back to `runtime.metadata['SERVED_MODEL_NAME']` and finally to the basename of the path on disk. Examples: `qwen2-7b`, `mistral-7b`.
- `dataset_name`: slug extracted from the configured dataset identifier. Primary source is `config.data.dataset_name`. If unset, use the basename of the preprocess output directory (without `_full`/`_nX` suffix); fallback to `dataset`. Examples: `drg_letters`, `drg_letters-debug`.
- `dataset_size`: `full` when `sample_size` is `null`, `'full'`, or `"full"`. Otherwise `n{sample_size}` (e.g. `n2048`, `n32`).
- `runX`: incrementing counter (`run1`, `run2`, …) scoped by configuration (default: per model+dataset). The counter must avoid collisions by scanning existing directories.

## Normalization Rules

- Lowercase all segments.
- Strip leading/trailing whitespace.
- Replace spaces and runs of non `[a-z0-9]` characters with a single `-` separator.
- Collapse consecutive separators.
- Trim separators at start/end.
- Cap individual segments to 64 characters. If truncation occurs, append a four-character deterministic hash suffix.
- If a normalized segment becomes empty, substitute `model`, `dataset`, or `size` respectively.

## Run Counter Semantics

- Scope configured via `naming.run_counter_scope`:
  - `global`: increment across all runs.
  - `model`: increment per `model_name`.
  - `model_dataset` (default): increment per combination.
- To compute `runX`, list existing paths under `outputs/` (and optionally `logs/`) matching the prefix `<model>-<dataset>-<size>-run`. Extract numeric suffix, pick `max + 1`. Start at `run1` if none exist.
- Allow explicit override via CLI/env (e.g., `FORCE_RUN_INDEX`). Fail with descriptive error when requested index already exists unless `--force` is supplied.
- When no write access or collision persists past `run9999`, fallback to appending `-tsYYYYmmddHHMMSS`.

## Reserved Characters & Safety

- Valid in final slug: `a-z`, `0-9`, and `-`.
- Disallow double hyphens at boundaries of Docker names (`-run1` remains acceptable).
- When used for Docker image/container names, ensure the overall length <= 128 characters.
- For filesystem paths on Windows, guard against total path exceeding 240 characters. Prefer shorter prefixes when necessary.

## Backwards Compatibility

- Config option `naming.use_legacy` (default `false`) controls opt-out.
- Env override `USE_LEGACY_NAMING=1` restores previous behavior per run.
- Provide migration script for existing runs; script must support `--dry-run` and `--yes` flags and produce a summary report.
- Maintain `outputs/latest` and `logs/latest.txt` pointers for existing tooling.

## Runtime Metadata & Tooling Contract

- Exported metadata via `_build_runtime_metadata`:
  - `RUN_NAME`: canonical value.
  - `RUN_NUMBER`: numeric portion.
  - `RUN_NAME_PREFIX`: `<model>-<dataset>-<size>`.
  - `RUN_DIR_NAME`: same as `RUN_NAME`.
  - `SERVED_MODEL_NAME`: remains canonical model slug.
- `run_pipeline.bat` consumes `RUN_NAME` for container/image naming, logs subdirectories, and for referencing the run in user-facing output.

## Implementation Checklist

1. Add `NamingConfig` dataclass and config defaults.
2. Implement `build_run_name` helper with sanitization utilities.
3. Integrate helper with `RunManager` and pipeline entrypoints.
4. Update wrappers (`run_pipeline.bat`, `docker-compose.yml`) to rely on `RUN_NAME`.
5. Adjust outputs/logs directory creation and pointer files.
6. Provide migration tooling and documentation.
7. Extend tests and CI checks for the new convention.
8. Roll out via staged verification using `debug_config.yaml`.

## Examples

| Model                | Dataset      | Sample size | Scope           | Result                    |
|----------------------|--------------|-------------|-----------------|---------------------------|
| `Qwen2-7B`           | `drg_letters`| `full`      | `model_dataset` | `qwen2-7b-drg_letters-full-run1`|
| `mistral-7b`         | `drg_letters-debug`  | `2048`      | `model_dataset` | `mistral-7b-drg_letters-debug-n2048-run2`|
| `phi-3-medium`       | *(missing)*  | `32`        | `model`         | `phi-3-medium-dataset-n32-run5`|
| `meta/llama-3-8b`    | `drg_letters full`| `null`      | `global`        | `meta-llama-3-8b-drg_letters-full-run17`|

## Open Questions

- Whether to include additional metadata (e.g., config profile) in the slug.
- Interaction with multi-node or distributed runs that spawn auxiliary containers.
- Impact on external services (TensorBoard, vLLM) expecting specific paths.
