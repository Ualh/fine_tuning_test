## Quick orientation for AI coding assistants

This repo is a local, Docker-first LoRA fine-tuning pipeline for AI models with Typer CLI, per-stage run folders, and a vLLM serving path. Keep edits tight and align with these patterns.

- Big picture (what touches what)
  - `run_pipeline.bat` orchestrates Windows workflows end-to-end: builds image, probes runtime via `print-runtime`, runs stages inside the `sft` container, captures container logs into each run folder.
  - `src/cli/main.py` defines stage commands: `preprocess-sft`, `finetune-sft`, `export-merged`, `eval-sft`, `print-runtime`, `smoke-test`. Contract glue lives in `_build_runtime_metadata` (paths, compose name, served model vars).
  - Modules map 1:1 to stages: `core/` (config, logging, resume, run manager, runtime probe, SSL bypass), `data/preprocess.py`, `training/` (`sft_trainer.py`, `lora_merge.py`), `eval/evaluator.py`, `serve/vllm_client.py`.

- Reproducible environments
  - Docker: `Dockerfile` (CUDA 12.1, Torch 2.2.2) + `docker-compose.yml` (`sft`, `vllm-server`, Dozzle, Open WebUI). Use `run_pipeline.bat up` then exec: `run_pipeline.bat bash` or `docker compose exec sft bash -lc "python3 -m src.cli.main print-runtime --format json"` (interactive `-it` is implied by wrapper; use plain `docker exec -it <container> bash -lc "…"` if you prefer).
  - Local dev: `.venv` + `requirements.txt`; run `pytest` before heavy stages. HF token via env (`HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`). SSL is intentionally disabled (see `core/ssl.py` and wrapper env) to survive corporate MITM.

- Logging, error handling, and debugging
  - Use `core.logger.configure_logging` in stages: console at INFO (single, minimal progress bar from Trainer), files at DEBUG. Live, detailed traces go to `run.log`; stdout/stderr are mirrored to `console.log` via `tee_std_streams`; wrapper saves Docker logs to `container.log`.
  - Wrap critical blocks with `try/except` and call `log_exception_with_locals(logger, msg, exc)` for Rich tracebacks with locals (already used in CLI commands). Always call `finalize_logger` in `finally`.
  - Run directories are created by `core.run_manager.RunManager` under `logs/log_vXX_.../<stage>/`, with `latest.txt` pointing to the newest stage folder.

- Configuration and conventions
  - Single source of truth: `config.yaml` → `core.config.ConfigLoader` (dataclasses). Paths resolve relative to repo root; UNC mounts supported via `paths.models_mount`.
  - Directory naming: `_default_preprocess_dir()` → `prepared/<dataset>_<full|N>`, `_default_output_dir()` → `outputs/<dataset>_nN_<model>`. `sample_size: null|full` means “full dataset”; else tag as `n{size}`.
  - Resume semantics: use `--resume-from` (CLI) or env to point to a trainer checkpoint dir; resolution handled by `core.resume.ResumeManager`.

- Stage specifics and examples
  - Preprocess: loads HF dataset, optional language filter, train/val split, chat templating via `ensure_chat_template`; writes `train.jsonl`, `val.jsonl`, and `metadata.json`.
  - Finetune: `training/SFTTrainerRunner` patches Transformers/Accelerate for compat, builds `SFTTrainer` with LoRA (`peft.LoraConfig`), saves adapter under `outputs/.../adapter/` and `trainer_state/`.
  - Export: `training/LoraMerger` merges adapter into base, saves to `outputs/.../merged/` and writes `metadata.json`.
  - Eval: `eval/Evaluator` optional perplexity on a JSONL split and prompt-based checks (language via `langdetect`), writes `eval/metrics.json`.
  - Serve: `docker-compose` service `vllm-server` reads `SERVED_MODEL_PATH` from runtime metadata; smoke test with `python -m src.cli.main smoke-test`.

- Orchestration truths that bite
  - Wrapper ←→ CLI contract: if you change paths/slug rules, update both `main._build_runtime_metadata` and `run_pipeline.bat :load_runtime` (compose project name, container paths, served model fields) and fix tests.
  - Tests to watch: `tests/test_cli_paths.py`, `test_pipeline_smoke.py`, `test_runtime_probe.py`, `test_accelerate_compat.py`, `test_sft_trainer_loader.py`, `test_hf_connectivity.py`.

- External dependencies and network pragmatics
  - HF hub access uses `disable_ssl_verification()` in each stage to align with wrapper env. Re-enabling SSL may break tests; adjust both places if you do.
  - vLLM container exposes port 8080; Open WebUI at 3000; Dozzle at 9999. Served model name/len come from `serve.*` in `config.yaml`.

If anything above is unclear or you need more examples (e.g., adding a compose service, adjusting LoRA targets, or where to mock HF calls in tests), ping which area to expand and we’ll tighten this guide.

## Debug recipes

When a stage fails, follow this disciplined debugging loop: reflect on 5–7 possible root causes, pick the 1–2 most likely, and add targeted logs/assertions to validate or falsify those hypotheses before changing code.

1. Quick status gathering
   - Inspect the latest run folder (pointer: `logs/latest.txt`) then open `run.log`, `console.log`, and `container.log` for timestamps and rich tracebacks.
   - Run `python -m src.cli.main print-runtime --format json --config config.yaml` (locally or inside container) and verify `PREPROCESS_DIR`, `MERGED_DIR`, and `SERVED_MODEL_PATH` match expectations.

2. Common suspects (reflect on these 5–7 sources):
    - identify comon failure modes relevant to the error

3. Narrow to 1–2 likely causes
   - Prioritize the two that best explain the observed error and are easiest to validate (e.g., config mismatch + missing checkpoint).

4. Add lightweight validations/logging (examples)
   - Insert temporary logger calls or assert statements near the failure site to print the resolved path, env vars, or presence of checkpoint files.
   - Re-run the failing command inside the container to stream logs live: use an attached session so tracebacks appear in your PowerShell window.

5. Validate and iterate
   - If validation falsifies a hypothesis, mark it off and test the next. Only after the key hypothesis is confirmed, implement a causal fix and add a guarded test or extra log to prevent regressions.

Guiding rule: never ship a code fix until you can demonstrate via logs that the failing state changes (before/after snippets), or you can reproduce locally with a minimized config (use `debug_config.yaml`).

## Contributor style — comments & docstrings

Write docstrings and comments as if explaining the module to a new contributor who just opened the repo. Be concise, precise, and point readers where to look next.

- Module-level docstring (top of file)
  - 1 line: what this module does (purpose).
  - 1 short sentence: where it sits in the pipeline (e.g., "Stage: preprocessing — writes train.jsonl/val.jsonl to prepared/").
  - Optional: one-line example invocation or a cross-file pointer (e.g., "See: src/cli/main.py for the Typer command that calls this").

- Class docstring
  - 1 short sentence describing responsibility.
  - 1–2 bullets: inputs/outputs and important side effects (files written, env vars used).

- Public function/method docstring
  - First line: short summary.
  - Follow with 1–3 lines describing important behaviour and failure modes.
  - Include Args / Returns / Raises sections (2–4 lines each). Keep examples brief; prefer linking to tests that demonstrate usage.

- Inline comments
  - Use inline comments only to explain *why*, not *what*. If the code is unclear, refactor it or add a small helper with a clear name instead of a long inline comment.
  - Use TODO or FIXME comments and include the issue number, e.g. TODO 123 or FIXME 123, when leaving non-trivial work for later.

- Tone and formatting
  - Use plain English, imperative verbs, and short sentences. Prefer concrete file references (e.g., `logs/latest.txt`, `src/core/resume.py`).
  - Keep docstrings compact and actionable — a newcomer should know where to look next (tests, CLI entrypoint, or run logs).

- Logging and diagnostics
  - Prefer `logger` instances (passed into modules) over `print()` for library code. Log key runtime values at INFO and detailed internals at DEBUG. Mirror stdout to `console.log` via `core.logger.tee_std_streams` when running CLI stages.

