## Context

This repo is a local, Docker-first LoRA fine-tuning pipeline for AI models with Typer CLI, per-stage run folders, awq converted model, a vLLM serving path, dozzle logs, and open WebUI integration. 

### Key directories and files

First, understand the repo structure and key files:
.
├── .github
├── .gitignore
├── .pytest_cache
├── .venv
├── Dockerfile                      # (CUDA/Torch image for sft runtime; base for training container)
├── Dockerfile.awq-runner           # (multi-stage image for isolated AWQ/llm-compressor runner)
├── README.md
├── TODO.md
├── config.yaml                     # (single source of truth for paths/serve settings; loaded by core.config)
├── config.smoke.yaml
├── debug_config.yaml
├── docker-compose.yml              # (compose services: sft, vllm-server, awq-runner sidecar, dozzle, open-webui)
├── files for CI / setup
├── open-webui                      # (Web UI files; exposes UI at 3000)
│   ├── cache
│   ├── uploads
│   ├── vector_db
│   ├── webui.db
│   └── ... (web UI related cache/uploads)
├── awq
│   ├── __init__.py
│   └── modules
│       ├── __init__.py
│       └── linear.py
├── docs
│   ├── 1.setup.md
│   ├── 2.preprocess_sft.md
│   ├── 3.finetune_sft.md
│   ├── 4.export_merged.md
│   ├── 5.awq-compression.md
│   ├── 6.eval_sft.md
│   ├── 7.serve_vllm.md
│   ├── 8.tensorboard.md
│   ├── guide.md
│   └── naming_spec.md
├── prepared
│   ├── awq_calibration.txt
│   ├── alpaca_2048
│   │   ├── metadata.json
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   ├── alpaca_2k_en
│   ├── alpaca_32
│   ├── alpaca_full
│   └── alpaca_full_en
├── outputs
│   ├── latest.txt                   # (pointer to last created outputs folder)
│   ├── autoif_qwen25_05b_lora
│   ├── run-qwen2-5-0-5b-alpaca-n2048-run7
│   └── run-qwen2-7b-alpaca-full-run1
│       ├── adapter/ (adapter files: .safetensors, tokenizer, chat_template, etc.)
│       ├── merged/ (merged model artifacts, metadata.json, model files)
│       └── trainer_state/ (checkpoints: checkpoint-*, runs/)
├── logs
│   ├── latest.txt                   # (points to newest run root; see RunManager logs/latest.txt)
│   ├── archives/
│   └── run-qwen2-7b-alpaca-full-run2/ (per-run logs and console/container logs)
├── sparse_logs
│   ├── oneshot_2025-11-03_16-20-32.log
│   └── ... (other sparse logs)
├── src
│   ├── __init__.py
│   ├── cli
│   │   ├── main.py        # Typer CLI (preprocess-sft, finetune-sft, export-merged, convert-awq, smoke-test, print-runtime) — (builds runtime metadata used by wrapper)
│   │   └── __init__.py
│   ├── core
│   │   ├── config.py      # (ConfigLoader: loads config.yaml; central source of paths and serve fields)
│   │   ├── logger.py      # (configure_logging, tee_std_streams; file INFO/DEBUG and run.log/console.log handling)
│   │   ├── run_manager.py # (RunManager: creates run dirs under logs/<run>/<stage> and manages latest.txt)
│   │   ├── runtime_probe.py# (probes runtime and builds runtime metadata used by wrapper/CLI)
│   │   ├── resume.py      # (ResumeManager: resolve --resume-from and checkpoint resolution)
│   │   ├── ssl.py         # (disable_ssl_verification helper used to match wrapper env)
│   │   └── tokenizer_utils.py
│   ├── data
│   │   └── preprocess.py  # (preprocess stage: builds train.jsonl/val.jsonl and metadata.json)
│   ├── training
│   │   ├── sft_trainer.py # (SFTTrainer runner: builds trainer with LoRA and writes adapter + trainer_state)
│   │   ├── lora_merge.py  # (merger: merges adapter into base and writes outputs/.../merged/)
│   │   ├── awq_runner.py / awq_converter.py # (AWQ runner CLI: in-container first, fallback local; writes metadata.json)
│   │   └── checkpointer.py
│   ├── eval
│   │   └── evaluator.py   # (Evaluator: optional perplexity and prompt-based checks; writes eval/metrics.json)
│   └── serve
│       └── vllm_client.py # (vLLM client used by smoke-test; reads SERVED_MODEL_PATH from runtime metadata)
└── tests
  ├── conftest.py
  ├── test_cli_paths.py            # (tests _build_runtime_metadata / path contract between wrapper and CLI)
  ├── test_runtime_probe.py        # (tests runtime_probe behavior)
  ├── test_pipeline_smoke.py       # (smoke test exercising end-to-end compose+stages)
  ├── test_sft_trainer_loader.py   # (trainer loading and accelerate/transformers compatibility)
  ├── test_hf_connectivity.py      # (HF hub connectivity tests; interacts with disable_ssl_verification)
  ├── test_config_loader.py
  ├── test_resume_manager.py
  ├── test_run_manager.py
  ├── test_tensorboard_logging.py
  └── awq
    └── test_awq_runner.py      # (tests AWQ conversion runner and metadata output)


## If need to debug a problem, otherwise skip

follow this disciplined debugging loop: reflect on 5–7 possible root causes, pick the 1–2 most likely, and add targeted logs/assertions to validate or falsify those hypotheses before changing code.

1. Quick status gathering
   - Inspect the latest run folder (pointer: `logs/latest.txt`) then open `run.log`, `console.log`, and `container.log` for timestamps and rich tracebacks.
   - Run `python -m src.cli.main print-runtime --format json --config config.yaml` (inside container) and verify `PREPROCESS_DIR`, `MERGED_DIR`, and `SERVED_MODEL_PATH` match expectations.

2. Common suspects (reflect on these 5–7 sources):
    - identify comon failure modes relevant to the error

3. Narrow to 1–2 likely causes
   - Prioritize the two that best explain the observed error and are easiest to validate

4. Add lightweight validations/logging (examples)
   - Insert temporary logger calls or assert statements near the failure site to print the resolved path, env vars, or presence of checkpoint files.
   - Re-run the failing command inside the container to stream logs live: use an attached session so tracebacks appear in your PowerShell window.

5. Validate and iterate
   - If validation falsifies a hypothesis, mark it off and test the next. Only after the key hypothesis is confirmed, implement a causal fix and add a guarded test or extra log to prevent regressions.

Guiding rule: never ship a code fix until you can demonstrate via logs that the failing state changes (before/after snippets), or you can reproduce locally with a minimized config (use `debug_config.yaml`).

## Always add clear docs

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
  - Prefer `logger` instances (passed into modules) over `print()` for library code. Log key runtime values at INFO and detailed internals at DEBUG. Mirror stdout to `console.log` via `core.logger

  ## ALWAYS:
  - ask when uncertain or when more context is needed