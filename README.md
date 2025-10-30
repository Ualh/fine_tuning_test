## Overview

This repository delivers a fully local workflow to fine-tune **Qwen/Qwen2.5-0.5B** on the AutoIF
instruction-following dataset with LoRA, evaluate the result, and serve it through **vLLM**. It follows
the spirit of `example_fine_tuning_qwen` while modernising the codebase with a modular `src/` layout,
Typer CLI, rich logging, resumable stages, and configuration-driven parameters.

Optimised for a single **GeForce RTX 5070 Ti (16 GB)** workstation:

- 2 000-sample EN/FR subset for quick iterations.
- LoRA (bf16/fp16-ready) with sequence packing for efficient VRAM use.
- Merged checkpoint ready for vLLM on port 8080.
- Everything runs locally: Docker, DVC cache, Hugging Face cache, model artefacts.

## Project structure

```
fine_tuning_test/
├── config.yaml            # Central configuration for paths, hyper-parameters, logging
├── docker-compose.yml     # Training container + vLLM server container
├── Dockerfile             # CUDA 12.1 image with PyTorch / Transformers / TRL stack
├── requirements.txt       # Python dependencies for local execution/tests
├── run_pipeline.bat       # Windows batch wrapper (build, preprocess, train, eval, serve)
├── src/                   # OOP modules: core/, data/, training/, eval/, serve/, cli/
├── tests/                 # Pytest suite covering orchestration utilities
└── README.md              # Project documentation
```

Runtime directories (ignored by git):

- `prepared/` – chat-formatted splits.
- `outputs/` – adapters, trainer state, merged model, evaluation reports.
- `logs/` – auto-incremented `log_vXX_YYYY-mm-dd_HH-MM-SS` folders with `run.log`, summaries, metadata.
- `data/` – optional raw cache if you export datasets locally.

## Quick start

1. Clone/open the repo in VS Code (PowerShell, GPU enabled).
2. Review `config.yaml` and adjust paths or hyper-parameters.
3. Build the Docker image: `run_pipeline.bat build`
4. Start the training container: `run_pipeline.bat up`
5. Execute stages:
   - `run_pipeline.bat preprocess-sft`
   - `run_pipeline.bat finetune-sft`
   - `run_pipeline.bat export-merged`
   - `run_pipeline.bat eval-sft`
   - `run_pipeline.bat serve-vllm`
6. Smoke-test the served model:
   - `python -m src.cli.main smoke-test --prompt "Résume AC215 en trois points."`

Each stage now reads dataset/model/paths from `config.yaml`; edit the YAML once and rerun the desired commands.
Inspect the resolved paths and docker-compose project name anytime (Docker-first):

```powershell
run_pipeline.bat bash
# Inside container (optional):
python3 -m src.cli.main print-runtime --format json
```

## Configuration

All tunables live in `config.yaml`:

- `paths`: prepared/output/log directories, Hugging Face cache, and vLLM models mount (`\\pc-27327\D\LLM`).
- `preprocess`: dataset, sample size, language filters, train/val split, sequence packing.
- `train`: LoRA hyper-parameters, scheduler, precision flags, logging cadence, resume options.
- `export`: merged checkpoint behaviour.
- `eval`: prompt suite plus generation parameters.
- `serve`: vLLM host/port/max context, served model name, and `served_model_relpath` (relative to `outputs/`).
- `logging`: console/file log levels and tqdm refresh cadence.

CLI options mirror these fields so you can override values without editing the YAML.

## AWQ conversion (llm-compressor)

This project ships an isolated `awq-runner` image and a small runner script that calls `llmcompressor`'s oneshot pathway to produce AWQ-compressed model outputs.

- Supported llm-compressor version used for verification: 0.8.1 (installed in the `awq-runner` image).
- The runner entrypoint is `python3 -m src.training.awq_runner`; it prefers a system `llmcompressor` CLI when available but falls back to `python -m llmcompressor.oneshot` if not.
- Minimal AWQ-related `config.yaml` keys (under `awq`): `enabled`, `gpu_enabled`, `method`, `scheme`, `num_calibration_samples`, `calib_text_file`, `use_smoothquant`, `smoothquant_strength`, `max_seq_length`, `output_suffix`, `ignore`.

Quick run (Windows PowerShell):

```powershell
run_pipeline.bat convert-awq CONFIG=debug_config.yaml
run_pipeline.bat convert-awq CONFIG=config.yaml
```

After a successful run inspect:

- `outputs/<run>/merged_awq/metadata.json` — should show `"returncode": 0` on success.
- `logs/<run>/convert-awq/container.log` — runner and llm-compressor logs for debugging.

Notes:

- The runner requires the merged model directory to exist (`outputs/.../merged`). If you see "Merged model path not found", run `run_pipeline.bat export-merged` first.
- If `llmcompressor` console script is missing in the container, the runner will use the Python entrypoint which is supported in tested setups.


## Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
```

Running `pytest` after installing dependencies validates the orchestration utilities before launching
resource-intensive stages.

## Preflight checks (HF access + orchestration smoke test)

Before you start the containers, run the new preflight tests to ensure your Hugging Face token works with
SSL verification disabled and that the Typer CLI wiring is healthy:

```powershell
pytest tests/test_hf_connectivity.py
pytest tests/test_pipeline_smoke.py
```

Tips:

- Place your token in `.env` (keys: `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`) or export it in the shell before
  running the tests. The HF tests will be auto-skipped if the token is missing.
- `tests/test_hf_connectivity.py` performs two fast calls: metadata retrieval via `huggingface_hub.HfApi`
  and a tokenizer download via `AutoTokenizer.from_pretrained`. Both calls rely on
  `src.core.ssl.disable_ssl_verification()` to bypass enterprise TLS interception.
- `tests/test_pipeline_smoke.py` monkeypatches heavy dependencies (datasets, TRL, Transformers) so the CLI
  runs completely offline while still creating the expected run directories and metadata files.

### SSL toggles (host + containers)

The wrapper (`run_pipeline.bat`), Docker image, and python utilities now set these environment variables to
force-disable SSL verification everywhere:

- `CURL_CA_BUNDLE=`, `REQUESTS_CA_BUNDLE=`, `SSL_CERT_FILE=`
- `PYTHONHTTPSVERIFY=0`, `HF_HUB_DISABLE_SSL_VERIFY=1`, `HF_HUB_ENABLE_XET=0`
- `GIT_SSL_NO_VERIFY=1`

Remove or override these variables if you later restore a trusted certificate chain.

## Docker pipeline

These wrappers always consult `config.yaml` (or the file passed with `CONFIG=...`) and reuse the derived
paths and names, so switching models/data only requires editing the YAML once.

```powershell
run_pipeline.bat build             # Build CUDA/PyTorch image
run_pipeline.bat up                # Start training container (reuse across stages)
run_pipeline.bat preprocess-sft    # Prepare N examples (see preprocess.sample_size)
run_pipeline.bat finetune-sft      # Run LoRA training (resumable)
run_pipeline.bat export-merged     # Merge adapters into base model
run_pipeline.bat eval-sft          # Lightweight evaluation checkpoints
run_pipeline.bat serve-vllm        # Launch vLLM on port 8080
run_pipeline.bat clean             # Stop project, remove orphans, prune unused resources
```

Each stage writes DEBUG logs to `logs/log_vXX_.../stage/run.log` while emitting succinct INFO messages
plus a single tqdm progress bar on the console.

### Extras: monitoring and UI

- Dozzle (container logs UI): http://localhost:9999
- Open WebUI (chat UI over OpenAI API): http://localhost:3000
- TensorBoard (training metrics): http://localhost:6006 — see `docs/8.tensorboard.md`

When you run `run_pipeline.bat serve-vllm`, both services are started alongside `vllm-server`.
Open WebUI is preconfigured to talk to the vLLM OpenAI endpoint inside the compose network.

## Resumability & logging

- Pass `RESUME_FROM=path/to/checkpoint` (or use `--resume-from` in the CLI) to continue a paused stage.
- `logs/latest.txt` always points to the most recent stage directory.
- Each stage now emits three log artefacts: `run.log` (DEBUG + rich tracebacks with locals),
  `console.log` (stdout/stderr with ANSI sequences stripped), and `container.log` (tail of
  `docker logs --tail 2000` for the training container captured by `run_pipeline.bat`).
- Metadata JSON files (`metadata.json`, `metrics.json`) summarise seeds, sizes, metrics, and timings.
- Output directories include the dataset sample size: `outputs/<dataset>_<nX|full>_<model>/...`.
- Console output stays tidy; dive into `run.log` for full stack traces and inspect `console.log`
  when you need the plain-text CLI output that appeared on screen.
  
Logging behaviour (what you see vs what is recorded)

- By default the interactive console is intentionally quiet and only shows WARNING and higher.
  This keeps the terminal readable during long runs. All DEBUG/INFO messages (including per-step
  and per-eval metric snapshots) are written to the stage `run.log` file under the run folder
  (e.g. `logs/log_v01_.../train/run.log`).
- Warnings, deprecation notes, and library INFO/DEBUG entries are also captured in `run.log` so
  you can inspect them later without cluttering the console.
- Progress bars are labelled by stage: `TRAIN` for fine-tuning, `EVAL` for evaluation, and
  `PREPROC-*` / `MAP` during preprocessing so it is clear which task is currently running.
- To change the behaviour, edit `config.yaml` under the `logging` section:
  - `console_level`: controls the interactive console level (e.g. `WARNING`, `INFO`, `DEBUG`).
  - `file_level`: controls what's written to `run.log` (typically `DEBUG`).
  After changing the config, rerun the desired stage; the CLI will apply the new levels when it creates the run dir.
- Regression coverage: `pytest tests/test_failure_logging.py` ensures failures persist rich traceback
  context in both log files.

## vLLM serving

- Point `serve.served_model_relpath` in `config.yaml` to the merged folder under `outputs/`; the helper
  expands it to `/models/...` inside the container. If the config value is falsy or explicitly set to the string `"none"` (case-insensitive), the CLI will default to the merged output directory for the
  project (the same directory produced by `export-merged`). When you provide a custom relpath it will be   normalized (backslashes -> forward slashes and leading/trailing slashes removed) before use. The runtime probe emits two variables you can inspect or inject into compose:
  - `SERVED_MODEL_RELPATH`: the chosen relative path under the project (normalized)
  - `SERVED_MODEL_PATH`: the container path, i.e. `/models/<SERVED_MODEL_RELPATH>`

  Note: `SERVED_MODEL_NAME` is preserved exactly as supplied in `config.yaml`. If you omit an explicit
  `served_model_name` you will need to provide the model name manually when running the smoke-test or
  when pointing clients at the vLLM server.
- `run_pipeline.bat serve-vllm` injects `SERVED_MODEL_PATH`, `SERVED_MODEL_NAME`, and
  `SERVED_MODEL_MAX_LEN` before calling `docker compose` with the dynamic project name.
- The `vllm-server` service still mounts `\\pc-27327\D\LLM`; copy or symlink your merged model there if
  you want to serve from the shared drive instead of local outputs.
- Smoke-test via the bundled helper or any OpenAI-compatible client:

```python
from src.serve.vllm_client import VLLMClient

client = VLLMClient(endpoint="http://localhost:8080", model="Qwen2.5-0.5B-SFT")
if client.healthcheck():
    print(client.generate("Give me two bullet tips for managing VRAM."))
```

Open WebUI tips:

- Visit http://localhost:3000 and create the first admin user on first launch.
- It auto-detects the OpenAI provider using the internal URL; alternatively set the OpenAI Base URL to
  `http://localhost:8080/v1` in Settings and use any placeholder API key if prompted.
  Then select the served model name (e.g., `Qwen2.5-0.5B-SFT`) when chatting.

## DVC (local only)

Initialise DVC in the project root the first time:

```powershell
dvc init
dvc config cache.type "reflink,symlink,hardlink,copy"
dvc add prepared
dvc add outputs
git add .dvc config.yaml prepared.dvc outputs.dvc
```

This keeps lineage for prepared splits and model artefacts without requiring a remote. Add one later via
`dvc remote add` when collaboration or backup is needed.

## Troubleshooting

- **Docker cannot access `\\pc-27327\D\LLM`**: map the share to a drive letter (e.g. `Z:`) and update
  `config.yaml` and `docker-compose.yml` accordingly.
- **CUDA OOM**: reduce `BATCH`, increase `GRAD_ACCUM`, or lower `CUTOFF`.
- **Preprocess slow**: disable packing (`--pack-sequences false`) or lower `MAX_WORKERS` for CPU-bound
  systems.
- **WANDB requests**: the container defaults to `WANDB_MODE=offline`; remove/override if you want online
  tracking later.
- **Dozzle shows no containers**: ensure Docker Desktop is running Linux containers; the socket mount
  `/var/run/docker.sock` must be available to the Dozzle container. If constrained by corporate policy,
  skip Dozzle and use `docker compose logs -f` instead.
- **Local Typer import errors**: the wrapper computes runtime info inside Docker first. If you still see
  `ModuleNotFoundError: typer` on your host, run stages via `run_pipeline.bat` (do not call Python locally),
  or run `run_pipeline.bat bash` and execute `python3 -m src.cli.main ...` inside the container.

- **`. était inattendu.` during `run_pipeline.bat up`**: some CUDA base images print a Unicode banner
  (NVIDIA ASCII art + emoji) before the Typer command emits `KEY=VALUE` pairs. The wrapper now routes
  the probe through `python3 -m src.core.runtime_probe` inside the container so banner noise is dropped
  automatically. If the shell still complains, rerun with `CONFIG=debug_config.yaml` (which enables
  `logging.debug_pipeline: true`) to surface extra context and share the console trace when reporting
  the issue.

## Dataset reference

- **Dataset**: `Post-training-Data-Flywheel/AutoIF-instruct-61k-with-funcs`
- **Authors**: Diao et al., Oct 2024 (AutoIF library, GPT-4o-mini generations)
- **Default subset**: 2 000 examples, EN/FR only. Increase `SAMPLE_SIZE` for full-dataset training.

## Next steps

- Expand pytest coverage with integration tests that mock Hugging Face APIs.
- Introduce a `dvc.yaml` pipeline connecting preprocess → train → export for full reproducibility.
- Add GitHub Actions workflows for linting/tests once you move beyond local runs.
- Experiment with newer CUDA / PyTorch builds optimised for the RTX 5070 Ti (Blackwell).
- Re-enable WANDB or other observability stacks when you are ready to store metrics externally.

