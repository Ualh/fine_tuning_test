## Overview

This repository delivers a fully local workflow to fine-tune Qwen models with LoRA, evaluate the result,
and serve it through **vLLM**. Two preprocessing/training modes are supported:

- **DRG classification (default)** – enrich raw hospital discharge letters via Oracle, build parquet
  splits, and train a sequence-classification head for DRG prediction (mirrors `example_code/finetune_drg.py`).
- **Hugging Face datasets (Alpaca, etc.)** – keep the original SFT flow for instruction tuning by setting
  `preprocess.mode: huggingface` and `preprocess.dataset_name` to any HF dataset.

Both modes share the same Typer CLI, Docker runtime, logging, and run management. The DRG path requires
Oracle credentials (see **Environment setup**) while the Hugging Face path remains ideal for smoke tests
or experimentation without production data access.

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
- `outputs/` – per-run folders (`outputs/<run-name>/adapter`, `merged`, `eval`, …). `outputs/latest.txt` points to the most recent run.
- `logs/` – per-run stage logs (`logs/<run-name>/<stage>/run.log`). `logs/latest.txt` points to the newest run root.
- `data/` – optional raw cache if you export datasets locally.

## Quick start

1. Clone/open the repo in VS Code (PowerShell, GPU enabled).
2. Create/adjust `.env` with the required secrets:
  - `HF_TOKEN` (optional, for gated models)
  - `ORACLE_USER`, `ORACLE_PASSWORD`, `ORACLE_DSN`, `ORACLE_NETWORK_ALIAS`, `ORACLE_TNS_PATH`
3. Review `config.yaml` and adjust paths or hyper-parameters (switch `preprocess.mode` as needed).
4. Build the Docker image: `run_pipeline.bat build`
5. Start the training container: `run_pipeline.bat up`
6. Execute stages:
  - `run_pipeline.bat preprocess-sft`
  - `run_pipeline.bat finetune-sft`
    - By default the pipeline will automatically chain: export-merged → convert-awq → eval-sft → serve-vllm using the same run directory. You can tune this in `orchestration.post_finetune`.
7. Smoke-test the served model:
   - `python -m src.cli.main smoke-test --prompt "Résume AC215 en trois points."`

Each stage now reads dataset/model/paths from `config.yaml`; edit the YAML once and rerun the desired commands. Standalone stages (export-merged, convert-awq, eval-sft, serve-vllm) reuse an existing run; they will refuse to create a fresh run without the required artefacts. Pass `--run-name`, or explicit directories (e.g., `--adapter-dir`, `--merged-dir`), or run `finetune-sft` first.
Inspect the resolved paths and docker-compose project name anytime (Docker-first):

```powershell
run_pipeline.bat bash
# Inside container (optional):
python3 -m src.cli.main print-runtime --format json
```

### Preprocessing modes

- `real_drg` (default): requires access to the hospital Oracle DW. `preprocess.real_data` controls the
  enrichment options (raw/sample directories, token filters, rare-label handling). All Oracle fields must
  be supplied either directly in the YAML or via the `.env` variables listed above.
- `huggingface`: disables Oracle enrichment and reverts to the original SFT flow. Set
  `preprocess.dataset_name` (e.g., `tatsu-lab/alpaca`) and `preprocess.sample_size` as desired. The CLI
  will import the legacy `DataPreprocessor` and the training stage will use `SFTTrainerRunner`.

Switching mode only requires editing `config.yaml` (or an override such as `debug_config.yaml`); downstream
commands automatically adapt to the chosen mode.

Sample configs ready for quick validation:
- `debug_config.yaml` – DRG letters stub-mode run (Oracle disabled, small splits).
- `config_debug_alpaca.yaml` – Hugging Face Alpaca sample run (tiny model, TRL SFT trainer).

`finetune-sft` mirrors the selected mode automatically. The DRG path now wires in the same ingredients as
`example_code/finetune_drg.py`: class-balanced focal or weighted cross-entropy, optional oversampling via a
`WeightedRandomSampler`, BitsAndBytes 4-bit loading (with graceful CPU fallback), manual tokenisation, and
extended metric logging (macro F1, abstention statistics, TensorBoard).

### Run naming & overrides

Every stage now resolves a canonical run slug following `<model>-<dataset>-<size>-runX` and stores all artefacts under `outputs/<run-name>/` and `logs/<run-name>/`. Use the helper to preview the slug without touching the filesystem:

```powershell
run_pipeline.bat run-name-preview
run_pipeline.bat run-name-preview RUN_INDEX=5
run_pipeline.bat run-name-preview FORCE_RUN_NAME=qwen2-7b-drg_letters-full-run42
```

`print-runtime` echoes the same values along with helper variables (`RUN_DIR_NAME`, `RUN_OUTPUTS_DIR`, `RUN_LOGS_DIR`, container paths, etc.). To override the next run, set any of the following before invoking the pipeline:

- `FORCE_RUN_NAME=<slug>` – use an explicit slug (must match `<prefix>-runX`).
- `FORCE_RUN_INDEX=<number>` or `RUN_INDEX=<number>` – reserve a specific counter.
- `LEGACY_RUN_NAME=<name>` + `USE_LEGACY_NAMING=1` – temporarily fall back to the legacy folder naming.

All overrides propagate to the runtime probe (`print-runtime`) and to downstream container commands automatically.

### Orchestration (post-finetune)

- `config.yaml` supports `orchestration.post_finetune` with a list of stage names to run after successful training. Default examples in this repo include:
  - `export-merged`, `convert-awq`, `eval-sft`, `serve-vllm`
- The wrapper executes these in order inside the same run folder so artefacts are chained correctly and no stray runs are created.

- The wrapper, CLI, and `docker-compose.yml` now default to the `sft` compose project (lowercase to satisfy Docker rules). Docker Desktop shows the stack as **sft** (displayed as “SFT” in docs/UI), so you no longer need to export `HOST_COMPOSE_PROJECT` manually for standard runs.
- To change the project name, set `HOST_COMPOSE_PROJECT=<name>` (or `COMPOSE_PROJECT_NAME`) before invoking the wrapper. The pipeline will adopt the override everywhere, including the runtime probe and downstream container calls, and still sanitize the value for Docker.
- Built images are tagged under the shared namespace `${SFT_IMAGE_NS:-sft}/…` so related artefacts sort together in Docker Desktop and other registries. Override the namespace or tag by exporting `SFT_IMAGE_NS` / `SFT_IMAGE_TAG` if you need custom naming.
- After adopting the new naming, feel free to prune any historical tags such as `app-sft` or `fine_tuning_test-sft` once no containers depend on them (`docker rmi <tag>`).

## Configuration

All tunables live in `config.yaml`:

- `paths`: prepared/output/log directories, Hugging Face cache, and vLLM models mount (`\\pc-27327\D\LLM`).
- `preprocess`: dataset, sample size, language filters, train/val split, sequence packing, and `mode`
  (`real_drg` for Oracle-backed classification, `huggingface` for datasets like Alpaca).
- `train`: LoRA hyper-parameters, scheduler, precision flags, logging cadence, resume options.
- `export`: merged checkpoint behaviour.
- `eval`: prompt suite plus generation parameters.
- `serve`: vLLM host/port/max context plus the default served name, AWQ preference (`prefer_awq`), and optional overrides (`served_model_relpath`, `model_name`).
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

- The runner requires the merged model directory to exist (`outputs/.../merged`). If you see "Merged model path not found", run `run_pipeline.bat export-merged` first. When run after `finetune-sft` with orchestration enabled, this step is executed automatically in the same run directory.
- If `llmcompressor` console script is missing in the container, the runner will use the Python entrypoint which is supported in tested setups.


## Environment setup

Create a `.env` file at the project root (or export the variables before running the pipeline):

```
HF_TOKEN=<optional huggingface token>
ORACLE_USER=<oracle username>
ORACLE_PASSWORD=<oracle password>
ORACLE_DSN=<host:port/service or EZConnect string>
ORACLE_NETWORK_ALIAS=<alias present in tnsnames.ora>
ORACLE_TNS_PATH=<directory containing tnsnames.ora>
```

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
run_pipeline.bat preprocess-sft    # Enrich + build DRG dataset (mode=real_drg) or pull HF splits (mode=huggingface)
run_pipeline.bat finetune-sft      # Train DRG classifier or run the original SFT LoRA loop
run_pipeline.bat export-merged     # Merge adapters into base model
run_pipeline.bat eval-sft          # Lightweight evaluation checkpoints
run_pipeline.bat serve-vllm        # Launch vLLM on port 8080
run_pipeline.bat clean             # Stop project, remove orphans, prune unused resources
```

Each stage writes DEBUG logs to `logs/<run-name>/<stage>/run.log` while emitting succinct INFO messages
plus a single tqdm progress bar on the console.

### Extras: monitoring and UI

- Dozzle (container logs UI): http://localhost:9999
- Open WebUI (chat UI over OpenAI API): http://localhost:3000
- TensorBoard (training metrics): http://localhost:6006 — see `docs/8.tensorboard.md`

When you run `run_pipeline.bat serve-vllm`, both services are started alongside `vllm-server`.
Open WebUI is preconfigured to talk to the vLLM OpenAI endpoint inside the compose network.

## Resumability & logging

- Pass `RESUME_FROM=path/to/checkpoint` (or use `--resume-from` in the CLI) to continue a paused stage.
- `logs/latest.txt` points to the most recent run root (`logs/<run-name>/`). `outputs/latest.txt` mirrors the same run under `outputs/`.
- Each stage emits three log artefacts inside `logs/<run-name>/<stage>/`: `run.log` (DEBUG + rich tracebacks with locals),
  `console.log` (stdout/stderr with ANSI sequences stripped), and `container.log` (tail of
  `docker logs --tail 2000` for the training container captured by `run_pipeline.bat`).
- Metadata JSON files (`metadata.json`, `metrics.json`) summarise seeds, sizes, metrics, and timings.
- Run directories already encode the dataset scope (e.g. `qwen2-7b-drg_letters-full-run3`), and subfolders retain the previous layout (`adapter`, `merged`, `eval`, etc.).
- Console output stays tidy; dive into `run.log` for full stack traces and inspect `console.log`
  when you need the plain-text CLI output that appeared on screen.
  
Logging behaviour (what you see vs what is recorded)

- By default the interactive console is intentionally quiet and only shows WARNING and higher.
  This keeps the terminal readable during long runs. All DEBUG/INFO messages (including per-step
  and per-eval metric snapshots) are written to the stage `run.log` file under the run folder
  (e.g. `logs/qwen2-7b-drg_letters-full-run1/train/run.log`).
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

- By default the runtime selects `<run>/merged_awq` when AWQ conversion succeeds. This behaviour is
  controlled by `serve.prefer_awq` (default `true`). When the directory is missing or the flag is `false`,
  the CLI falls back to `<run>/merged`.
- Override the directory by setting `serve.served_model_relpath` to a path relative to `outputs/` (for
  example `my-run/merged_awq`). Set it to `null` to keep the automatic selection. Use `serve.model_name`
  if you need the vLLM endpoint to announce a specific name while retaining a different default locally.
- The runtime probe emits:
  - `SERVED_MODEL_RELPATH`: relative to `outputs/` when possible (e.g. `run-x/merged_awq`).
  - `SERVED_MODEL_PATH`: `/models/<SERVED_MODEL_RELPATH>` when the path is under `outputs/`.
  - `SERVED_MODEL_SOURCE`: `awq`, `merged`, or `override` to simplify debugging.
- `run_pipeline.bat serve-vllm` exports these variables before calling `docker compose` with the resolved
  project name.
- The `vllm-server` service now mounts only `./outputs:/models`, mirroring the runtime expectation that
  served models live under `outputs/`.
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

- **Docker cannot access `\\pc-27327\Projets\DRG-Prediction`**: map the share to a drive letter (e.g. `Z:`) and update
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

