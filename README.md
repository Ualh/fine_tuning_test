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

Override any default with `KEY=VALUE` pairs. Example:

```
run_pipeline.bat finetune-sft EPOCHS=2 LR=1.5e-5 OUTPUT_DIR=outputs/my_experiment
```

## Configuration

All tunables live in `config.yaml`:

- `paths`: prepared/output/log directories, Hugging Face cache, and vLLM models mount (`\\pc-27327\D\LLM`).
- `preprocess`: dataset, sample size, language filters, train/val split, sequence packing.
- `train`: LoRA hyper-parameters, scheduler, precision flags, logging cadence, resume options.
- `export`: merged checkpoint behaviour.
- `eval`: prompt suite plus generation parameters.
- `serve`: vLLM host/port/max context/served model name.
- `logging`: console/file log levels and tqdm refresh cadence.

CLI options mirror these fields so you can override values without editing the YAML.

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

```powershell
run_pipeline.bat build             # Build CUDA/PyTorch image
run_pipeline.bat up                # Start training container (reuse across stages)
run_pipeline.bat preprocess-sft    # Prepare 2k EN/FR examples
run_pipeline.bat finetune-sft      # Run LoRA training (resumable)
run_pipeline.bat export-merged     # Merge adapters into base model
run_pipeline.bat eval-sft          # Lightweight evaluation checkpoints
run_pipeline.bat serve-vllm        # Launch vLLM on port 8080
```

Each stage writes DEBUG logs to `logs/log_vXX_.../stage/run.log` while emitting succinct INFO messages
plus a single tqdm progress bar on the console.

## Resumability & logging

- Pass `RESUME_FROM=path/to/checkpoint` (or use `--resume-from` in the CLI) to continue a paused stage.
- `logs/latest.txt` always points to the most recent stage directory.
- Each stage now emits three log artefacts: `run.log` (DEBUG + rich tracebacks with locals),
  `console.log` (stdout/stderr with ANSI sequences stripped), and `container.log` (tail of
  `docker logs --tail 2000` for the training container captured by `run_pipeline.bat`).
- Metadata JSON files (`metadata.json`, `metrics.json`) summarise seeds, sizes, metrics, and timings.
- Console output stays tidy; dive into `run.log` for full stack traces and inspect `console.log`
  when you need the plain-text CLI output that appeared on screen.
- Regression coverage: `pytest tests/test_failure_logging.py` ensures failures persist rich traceback
  context in both log files.

## vLLM serving

- `vllm-server` reuses the shared `\\pc-27327\D\LLM` mount. Copy/symlink merged models into that share.
- Change the compose command or `serve` section in `config.yaml` if you want a different directory/port.
- Smoke-test via the bundled helper or any OpenAI-compatible client:

```python
from src.serve.vllm_client import VLLMClient

client = VLLMClient(endpoint="http://localhost:8080", model="Qwen2.5-0.5B-SFT")
if client.healthcheck():
    print(client.generate("Give me two bullet tips for managing VRAM."))
```

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

