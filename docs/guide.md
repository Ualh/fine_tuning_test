# Fine-tuning Guide (Windows + Docker)

Short, up-to-date walkthrough for this repository. Assumes Windows with PowerShell and Docker Desktop.

## 1) Prereqs
- Windows 11/10, Docker Desktop (WSL2 engine enabled)
- Optional GPU: recent NVIDIA driver; Docker Desktop GPU support enabled
- HF token if the base model/dataset is gated

Quick GPU checks (optional):
```powershell
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## 2) Configure once
Create `.env` next to `docker-compose.yml`:
```
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX
```
Review `config.yaml` and adjust as needed. For full-epoch runs, ensure:
```yaml
train:
  max_steps: null
```

GPU vs CPU:
- Training (`sft`): keep `gpus: all` for GPU in `docker-compose.yml`. Set `train.bf16: true` (and optionally `fp16: true`). To force CPU, remove `gpus: all` and set `bf16: false`, `fp16: false`.
- Serving (`vllm-server`): `VLLM_TARGET_DEVICE: cuda` for GPU or `cpu` for CPU.

## 3) Build and start
```powershell
.\run_pipeline.bat build
.\run_pipeline.bat up
```

Optional shell inside the training container:
```powershell
.\run_pipeline.bat bash
```

## 4) Run the pipeline
All stages read from `config.yaml`. You can override most options via wrapper environment variables.

Preprocess (creates JSONL splits under `prepared/`):
```powershell
.\run_pipeline.bat preprocess-sft
```

Fine-tune (writes adapters and trainer state under `outputs/`):
```powershell
.\run_pipeline.bat finetune-sft
```

Export merged model (writes to `outputs/<run>/merged`):
```powershell
.\run_pipeline.bat export-merged
```

Evaluate (quick sanity metrics and prompt generations):
```powershell
.\run_pipeline.bat eval-sft
```

Quantize to AWQ (llm-compressor via awq-runner):
```powershell
.\run_pipeline.bat convert-awq
```

Serve with vLLM (mounts `./outputs` to `/models`; auto-selects `<run>/merged_awq` when available):
```powershell
.\run_pipeline.bat serve-vllm
```
By default this exposes http://localhost:8080 (configurable via `serve.host`/`serve.port`).

Smoke-test the endpoint:
```powershell
python -m src.cli.main smoke-test --prompt "Résume AC215 en trois points."
```

## 5) Useful helpers
Print resolved runtime (paths, compose project, served model path):
```powershell
python -m src.cli.main print-runtime --format json
```

Start/stop containers:
```powershell
.\run_pipeline.bat up
.\run_pipeline.bat down
```

TensorBoard dashboard:
```powershell
.\run_pipeline.bat tensorboard-up
```

## 6) Logs and naming
Logs are stored under `logs/<run-name>/<stage>/` and the latest run root is tracked in `logs/latest.txt`.
Run naming follows `<model>-<dataset>-<size>-runX` and is used for both `outputs/` and `logs/`.

Logging behaviour
- The CLI routes verbose and debug output to the per-run `run.log` file so you can inspect per-step metric snapshots, library INFO/DEBUG, and full tracebacks without cluttering the console.
- The interactive console defaults to WARNING+. To show more, set `logging.console_level` in `config.yaml` to `INFO` or `DEBUG`. `logging.file_level` controls the level for `run.log` (keep at `DEBUG`).

## 7) Tips
- Out-of-memory? Lower `train.batch_size`, raise `train.gradient_accumulation`, keep `train.gradient_checkpointing: true`.
- Precision hiccups? Turn off `fp16` first; keep `bf16` if on Ampere/Ada.
- SSL/proxy issues are already handled (verification disabled). Remove those envs later if not needed.