# Fine-tuning Guide: Qwen2.5-0.5B SFT on Windows with GPU + Docker

## 1) Prereqs: Windows + Docker + GPU

- Windows 11/10 with WSL2 and Docker Desktop
- Latest NVIDIA GPU driver (CUDA-capable, 535+)
- Docker Desktop: enable “Use the WSL 2 based engine” and WSL integration for your distro
- Verify GPU visibility on host:
```powershell
nvidia-smi
```
- Verify Docker sees the NVIDIA runtime:
```powershell
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

If both work, you’re ready for GPU inside containers.

## 2) Enable GPU in docker-compose

Edit docker-compose.yml:
- For training container `qwen25-05b`, uncomment/add GPU resources:
```yaml
services:
  qwen25-05b:
    # ...
    # gpus: all        # <= UNCOMMENT this line
    # Fine-tuning Guide (A → Z)

    Short, up-to-date walkthrough for this repository on Windows + Docker.

    ## 1) Prereqs
    - Windows 11/10, Docker Desktop (WSL2 engine enabled)
    - Optional GPU: recent NVIDIA driver; Docker Desktop GPU support enabled
    - HF token if the model/dataset is gated

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
    - Training (`sft`): `docker-compose.yml` should have `gpus: all` for GPU. Set `train.bf16: true` (and optionally `fp16: true`). To force CPU, remove `gpus: all` and set `bf16: false`, `fp16: false`.
    - Serving (`vllm-server`): `VLLM_TARGET_DEVICE: cuda` for GPU (default here) or `cpu` for CPU.

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
    All stages read from `config.yaml`. You can override most options via CLI flags if needed.

    Preprocess (creates JSONL splits under `prepared/`):
    ```powershell
    .\run_pipeline.bat preprocess-sft
    ```

    Fine-tune (writes adapters and trainer state under `outputs/`):
    ```powershell
    .\run_pipeline.bat finetune-sft
    ```

    Export merged model (writes to `outputs/.../merged`):
    ```powershell
    .\run_pipeline.bat export-merged
    ```

    Evaluate (quick sanity metrics and prompt generations):
    ```powershell
    .\run_pipeline.bat eval-sft
    ```

    ## 5) Serve with vLLM
    The server mounts `./outputs` to `/models` and uses `serve.served_model_relpath` to locate the merged model.
    ```powershell
    .\run_pipeline.bat serve-vllm
    ```
    By default this exposes http://localhost:8080 (configurable via `serve.host`/`serve.port`).

    Smoke-test the endpoint:
    ```powershell
    python -m src.cli.main smoke-test --prompt "Résume AC215 en trois points."
    ```

    ## 6) Useful helpers
    Print resolved runtime (paths, compose project, served model path):
    ```powershell
    python -m src.cli.main print-runtime --format json
    ```

    Start/stop containers:
    ```powershell
    .\run_pipeline.bat up
    .\run_pipeline.bat down
    ```

    Logs are under `logs/log_vXX_.../<stage>/` and the latest path is stored in `logs/latest.txt`.

    Logging behaviour

    - The CLI routes verbose and debug output to the per-run `run.log` file so you can inspect
      per-step metric snapshots, library INFO/DEBUG, and full tracebacks there without cluttering
      the interactive console.
    - The interactive console is intentionally set to show only WARNING+ by default. If you prefer
      more verbose terminal output, update the `logging.console_level` value in `config.yaml` to
      `INFO` or `DEBUG`. `logging.file_level` controls the granularity written to `run.log` (keep
      this at `DEBUG` to capture everything).


    ## 7) Tips
    - Out-of-memory? Lower `train.batch_size`, raise `train.gradient_accumulation`, keep `train.gradient_checkpointing: true`.
    - Precision hiccups? Turn off `fp16` first; keep `bf16` if on Ampere/Ada.
    - SSL/proxy issues are already handled (verification disabled). Remove those envs later if not needed.
  5) `.run_pipeline.bat finetune-sft`
  6) `.run_pipeline.bat export-merged`
  7) Copy merged to the share and `docker compose up -d vllm-server`
  8) Test the endpoint or use the CLI smoke-test

If you confirm whether you want GPU serving (vs CPU) and that you’re ready to remove the smoke cap, I can apply those small YAML/compose edits for you.