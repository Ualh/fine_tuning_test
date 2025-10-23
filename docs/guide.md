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
    # or use the devices reservation if you prefer:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - capabilities: [gpu]
```
- For `vllm-server`, keep `runtime: nvidia` and switch device to GPU if you want GPU serving:
```yaml
  vllm-server:
    runtime: nvidia
    # gpus: all        # optionally add
    environment:
      VLLM_TARGET_DEVICE: cuda  # change from 'cpu' to 'cuda' for GPU serving
```

Save the file.

## 3) Flip config to a “full run”

Open config.yaml and set:
- For full epochs, remove the smoke cap:
```yaml
train:
  # ...
  max_steps: null
```
- Keep `bf16: true` and `fp16: true` for GPU; on Ada/Ampere this is fine. If you hit precision issues, turn off `fp16` first and keep `bf16` on.

Tip: If you need to fit VRAM, reduce `batch_size` and increase `gradient_accumulation` proportionally. Gradient checkpointing is already enabled.

## 4) Bring up the training container and sanity-check CUDA

From the repo root:
```powershell
.\run_pipeline.bat up
```

GPU sanity inside the container:
```powershell
docker exec -it qwen25-05b bash -lc "nvidia-smi"
docker exec -it qwen25-05b bash -lc "python3 - << 'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
PY"
```
You should see your GPU name and `cuda_available: True`.

## 5) Preprocess SFT data

This will create chat-formatted splits in `prepared/` based on your config (2k EN/FR subset):
```powershell
.\run_pipeline.bat preprocess-sft
```
Outputs:
- Data at `prepared/alpaca_2k_en` (by default)
- Logs under `logs/log_vXX_.../preprocess/`

## 6) Fine-tune on GPU

Run with current defaults (Qwen2.5-0.5B, LoRA, packing, bf16/fp16):
```powershell
.\run_pipeline.bat finetune-sft
```

Notes:
- If you paused and want to resume, pass `RESUME_FROM=outputs/autoif_qwen25_05b_lora/checkpoint-XXXX`.
- Watch `logs/log_vXX_.../train/run.log` and your console for progress.
- Tuning for VRAM:
  - If OOM: lower `BATCH` or increase `GRAD_ACCUM` in run_pipeline.bat overrides: e.g. `BATCH=2 GRAD_ACCUM=16`.
  - Gradient checkpointing is already on for memory savings.

## 7) Export merged checkpoint

This merges LoRA adapters into a single Hugging Face model folder:
```powershell
.\run_pipeline.bat export-merged
```
Result: `outputs/autoif_qwen25_05b_lora/merged`

## 8) Quick evaluation

Runs a few prompts from config.yaml against the merged model:
```powershell
.\run_pipeline.bat eval-sft
```
Outputs go to `outputs/autoif_qwen25_05b_lora/eval` and logs under `logs/`.

## 9) Serve with vLLM and test

Two options:

- GPU serving (recommended):
  - In docker-compose.yml, set `VLLM_TARGET_DEVICE: cuda` and ensure GPU is enabled (see step 2).
- CPU serving (already configured by default): no changes needed.

vLLM expects the merged model under the mount `\\pc-27327\D\LLM` mapped to `/models`. Copy your merged output there (or change the compose mount to point to your local `outputs/` folder).

Copy merged to the share (example path; adjust if needed):
```powershell
# Create the target folder and copy merged model
$target="\\pc-27327\D\LLM\autoif_qwen25_05b_lora\merged"
New-Item -ItemType Directory -Force -Path $target | Out-Null
Copy-Item -Recurse -Force .\outputs\autoif_qwen25_05b_lora\merged\* $target\
```

Start vLLM:
```powershell
docker compose up -d vllm-server
```

Sanity test (simple prompt via your CLI, if you wired a smoke-test command):
```powershell
python -m src.cli.main smoke-test --prompt "Résume AC215 en trois points."
```
Or use an OpenAI-compatible client against http://localhost:8080 (served model name is `Qwen2.5-0.5B-SFT` per compose).

## 10) Hugging Face token and SSL tips

- Put your token in a `.env` file next to docker-compose.yml as `HF_TOKEN=hf_...`. The compose passes it as env.
- SSL is already disabled in the Dockerfile, compose, and Python utils to bypass corporate TLS interception.

## Optional: Preflight tests (fast)

Run these locally before heavy stages:
```powershell
pytest tests/test_hf_connectivity.py
pytest tests/test_pipeline_smoke.py
```

## Troubleshooting

- Container can’t see GPU:
  - Ensure `gpus: all` is set for `qwen25-05b`
  - `docker exec -it qwen25-05b bash -lc "nvidia-smi"`
  - Update NVIDIA driver / Docker Desktop; verify `docker run --gpus all nvidia/cuda:... nvidia-smi` works on host
- OOM during training:
  - Reduce `BATCH`, increase `GRAD_ACCUM`, keep gradient checkpointing enabled
  - Ensure `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` is set (it is in the Dockerfile)
- vLLM can’t find model:
  - Ensure the merged folder exists under the network share mapped to `/models` OR change the volume to mount your local `outputs/` directly

## Quick recap

- You’ve already built the image. Next steps:
  1) Enable GPUs in compose (uncomment `gpus: all` and optionally set vLLM to `cuda`)
  2) Set `train.max_steps: null` in config.yaml for a full run
  3) `.run_pipeline.bat up`
  4) `.run_pipeline.bat preprocess-sft`
  5) `.run_pipeline.bat finetune-sft`
  6) `.run_pipeline.bat export-merged`
  7) Copy merged to the share and `docker compose up -d vllm-server`
  8) Test the endpoint or use the CLI smoke-test

If you confirm whether you want GPU serving (vs CPU) and that you’re ready to remove the smoke cap, I can apply those small YAML/compose edits for you.