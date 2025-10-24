## update plan 
### 1) Make the served model configurable in config.yaml

- Add a new field in `serve`:
  - served_model_relpath: relative path under outputs to the merged model directory.
  - Keep the existing served_model_name and max_model_len.
  - Document clearly that the model must live under outputs (host) which is mounted to `/models` (container). Example: `autoif_qwen25_05b_lora/merged` becomes `/models/autoif_qwen25_05b_lora/merged` inside vLLM.
- Wire-up strategy (two options—please choose):
  1) Simple and robust: use a .env file for Docker and document a one-liner to set:
     - SERVED_MODEL_PATH=/models/<served_model_relpath>
     - SERVED_MODEL_NAME=<served_model_name>
     - MAX_MODEL_LEN=<max_model_len>
     Then keep config.yaml as the source of truth in docs; users copy values into .env before serve.
  2) Fully automated: enhance `run_pipeline.bat serve-vllm` to parse config.yaml for these fields and set the env vars for `docker compose up`. This is doable in batch (findstr + for /f), but a bit brittle. If you prefer, I’ll implement it.

What I’ll change (after approval):
- Update config.yaml serve section with served_model_relpath and comments.
- Update docker-compose.yml vllm-server command to use env vars for name and max len too:
  - --served-model-name ${SERVED_MODEL_NAME:-Qwen2.5-0.5B-SFT}
  - --max-model-len ${MAX_MODEL_LEN:-2048}
- Optionally update run_pipeline.bat to export SERVED_MODEL_PATH (and optionally SERVED_MODEL_NAME, MAX_MODEL_LEN) before calling `docker compose up -d vllm-server` based on config.yaml (if you pick option 2).

Assumptions (please confirm):
- Default merged output exists at merged.
- You’re fine keeping 2048 as a default max context.

### 2) Start the server

Two supported ways:
- Wrapper: `run_pipeline.bat serve-vllm` (will rely on env vars if we do Option 1, or auto-read config if Option 2).
- Compose: `docker compose up -d vllm-server`

I’ll add a short note to README showing both.

### 3) Open Windows Defender Firewall for TCP 8080

PowerShell command to add an inbound rule:
```powershell
New-NetFirewallRule -DisplayName "vLLM 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
```

I’ll add this to README “vLLM serving” section.

### 4) Nicer URL and persistence

- Since the host appears to be pc-27327, others on the same network can reach http://pc-27327:8080 if:
  - Firewall allows 8080 (above).
  - Their DNS resolves pc-27327 correctly. If not, they can add a hosts entry on their own machines:
    - Windows hosts file: C:\Windows\System32\drivers\etc\hosts
    - Add: 192.168.x.y  pc-27327
- Persistence:
  - vLLM uses your outputs mount; it’s already persistent on the host.
  - OpenWebUI data will be persisted via a new host bind: ./open-webui:/app/backend/data (see below).

I’ll add a short README note explaining both.

### 5) Add Dozzle to docker-compose

Avoid port conflict with vLLM on 8080 by using 9999:
- Service:
  - name: dozzle
  - container: amir20/dozzle:latest
  - mounts: /var/run/docker.sock
  - env: DOZZLE_ENABLE_ACTIONS=true, DOZZLE_NO_ANALYTICS=true
  - ports: "9999:8080"
- You’ll access it at http://localhost:9999 (or http://pc-27327:9999) to tail logs easily.

I’ll add the service to docker-compose.yml and a sentence to README.

### 6) Add OpenWebUI to docker-compose

- Service:
  - image: ghcr.io/open-webui/open-webui:main
  - ports: "3000:3000"
  - env:
    - WEBUI_AUTH=True
    - OPENAI_API_BASE_URLS=http://vllm-server:8000/v1
    - ENABLE_OLLAMA_API=False
    - PORT=3000
    - OFFLINE_MODE=0
    - HF_HUB_OFFLINE=0
  - volumes:
    - ./open-webui:/app/backend/data
  - No network_mode: host (Windows doesn’t support it).
  - Depends on vllm-server (optional but nice).

You’ll access it at http://localhost:3000 (or http://pc-27327:3000). Inside the WebUI, set an API key placeholder if needed; base URL points at the internal service name, so it should work out of the box.

### 7) Test and call the endpoint

- Health check from your machine:
  - http://localhost:8080/v1/models
- CLI quick test:
  - python -m src.cli.main smoke-test --prompt "Say hello in French."
- From OpenWebUI: open http://localhost:3000, ensure API base URL is already set to vllm-server’s internal URL, and chat.
- From the network: http://pc-27327:8080 (and http://pc-27327:3000) once DNS/hosts is in place.

I’ll add a short “Try it” block in the README.



## Clarifications

- Which default merged model directory should we use? I’ll assume merged unless you prefer one of the other runs in `outputs/autoif_qwen25_05b_lora_full*`.
- Do you prefer Option 1 (.env-based) or Option 2 (auto-parse config.yaml in the batch script) for making the serve model path/name configurable at runtime?
- Are the proposed ports okay?
  - vLLM: 8080
  - Dozzle: 9999 (instead of conflicting 8080)
  - OpenWebUI: 3000
- Should I also make served_model_name configurable via .env (and pass it into vLLM) to exactly mirror config.yaml?
