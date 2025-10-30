# v2 Docker images and containers
- [x] verify that docker is the default run, not python in .bat file
- [x] Build Docker image via `run_pipeline.bat build`
- [x] Launch training container (`run_pipeline.bat up`) using smoke config
- [x] New regression test
    - [x] Inspect runtime_probe.py and design a pytest that mimics the Docker probe output (including a CUDA banner).
    - [x] Implement test under tests (e.g., tests/test_runtime_probe.py) asserting non-KEY=VALUE lines are filtered and that DEBUG_PIPELINE is emitted when requested.
    - [x] Run tests inside container: docker compose run --rm --no-deps -T sft pytest
- [x] Config flag wiring
    - [x] Add logging.debug_pipeline: false to config.yaml with a concise inline comment (enables verbose batch traces).
    - [x] Update config.py LoggingConfig dataclass to include debug_pipeline: bool = False and ensure parsing handles the optional field.
    - [x] Update main.py (_build_runtime_metadata) to set DEBUG_PIPELINE=1 when true, else 0 so run_pipeline.bat receives it.
- [x] Debug configuration
    - [x] Create debug_config.yaml: reduced dataset (sample_size: 16), CPU-friendly settings (disable bf16/fp16), smaller batch sizes, logging.debug_pipeline: true.
    - [x] Launch with run_pipeline.bat up CONFIG=debug_config.yaml and confirm debug tracing flows through.
- [x] Documentation updates
    - [x] Add a clear inline comment for logging.debug_pipeline in config.yaml.
    - [x] Extend docs/1.setup.md with a short section describing how to enable debug mode and use debug_config.yaml.
- [ ] Batch cleanup
    - [ ] Remove temporary debug traces and helper blocks from run_pipeline.bat, keep only essential in-container filtering and minimal logging.
    - [x] Retest run_pipeline.bat up (default config) and verify pipeline starts without ". était inattendu." errors.

# v3 Preprocess, SFT, merge, console cleaning
- [x] Execute `run_pipeline.bat preprocess-sft` (smoke config)
- [x] Execute `run_pipeline.bat finetune-sft` fresh run
- [x] verify Naming convention: run_pipeline.bat commands should always use the correct naming
- [x] make CLI console less noisy
- [x] Execute `run_pipeline.bat export-merged`

# v4 awq convert 
- [ ] save as `awq` --> vLLM using llm-compressor (pip install llmcompressor)
- [x] Revert the autoawq attempt
    - Edit requirements.txt to remove `autoawq==...`.
    - Edit Dockerfile to remove autoawq install line and trigger rebuild later.
    - Move awq_converter.py -> `src/training/awq_converter_deprecated.py` (or delete) and leave a tiny shim that logs deprecation and points to the new converter.
    - Update or delete test_awq_conversion.py.
- [x] Add llm-compressor dependency
- [x] Implement converter that calls upstream CLI - Add `src/training/llm_compressor_converter.py` exposing `convert_to_llc(merged_path, out_path, options, force, logger)`
- [x] CLI & wrapper wiring - `convert-llc` Typer command to main.py that delegates to the new converter and uses `llm_compressor` config block.
- [x] Add the llm_compressor config dataclass and parse it in config.py so we can load options from config files on top of CLI args.
- [x] wire config defaults into the convert-llc Typer command (so CLI flags override config values and omitted flags fall back to cfg.llm_compressor)
- [x] extend run_pipeline.bat to call this new CLI when `llm_compressor.enable` is true (or add a separate wrapper invocation `convert-llc`).
- [x] add unit test for 
    - [x] Config parsing — test `ConfigLoader` YAML handling and edge cases.
    - [x] CLI precedence — verify CLI flags override config values.
    - [x] Converter behavior — mock CLI presence, subprocess, and fallback logic.
- [ ] debug package conflicts, isolate llm_compressor because of numpy version issues.
    - [x] create dedicated Dockerfile.llc for llm-compressor runs
    - [x] Update llc-runner Dockerfile to install llm-compressor and its dependencies.
    ### 1. Create GPU-Capable `awq-runner` Dockerfile
    - [x] update `Dockerfile.llc-runner` to `Dockerfile.awq-runner` with `AWQ_GPU` build arg.
    - [x] Use CUDA base image if GPU enabled; otherwise use slim Python.
    - [x] Install `llmcompressor`, `PyYAML`, and optional modifiers.
    ### 2. Implement `scripts/awq_runner.py`
    - [x] Parse CLI args: `--config`, `--merged`, `--out`, `--force`, `--gpu/--cpu`.
    - [x] Load config and validate AWQ section.
    - [x] Call `llmcompressor.oneshot(...)` and capture stdout/stderr.
    - [x] Write `metadata.json` with full schema and exit codes.
    ### 3. Add `awq-runner` to `docker-compose.yml`
    - [x] Define service with workspace volume mount.
    - [x] Add GPU support via `gpus: all` and `runtime: nvidia` or override file.
    ### 4. CLI Changes and Cleanup
    - [x] Remove `llm_compressor_converter.py` and related helpers.
    - [x] Remove `convert-llc` command from `main.py`.
    - [x] Add `convert-awq` command that shells out to `awq-runner` or runs locally.
    ### 5. Update Config Schema
    - [x] Replace `LlmCompressorConfig` with `AwqConversionConfig`.
    - [x] Include fields: `enabled`, `gpu_enabled`, `method`, `scheme`, `extra_args`, etc.
    - [x] Update `config.yaml` and `debug_config.yaml` samples.
    ### 6. Update `run_pipeline.bat`
    - [x] Add `:convert-awq` target for manual invocation.
    - [x] In `:export`, conditionally run AWQ if enabled or forced.
    - [x] Support `FORCE_AWQ=1` or `/force` flag.
    ### 7. Update Tests
    - [x] Remove tests for old converter.
    - [x] Add unit tests for `awq_runner.py` (mock oneshot, assert metadata, config, container).
    ### 8. Update Requirements and Docs
    - [x] Remove `llmcompressor` from repo-wide `requirements.txt`.
    - [x] Update `docs/awq-compression.md` with runner usage, build steps, CLI flags, and troubleshooting.
    ### 9. CI Pipeline Updates
- [ ] Ensure main CI skips `llmcompressor` install.
    - [ ] Add optional `awq-integration` job for GPU-enabled runners.
    ### 10. Verification Checklist
    - [x] Build runner: docker compose build awq-runner
    - [x] DEBUG error : [failed to solve: process "/bin/sh -c pip3 install --no-cache-dir --upgrade pip setuptools wheel && pip3 install --no-cache-dir llmcompressor==${LLMCOMPRESSOR_VERSION} PyYAML" did not complete successfully: exit code: 1]
        - [x] Rebuild with plain logs to check if version is empty
        - [x] Build GPU stage directly to isolate the issue  
        - [x] Inspect Dockerfile top lines for ARG declaration
        - [x] redeclared ARG for LLMCOMPRESSOR_VERSION in the GPU stage
        - [x] Rebuild to confirm fix
    - [x] debug
        - [x] **Check which converter path is running**
            - Inspect `run_pipeline.bat` for `convert-awq` command.
            - Verify `main.py` calls `docker compose run awq-runner` (not AutoAWQ).
            - Look for `llmcompressor compress` in logs; if you see `awq.quantize.quantizer`, it’s old code.
        - [x] **Rebuild the awq-runner image without cache**
            - Run:  
                ```powershell
                docker compose build --no-cache awq-runner
                ```
            - Confirm updated `awq_runner.py` and llm-compressor are included.
        - [x] **Verify llm-compressor CLI availability**
            - Exec into container:  
                ```powershell
                docker compose run --rm --no-deps --entrypoint bash awq-runner -lc "llmcompressor --help || python3 -m llmcompressor.oneshot --help"
                ```
            - Current status: only the `oneshot` entrypoint resolves; plain `llmcompressor` script is missing in 0.8.1. Track upstream or add our own console script shim so the wrapper can prefer CLI when available.
        - [x] **Remove AutoAWQ from runtime**
            - awq-runner: `pip show autoawq` already returns "not installed" (✅).
            - sft image still ships AutoAWQ 0.2.9 — strip it from the Dockerfile/requirements and rebuild (`docker compose build --no-cache sft`).
            - After rebuild, rerun `pip show autoawq` inside `sft`; ensure only `llmcompressor` remains so no legacy fallback occurs.
        - [x] **Validate wiring of convert-awq call**
            - Ensure `run_pipeline.bat convert-awq` uses:
                - `--config <workspace>/config.yaml`
                - `--merged <workspace>/outputs/.../merged`
                - `--out <workspace>/outputs/.../merged_awq`
            - Check GPU flags in docker-compose for awq-runner.
        - [x] **Use local calibration data**
            - Update `config.yaml` and `debug_config.yaml`:
                - `awq.calib_text_file` → local file path.
                - `awq.num_calibration_samples` → 128–512.
                - `awq.max_seq_length` → 1024–2048.
            - For AWQ W4A16, use CLI path first.
        - [x] **Re-run convert-awq and validate outputs**
        - Check:
            - `outputs/<run>/merged_awq` exists.
            - `metadata.json` → `returncode == 0`.
            - Logs in `logs/log_vXX.../convert-awq/` show no exceptions.
        - [ ] **Document working settings**
        - Record configs and versions in `5.awq-compression.md` and README.
        
    - [x] Inspect `metadata.json` and logs for correctness.
- [x] Cleanup - Remove deprecated AWQ code

# v5
## implementing tensorboard to track live metrics via port 6006
- [ ] Add `tensorboard>=2.16` to requirements.txt and rebuild the `sft` image, checking for dependency conflicts
- [ ] Wire Trainer to `report_to=["tensorboard"]`, set `logging_dir` to `<run_dir>/tensorboard`, and tune `logging_steps`
- [ ] Add `training.report_to`, `training.logging_dir`, and `training.logging_steps` to config.yaml and config dataclass
- [ ] Expose `TENSORBOARD_LOGDIR` in `_build_runtime_metadata` (main.py)
- [ ] Add `tensorboard` service to docker-compose.yml (mount logs, port 6006, command to serve logdir) which shall be accessible on other machines in the same network
- [ ] Add run_pipeline.bat targets: `tensorboard-up`, `tensorboard-down`, `tensorboard-logs` and print TB URL on training start
- [ ] Create docs/8.tensorboard.md and link from README and finetune docs
- [ ] Add tests/test_tensorboard_logging.py to assert a per-run tensorboard/ folder and events file after a minimal finetune
- [ ] Add .gitignore entries for tensorboard output and run format/type checks

# v6
- [ ] Execute `run_pipeline.bat finetune-sft RESUME_FROM=latest` for resume validation
- [ ] Execute `run_pipeline.bat export-merged RESUME_FROM=latest` (idempotence check)
- [ ] Execute `run_pipeline.bat eval-sft`
- [ ] Execute `run_pipeline.bat serve-vllm`
- [ ] Capture & document any fixes/tests for encountered issues