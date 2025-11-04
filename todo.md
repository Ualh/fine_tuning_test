# v2 ✅ Docker images and containers
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

# v3 ✅ Preprocess, SFT, merge, console cleaning
- [x] Execute `run_pipeline.bat preprocess-sft` (smoke config)
- [x] Execute `run_pipeline.bat finetune-sft` fresh run
- [x] verify Naming convention: run_pipeline.bat commands should always use the correct naming
- [x] make CLI console less noisy
- [x] Execute `run_pipeline.bat export-merged`

# v4 ✅ awq convert 
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
            - Logs in `logs/<run>/convert-awq/` show no exceptions.
        - [ ] **Document working settings**
        - Record configs and versions in `5.awq-compression.md` and README.
        
    - [x] Inspect `metadata.json` and logs for correctness.
- [x] Cleanup - Remove deprecated AWQ code

# v5 ✅
## implementing tensorboard to track live metrics via port 6006
- [x] Add `tensorboard>=2.16` to requirements.txt and rebuild the `sft` image, checking for dependency conflicts
- [x] Wire Trainer to `report_to=["tensorboard"]`, set `logging_dir` to `<run_dir>/tensorboard`, and tune `logging_steps`
- [x] Add `training.report_to`, `training.logging_dir`, and `training.logging_steps` to config.yaml and config dataclass
- [x] Expose `TENSORBOARD_LOGDIR` in `_build_runtime_metadata` (main.py)
- [x] Add `tensorboard` service to docker-compose.yml (mount logs, port 6006, command to serve logdir) which shall be accessible on other machines in the same network
- [x] Add run_pipeline.bat targets: `tensorboard-up`, `tensorboard-down`, `tensorboard-logs` and print TB URL on training start
- [x] Create docs/8.tensorboard.md and link from README and finetune docs
- [x] Add tests/test_tensorboard_logging.py to assert a per-run tensorboard/ folder and events file after a minimal finetune
- [x] Add .gitignore entries for tensorboard output and run format/type checks

# v6 ✅ training debugging TypeError while constructing TrainingArguments in sft_trainer.py

- [x] Reproduce the TypeError locally using a minimal debug run.
- [x] Patch `sft_trainer.py` to only pass supported kwargs to `TrainingArguments`.
- [x] Use `inspect.signature` to check if `logging_dir` is supported before passing it.
- [x] Add a unit test to ensure `TrainingArguments` construction doesn’t raise `TypeError`.
- [x] Run `test_tensorboard_logging.py` to verify logging behavior.
    - [x] debug
- [x] Run `test_pipeline_smoke.py` to verify pipeline integrity.
 - [x] Run a short finetune using `debug_config.yaml` inside the container.  
     (Run completed; short debug finetune finished successfully.)
 - [x] Confirm training starts and logs progress past `TrainingArguments` creation.  
     (Logs show "TensorBoard logging enabled" and the compatibility info message.)
 - [x] Confirm TensorBoard events are written to the correct directory.  
     (Event files were written by Trainer under `outputs/<run>/trainer_state/runs/...` — note: `logging_dir` was skipped by TrainingArguments in this runtime, so the Trainer wrote events to its default `trainer_state/runs` location instead of `logs/<run>/train/tensorboard`.)
 - [x] Add an info-level log if `logging_dir` is skipped due to signature mismatch.  
     (Implemented: `sft_trainer.py` logs an INFO when `logging_dir` is not accepted by `TrainingArguments`.)
- [x] Add a small TrainerCallback (or a simple tensorboard SummaryWriter) that always writes metrics to your canonical tensorboard_dir regardless of whether TrainingArguments accepted logging_dir. 

# v7 ✅ naming convention fix
- [x] standardize naming convention across all runs, logs, outputs, images, containers via `<model_name>-<dataset_name>-<dataset_size>-runX`
    - [x] Design naming spec & contract — write spec for `<model_name>-<dataset_name>-<dataset_size>-runX`, canonical sources, normalization, run counter rules, examples and pseudocode
    - [x] Add naming config & dataclass — add `naming` section to config.yaml and dataclass fields (normalize, separator, run_counter_scope, run_prefix, legacy_allowed)
    - [x] Implement canonical name builder function — create `src/core/run_naming.py` with `build_run_name(...)`, sanitization, separator handling and deterministic fallback; add unit tests
    - [x] Integrate with run manager — update run_manager.py to call builder when creating runs and return `run_name` and `run_dir`
    - [x] Wire into CLI runtime metadata — update main.py `_build_runtime_metadata` to export `RUN_NAME`, `RUN_DIR_NAME`, `RUN_NUMBER`, `SERVED_MODEL_NAME`
    - [x] Update run_pipeline.bat naming & container/image tags — use `RUN_NAME` for container names, image tags and output folders; add `FORCE_RUN_NAME`/`LEGACY_RUN_NAME` override and preview target
    - [x] Docker Compose & service names — update docker-compose.yml to use `RUN_NAME` for `container_name` or labels where appropriate without breaking service references
    - [x] Outputs/logs directory patterns and symlinks — create `outputs/<RUN_NAME>` and `logs/<RUN_NAME>` layout and maintain `outputs/latest`/`logs/latest` pointers; support dry-run preview
    - [x] Update docs and README — update README.md and 1.setup.md to document the convention, overrides
    - [x] verify the name images problem, Inconsistent project name source: one command run directly on the host (build) uses the host directory as the project name; later steps call print-runtime inside a container and that produces a project name derived from inside‑container paths. 
        - [x] ensure the merge-export step needs a adapter path to work as a standalone command, as otherwise it will create a new ..._runX output which will not have the adapter folder in it
        - [x] ensure that the preprocess-sft, convert-awq, eval-sft and serve-vllm commands also work as standalone commands without creating new run folders that do not have the expected content in them.
        - [x] modify the commands so that finetune-sft, if provided in the config, runs merge-export, convert-awq, eval-sft and serve-vllm automatically at the end of the finetune step so we avoid the problem of isolated commands that will create new run folders 
- [x] confirm run_pipeline.bat build only creates one image per run name, not multiple images with different tags for the same run name
- [x] confirm run_pipeline.bat up only creates one container per run name, not multiple containers with different names for the same run name
- [x] confirm that run_pipeline.bat preprocess-sft creates and use only one container, one image and one output folder per run name
    - [x] prepared/<dataset>_<size>/train.jsonl and val.jsonl exist.
    - [x] logs/<run>/preprocess/console.log contains the stage logs.
    - [x] latest.txt updated.
- [x] confirm that run_pipeline.bat finetune-sft
    - [x] does not creates and use other containers
    - [x] does not creates and use other images
    - [x] does not creates and use other output folders
    - [x] outputs/<run>/adapter/ exists after training.
    - [x] logs/<run>/train/console.log shows training summary.
- [x] confirm that run_pipeline.bat export-merged
    - [x] does not creates and use other containers
    - [x] does not creates and use other images
    - [x] does not creates and use other output folders
    - [x] outputs/<run>/merged/ contains model files (config.json, model.safetensors, tokenizer, metadata.json).
- [x] confirm that run_pipeline.bat convert-awq
    - [x] does not creates and use other containers
    - [x] does not creates and use other images
    - [x] does not creates and use other output folders
    - [x] outputs/<run>/merged_awq/metadata.json exists and has "returncode": 0. 
    - [x] logs/<run>/convert-awq/container.log has AWQ logs.
- [x] confirm that run_pipeline.bat eval-sft
    - [x] does not creates and use other containers
    - [x] does not creates and use other images
    - [x] does not creates and use other output folders
    - [x] outputs/<run>/eval/metrics.json created and contains results.
- [x] confirm that run_pipeline.bat serve-vllm
    - [x] does not creates and use other containers
    - [x] does not creates and use other images
    - [x] does not creates and use other output folders
    - [x] vLLM / Open WebUI / Dozzle services are up (check docker compose ps).
    - [x] vLLM endpoint health check: `python -m src.cli.main smoke-test --endpoint http://localhost:8080 --model Qwen2.5-0.5B-SFT`
    - [x] Open WebUI accessible at http://localhost:8081.
    - [x] Dozzle accessible at http://localhost:8082.
    - [x] Inspect and propose a small fix to the vllm-server service (volume/port) so the model loads and the smoke-test passes.
- [x] confirm full end to end run with `run_pipeline.bat finetune-sft CONFIG=config.yaml` works
    - [x] After all stages finish, you have outputs/<run>/adapter, outputs/<run>/merged, outputs/<run>/merged_awq, outputs/<run>/eval.
    - [x] One stable compose project was used for build/run (no proliferation of per-stage project names).
    - [x] vLLM serve step failed: container sft-vllm-server-1 exited with OSError: Can't load the configuration of '/models/autoif_qwen25_05b_lora/merged' (missing mount or wrong path). No serve log folder created.
    - [x] Noted AWQ used only 20 calibration samples vs. requested 128 (recorded in metadata); may impact quantization quality

# v8 docs update
- [ ] update docs with anything that is outdated or non-consistent
    - [ ] update README.md
    - [ ] update docs/1.setup.md
    - [ ] update docs/2.preprocess-sft.md
    - [ ] update docs/3.finetune-sft.md
    - [ ] update docs/4.export-merged.md
    - [ ] update docs/5.awq-compression.md
    - [ ] update docs/6.eval-sft.md
    - [ ] update docs/7.serve-vllm.md
    - [ ] update docs/8.tensorboard.md
    - [ ] update guide.md

# v9 small fixes
- [ ] add metadata to each log folder. So we have all info about the configs used in detailled for reproducibility purposes and info about the parameters used for each step.
- [ ] remove sparse_logs if not needed
- [ ] refractor src/ and scripts/ duplicates. Keep only src/ and remove scripts/. Be sure that the convert-awq still works after the refractoring.
- [ ] make the ouput of the convert-awq better, less noisy, only important info
- [ ] add progress bar or something for the user to know the progress of the quantization step, currently it seems to hang for long periods of time without any output
- [ ] `torch_dtype` is deprecated! Use `dtype` instead!
- [ ] refractor configs into one folder `configs/` with `smoke.yaml`, `debug.yaml`, `default.yaml`
- [ ] update all docs to reflect the new config paths
- refractor tests so that they are separated into their respective folders: `tests/pipeline/`, `tests/training/`, `tests/conversion/` and so on.
- [ ] test the tests after the move to ensure they still run correctly.
- [ ] refractor and put all dockerfiles into a `docker/` folder for better organization.
- [ ] verify all files are updated to reflect the new paths.
- [ ] assess if the awq/ folder can be removed safely, if yes remove it, and verify
- [ ] remove awq/ in docs and merge the file in it into the `5.awq-compression.md` doc.
- [ ] check if '/ folder can be removed safely, if yes remove it, and verify

# v10
- [ ] clean a bit the docker image and containers, reduce duplication and run only one image per service + one container per run/model
- [ ] on tensorboard, we observe that each run has the name of the log_vxx/train/tensorboard folder, we should fix that so that we have the model_name-dataset_name-dataset_size_runX

# v11
- [ ] Execute `run_pipeline.bat eval-sft`
- [ ] Execute `run_pipeline.bat serve-vllm`
- [ ] Capture & document any fixes/tests for encountered issues
