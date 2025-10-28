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

# v3 Preprocess
- [x] Execute `run_pipeline.bat preprocess-sft` (smoke config)
- [x] verify Naming convention: run_pipeline.bat commands should always use the correct naming

# v4 , SFT and merge
- [ ] Execute `run_pipeline.bat finetune-sft` fresh run
- [ ] Execute `run_pipeline.bat finetune-sft RESUME_FROM=latest` for resume validation
- [ ] Execute `run_pipeline.bat export-merged`
- [ ] Execute `run_pipeline.bat export-merged RESUME_FROM=latest` (idempotence check)
- [ ] save as `awq` --> vLLM

# v4
- [ ] Execute `run_pipeline.bat eval-sft`
- [ ] Execute `run_pipeline.bat serve-vllm`
- [ ] Capture & document any fixes/tests for encountered issues
- [ ] Make eval part better by implementing tensorboard to track live metrics via port 6006