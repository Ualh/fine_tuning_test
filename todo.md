# Verification Checklist (2025-10-24)

- [] verify that docker is the default run, not python in .bat file
- [] Build Docker image via `run_pipeline.bat build`
- [ ] Launch training container (`run_pipeline.bat up`) using smoke config
- [ ] Run full `pytest` suite inside container
- [ ] Execute `run_pipeline.bat preprocess-sft` (smoke config)
- [ ] Execute `run_pipeline.bat finetune-sft` fresh run
- [ ] Execute `run_pipeline.bat finetune-sft RESUME_FROM=latest` for resume validation
- [ ] Execute `run_pipeline.bat export-merged`
- [ ] Execute `run_pipeline.bat export-merged RESUME_FROM=latest` (idempotence check)
- [ ] Execute `run_pipeline.bat eval-sft`
- [ ] Execute `run_pipeline.bat serve-vllm`
- [ ] Capture & document any fixes/tests for encountered issues
- [ ] Make eval part better by implementing tensorboard to track live metrics via port 6006
- [ ] save as `awq` --> vLLM