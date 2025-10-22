@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Always run from the script directory so docker-compose.yml and .env are found
set "_SCRIPT_DIR=%~dp0"
pushd "%_SCRIPT_DIR%" >nul 2>&1
set "PIPELINE_EXIT_CODE="

REM ============================================================================
REM ============================ Default values ================================
REM ============================================================================
set "CURL_CA_BUNDLE="
set "REQUESTS_CA_BUNDLE="
set "SSL_CERT_FILE="
set "PYTHONHTTPSVERIFY=0"
set "HF_HUB_DISABLE_SSL_VERIFY=1"
set "HF_HUB_ENABLE_XET=0"
set "GIT_SSL_NO_VERIFY=1"

set "SERVICE=qwen25-05b"
set "CONFIG=config.yaml"
set "DATASET=tatsu-lab/alpaca"
set "SAMPLE_SIZE=2000"
set "FILTER_LANGS=en,fr"
set "TEST_SIZE=0.2"
set "CUTOFF=2048"
set "SEED=215"
set "MAX_WORKERS=4"
set "PACK_SEQS=true"
set "PREP_DIR=prepared/alpaca_2k_en"

set "BASE_MODEL=Qwen/Qwen2.5-0.5B"
set "BATCH=4"
set "GRAD_ACCUM=8"
set "EPOCHS=1"
set "LR=2e-5"
set "MIN_LR=5e-6"
set "WEIGHT_DECAY=0.01"
set "WARMUP_RATIO=0.03"
set "SCHEDULER=cosine"
set "LORA_R=16"
set "LORA_ALPHA=32"
set "LORA_DROPOUT=0.05"
set "LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
set "GRAD_CHECKPOINT=true"
set "BF16=true"
set "FP16=true"
set "LOG_STEPS=20"
set "EVAL_STEPS=100"
set "OUTPUT_DIR=outputs/autoif_qwen25_05b_lora"

set "MERGED_DIR=%OUTPUT_DIR%/merged"
set "EVAL_DIR=%OUTPUT_DIR%/eval"
set "SERVE_PORT=8080"

REM ============================================================================
REM ============================ Parse arguments ===============================
REM ============================================================================
if "%~1"=="" goto :help
set "TARGET=%~1"
shift

:parse_args
if "%~1"=="" goto :dispatch
set "ARG=%~1"
echo(%ARG%| findstr "=" >nul
if not errorlevel 1 (
    for /f "tokens=1,* delims==" %%A in ("%ARG%") do (
        set "%%A=%%B"
    )
    shift
    goto :parse_args
)
if "%~2"=="" (
    echo [ERROR] Missing value for parameter %~1
    exit /b 2
)
call set "%~1=%~2"
shift
shift
goto :parse_args

REM ============================================================================
REM ============================== Dispatch ====================================
REM ============================================================================
:dispatch
echo [ARGS] TARGET=%TARGET%
if /I "%TARGET%"=="build"           goto :build
if /I "%TARGET%"=="up"              goto :up
if /I "%TARGET%"=="down"            goto :down
if /I "%TARGET%"=="bash"            goto :bash
if /I "%TARGET%"=="preprocess-sft" goto :preprocess
if /I "%TARGET%"=="finetune-sft"   goto :finetune
if /I "%TARGET%"=="export-merged"  goto :export
if /I "%TARGET%"=="eval-sft"       goto :evaluate
if /I "%TARGET%"=="serve-vllm"     goto :serve
if /I "%TARGET%"=="show-last"      goto :showlast
goto :help

REM ============================================================================
REM ============================= Commands =====================================
REM ============================================================================
:build
docker compose build
goto :eof

:up
docker compose up -d %SERVICE%
goto :eof

:down
docker compose down
goto :eof

:bash
docker exec -it %SERVICE% bash
goto :eof

:preprocess
echo [PREPROCESS] PREP_DIR=%PREP_DIR%
rem Build command without line-continuation carets to avoid PowerShell/CMD parsing issues
set "CMD=python3 -m src.cli.main preprocess-sft"
rem Using default config path from CLI
set "CMD=%CMD% --dataset-name=%DATASET%"
set "CMD=%CMD% --sample-size=%SAMPLE_SIZE%"
set "CMD=%CMD% --filter-langs=%FILTER_LANGS%"
set "CMD=%CMD% --test-size=%TEST_SIZE%"
set "CMD=%CMD% --cutoff-len=%CUTOFF%"
set "CMD=%CMD% --seed=%SEED%"
set "CMD=%CMD% --max-workers=%MAX_WORKERS%"
rem Boolean flags: Typer expects --pack-sequences / --no-pack-sequences
if /I "%PACK_SEQS%"=="true"  set "CMD=%CMD% --pack-sequences"
if /I "%PACK_SEQS%"=="false" set "CMD=%CMD% --no-pack-sequences"
set "CMD=%CMD% --output=%PREP_DIR%"
echo [CMD] %CMD%
docker exec -it %SERVICE% bash -lc "%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs preprocess
goto :eof

:finetune
echo [FINETUNE] OUTPUT=%OUTPUT_DIR%
set "CMD=python3 -m src.cli.main finetune-sft"
rem Using default config path from CLI
set "PREP_DIR_ARG=%PREP_DIR:\=/%"
set "OUTPUT_DIR_ARG=%OUTPUT_DIR:\=/%"
set "CMD=%CMD% --data-dir=%PREP_DIR_ARG%"
set "CMD=%CMD% --output-dir=%OUTPUT_DIR_ARG%"
set "CMD=%CMD% --base-model=%BASE_MODEL%"
set "CMD=%CMD% --cutoff-len=%CUTOFF%"
set "CMD=%CMD% --batch-size=%BATCH%"
set "CMD=%CMD% --gradient-accumulation=%GRAD_ACCUM%"
set "CMD=%CMD% --epochs=%EPOCHS%"
set "CMD=%CMD% --learning-rate=%LR%"
set "CMD=%CMD% --min-learning-rate=%MIN_LR%"
set "CMD=%CMD% --weight-decay=%WEIGHT_DECAY%"
set "CMD=%CMD% --warmup-ratio=%WARMUP_RATIO%"
set "CMD=%CMD% --lr-scheduler=%SCHEDULER%"
set "CMD=%CMD% --lora-r=%LORA_R%"
set "CMD=%CMD% --lora-alpha=%LORA_ALPHA%"
set "CMD=%CMD% --lora-dropout=%LORA_DROPOUT%"
set "CMD=%CMD% --lora-targets=%LORA_TARGETS%"
rem Boolean flags: Typer expects --<flag> / --no-<flag>
if /I "%GRAD_CHECKPOINT%"=="true"  set "CMD=%CMD% --gradient-checkpointing"
if /I "%GRAD_CHECKPOINT%"=="false" set "CMD=%CMD% --no-gradient-checkpointing"
if /I "%BF16%"=="true"  set "CMD=%CMD% --bf16"
if /I "%BF16%"=="false" set "CMD=%CMD% --no-bf16"
if /I "%FP16%"=="true"  set "CMD=%CMD% --fp16"
if /I "%FP16%"=="false" set "CMD=%CMD% --no-fp16"
set "CMD=%CMD% --logging-steps=%LOG_STEPS%"
set "CMD=%CMD% --eval-steps=%EVAL_STEPS%"
rem Optional resume: only if explicitly provided or checkpoint exists
if defined RESUME_FROM (
    set "RESUME_ARG=%RESUME_FROM:\=/%"
    set "CMD=%CMD% --resume-from=%RESUME_ARG%"
) else (
    set "RESUME_DIR=%OUTPUT_DIR%\trainer_state"
    if exist "!RESUME_DIR!" (
        set "LATEST_CHECKPOINT="
        set "LATEST_STEP="
        for /f "delims=" %%C in ('dir /b /ad "!RESUME_DIR!\checkpoint-*" 2^>nul') do (
            set "STEP="
            for /f "tokens=2 delims=-" %%S in ("%%C") do set "STEP=%%S"
            if not defined STEP set "STEP=0"
            if not defined LATEST_STEP (
                set "LATEST_STEP=!STEP!"
                set "LATEST_CHECKPOINT=%%C"
            ) else if !STEP! gtr !LATEST_STEP! (
                set "LATEST_STEP=!STEP!"
                set "LATEST_CHECKPOINT=%%C"
            )
        )
        if defined LATEST_CHECKPOINT (
            set "CMD=%CMD% --resume-from=%OUTPUT_DIR_ARG%/trainer_state/!LATEST_CHECKPOINT!"
        ) else if exist "!RESUME_DIR!\trainer_state.json" (
            set "CMD=%CMD% --resume-from=%OUTPUT_DIR_ARG%/trainer_state"
        ) else (
            rem Trainer state exists but no checkpoints to resume from
        )
    ) else (
        rem No resume on first run
    )
)
echo [CMD] %CMD%
docker exec -it %SERVICE% bash -lc "%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs finetune
goto :eof

:export
echo [EXPORT] MERGED_DIR=%MERGED_DIR%
set "CMD=python3 -m src.cli.main export-merged"
rem Using default config path from CLI
set "CMD=%CMD% --adapter-dir=%OUTPUT_DIR%/adapter"
set "CMD=%CMD% --output-dir=%MERGED_DIR%"
set "CMD=%CMD% --base-model=%BASE_MODEL%"
echo [CMD] %CMD%
docker exec -it %SERVICE% bash -lc "%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs export
goto :eof

:evaluate
echo [EVAL] EVAL_DIR=%EVAL_DIR%
set "CMD=python3 -m src.cli.main eval-sft"
rem Using default config path from CLI
set "CMD=%CMD% --model-dir=%MERGED_DIR%"
set "CMD=%CMD% --output-dir=%EVAL_DIR%"
set "CMD=%CMD% --cutoff-len=%CUTOFF%"
echo [CMD] %CMD%
docker exec -it %SERVICE% bash -lc "%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs eval
goto :eof

:serve
echo [SERVE] Port %SERVE_PORT%
docker compose up -d vllm-server
echo [INFO] vLLM server exposed on http://localhost:%SERVE_PORT%
goto :eof

:showlast
if not exist logs\latest.txt (
    echo [WARN] logs/latest.txt not found yet.
    goto :eof
)
type logs\latest.txt
goto :eof

:capture_container_logs
setlocal EnableExtensions EnableDelayedExpansion
set "STAGE=%~1"
if not exist logs\latest.txt goto :capture_done
set "LATEST="
for /f "usebackq tokens=* delims=" %%L in ("logs\latest.txt") do set "LATEST=%%L"
if not defined LATEST goto :capture_done
set "HOST_PATH=!LATEST:/app/=!"
set "HOST_PATH=!HOST_PATH:/=\!"
if "!HOST_PATH:~0,1!"=="\" set "HOST_PATH=!HOST_PATH:~1!"
if not defined HOST_PATH goto :capture_done
if not exist "!HOST_PATH!" goto :capture_done
set "TARGET_FILE=!HOST_PATH!\container.log"
echo [LOGS] Capturing docker logs for %STAGE% into !TARGET_FILE!
docker logs --details --timestamps --tail 2000 %SERVICE% >"!TARGET_FILE!" 2>&1
if errorlevel 1 (
    echo [WARN] Unable to capture docker logs for %STAGE%>>"!TARGET_FILE!"
)

:capture_done
endlocal
exit /b 0

:help
echo Usage: run_pipeline.bat ^<target^> [KEY=VALUE ...]
echo.
echo Targets:
echo    build                Build Docker image
echo    up                   Start training container
echo    down                 Stop all containers
echo    bash                 Open bash shell inside training container
echo    preprocess-sft       Run preprocessing stage
echo    finetune-sft         Run LoRA fine-tuning stage
echo    export-merged        Merge LoRA adapter into base model
echo    eval-sft             Run evaluation checks
echo    serve-vllm           Launch vLLM server on merged model
echo    show-last            Print path to latest log folder
echo.
echo Example:
echo    run_pipeline.bat preprocess-sft SAMPLE_SIZE=2000 PREP_DIR=prepared/demo
exit /b 1

:eof
REM Return to original directory
popd >nul 2>&1
if defined PIPELINE_EXIT_CODE (
    set "_SCRIPT_STATUS=%PIPELINE_EXIT_CODE%"
) else (
    set "_SCRIPT_STATUS=%ERRORLEVEL%"
)
set "PIPELINE_EXIT_CODE="
exit /b %_SCRIPT_STATUS%
