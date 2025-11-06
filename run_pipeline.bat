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
REM ===========================remove SSL===============================
set "SSL_CERT_FILE="
set "PYTHONHTTPSVERIFY=0"
set "HF_HUB_DISABLE_SSL_VERIFY=1"
set "HF_HUB_ENABLE_XET=0"
set "GIT_SSL_NO_VERIFY=1"
REM ==========================configure pipeline===========================
set "CONFIG=config.yaml"
set "SERVICE=sft"
set "VLLM_SERVICE=vllm-server"
REM ========================runtime metadata vars==========================
set "RUNTIME_LOADED="
set "RUNTIME_FILE="
set "RUNTIME_RAW="
set "COMPOSE_PROJECT="
set "SERVED_MODEL_PATH="
set "SERVED_MODEL_NAME="
set "SERVED_MODEL_MAX_LEN="
set "SERVE_PORT="
set "PIPELINE_STAGE="

if not defined HOST_COMPOSE_PROJECT (
    if defined COMPOSE_PROJECT_NAME (
        set "HOST_COMPOSE_PROJECT=%COMPOSE_PROJECT_NAME%"
    )
)
if not defined HOST_COMPOSE_PROJECT (
    if defined COMPOSE_PROJECT (
        set "HOST_COMPOSE_PROJECT=%COMPOSE_PROJECT%"
    )
)
if not defined HOST_COMPOSE_PROJECT (
    set "HOST_COMPOSE_PROJECT=sft"
)
if not defined COMPOSE_PROJECT (
    set "COMPOSE_PROJECT=%HOST_COMPOSE_PROJECT%"
)

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
:: Support simple flags like /force by setting an env var (e.g. /force -> FORCE_AWQ=1)
if "%ARG:~0,1%"=="/" (
    set "FLAGNAME=%ARG:~1%"
    if /I "%FLAGNAME%"=="force" (
        set "FORCE_AWQ=1"
    ) else (
        rem Normalize: convert leading slash flags to uppercase env var with value 1
        set "%FLAGNAME%=1"
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
if /I "%TARGET%"=="convert-awq"   goto :convert_awq
if /I "%TARGET%"=="tensorboard-up" goto :tensorboard_up
if /I "%TARGET%"=="tensorboard-down" goto :tensorboard_down
if /I "%TARGET%"=="tensorboard-logs" goto :tensorboard_logs
if /I "%TARGET%"=="run-name-preview" goto :run_name_preview
if /I "%TARGET%"=="show-last"      goto :showlast
if /I "%TARGET%"=="clean"           goto :clean
if /I "%TARGET%"=="prune"           goto :clean

goto :help

REM ============================================================================
REM ============================= Commands =====================================
REM ============================================================================
:build
REM Build should use the same compose project as other stages; resolve via
REM runtime probe to keep image tags and container names consistent.
call :ensure_runtime
if errorlevel 1 goto :eof
call :compose build
goto :eof

:up
set "PIPELINE_STAGE=up"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
    call :compose up %COMPOSE_NO_BUILD_FLAG% -d %SERVICE%
set "PIPELINE_STAGE="
goto :eof

:down
set "PIPELINE_STAGE=down"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
call :compose down --remove-orphans
set "PIPELINE_STAGE="
goto :eof

:bash
set "PIPELINE_STAGE=bash"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
call :compose exec %SERVICE% bash
set "PIPELINE_STAGE="
goto :eof

:preprocess
set "PIPELINE_STAGE=preprocess-sft"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
if defined PREPROCESS_DIR_CONTAINER echo [PREPROCESS] output=!PREPROCESS_DIR_CONTAINER!
set "CONFIG_POSIX=%CONFIG%"
set "CONFIG_POSIX=%CONFIG_POSIX:\=/%"
set "CMD=python3 -m src.cli.main preprocess-sft --config '%CONFIG_POSIX%'"
echo [CMD] %CMD%
    call :compose up %COMPOSE_NO_BUILD_FLAG% -d %SERVICE%
call :compose exec %SERVICE% bash -lc "!RUN_ENV_EXPORT!%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs preprocess
set "PIPELINE_STAGE="
goto :eof

:finetune
set "PIPELINE_STAGE=finetune-sft"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
if defined PREPROCESS_DIR_CONTAINER echo [FINETUNE] data=!PREPROCESS_DIR_CONTAINER!
if defined OUTPUT_DIR_CONTAINER echo [FINETUNE] output=!OUTPUT_DIR_CONTAINER!
if defined TENSORBOARD_LOGDIR echo [FINETUNE] TensorBoard=http://localhost:6006 (logdir=!TENSORBOARD_LOGDIR!)
REM If training requests tensorboard, ensure the service is up so users can
REM view metrics during the run. The runtime probe sets TENSORBOARD_ENABLED=1
REM when `tensorboard` is in `train.report_to`.
if "%TENSORBOARD_ENABLED%"=="1" (
    echo [TENSORBOARD] Config requests TensorBoard; ensuring service is running
    echo [TENSORBOARD] Ensuring tensorboard image is built
    call :compose build tensorboard
    call :compose up -d tensorboard
)
set "CONFIG_POSIX=%CONFIG%"
set "CONFIG_POSIX=%CONFIG_POSIX:\=/%"
set "CMD=python3 -m src.cli.main finetune-sft --config '%CONFIG_POSIX%'"
echo [CMD] %CMD%
    call :compose up %COMPOSE_NO_BUILD_FLAG% -d %SERVICE%
call :compose exec %SERVICE% bash -lc "!RUN_ENV_EXPORT!%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs finetune
if "%PIPELINE_EXIT_CODE%"=="0" (
    if defined POST_FINETUNE_TARGETS (
        echo [PIPELINE] post-finetune targets: !POST_FINETUNE_TARGETS!
        for %%S in (!POST_FINETUNE_TARGETS!) do (
            call :run_post_stage %%S
            if errorlevel 1 (
                set "PIPELINE_EXIT_CODE=!ERRORLEVEL!"
                set "PIPELINE_STAGE="
                goto :eof
            )
        )
    )
)
set "PIPELINE_STAGE="
goto :eof

:export
set "PIPELINE_STAGE=export-merged"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
if defined MERGED_DIR_CONTAINER echo [EXPORT] merged=!MERGED_DIR_CONTAINER!
set "CONFIG_POSIX=%CONFIG%"
set "CONFIG_POSIX=%CONFIG_POSIX:\=/%"
set "CMD=python3 -m src.cli.main export-merged --config '%CONFIG_POSIX%'"
echo [CMD] %CMD%
    call :compose up %COMPOSE_NO_BUILD_FLAG% -d %SERVICE%
call :compose exec %SERVICE% bash -lc "!RUN_ENV_EXPORT!%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs export
:: Post-export: Optionally run AWQ conversion using the isolated awq-runner
:: sidecar to avoid dependency conflicts (NumPy 2.x vs NumPy 1.x). The runner
:: reads the repo `config.yaml` (the runtime probe exposes AWQ_ENABLED and
:: AWQ_OUTPUT_SUFFIX). The conversion will run when AWQ_ENABLED=1 or if the
:: caller set FORCE_AWQ=1 (or passed /force).
set "RUN_AWQ=0"
if defined FORCE_AWQ set "RUN_AWQ=1"
if "%AWQ_ENABLED%"=="1" set "RUN_AWQ=1"
    
    REM Verbose AWQ decision logging
    echo [AWQ] Decision vars: AWQ_ENABLED=%AWQ_ENABLED% FORCE_AWQ=%FORCE_AWQ% MERGED_DIR=%MERGED_DIR%
    echo [AWQ] Computed RUN_AWQ=%RUN_AWQ%
if "%RUN_AWQ%"=="1" (
    if not defined FORCE_AWQ (
        if defined POST_FINETUNE_TARGETS (
            echo !POST_FINETUNE_TARGETS!| findstr /I /C:"convert-awq" >nul
            if not errorlevel 1 (
                echo [AWQ] convert-awq scheduled after finetune; skipping inline AWQ in export stage
                set "RUN_AWQ=0"
            )
        )
    )
)
if "%RUN_AWQ%"=="1" (
    echo [AWQ] Launching convert-awq via Python orchestrator
    call :run_awq_cli
    set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
    call :capture_container_logs convert-awq awq-runner
)
set "PIPELINE_STAGE="
goto :eof

:: ---------------------------------------------------------------------------
:convert_awq
set "PIPELINE_STAGE=convert-awq"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
echo [AWQ] Starting AWQ conversion check
:: Decide whether to run AWQ: respect runtime AWQ_ENABLED or explicit FORCE_AWQ
set "RUN_AWQ=0"
if defined FORCE_AWQ set "RUN_AWQ=1"
if "%AWQ_ENABLED%"=="1" set "RUN_AWQ=1"
if "%RUN_AWQ%"=="1" (
    if not defined MERGED_DIR (echo [ERROR] MERGED_DIR not set in runtime; cannot run AWQ & exit /b 2)
    echo [AWQ] Launching convert-awq via Python orchestrator
    call :run_awq_cli
    set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
    call :capture_container_logs convert-awq awq-runner
) else (
    echo [AWQ] AWQ conversion not requested (AWQ_ENABLED=%AWQ_ENABLED% FORCE_AWQ=%FORCE_AWQ%)
)
set "PIPELINE_STAGE="
goto :eof

:tensorboard_up
set "PIPELINE_STAGE=tensorboard-up"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
echo [TENSORBOARD] Launching dashboard on http://localhost:6006
if defined TENSORBOARD_LOGDIR echo [TENSORBOARD] tracking !TENSORBOARD_LOGDIR!
REM Ensure the tensorboard image exists. Build explicitly to avoid `--no-build` skipping
echo [TENSORBOARD] Ensuring tensorboard image is built
call :compose build tensorboard
if errorlevel 1 (
    echo [WARN] Building tensorboard image failed or aborted; attempting to start without build
)
call :compose up -d tensorboard
set "PIPELINE_STAGE="
goto :eof

:tensorboard_down
set "PIPELINE_STAGE=tensorboard-down"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
echo [TENSORBOARD] Stopping dashboard service
call :compose stop tensorboard >nul 2>&1
set "PIPELINE_STAGE="
goto :eof

:tensorboard_logs
set "PIPELINE_STAGE=tensorboard-logs"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
echo [TENSORBOARD] Streaming container logs (Ctrl+C to stop)
call :compose logs -f tensorboard
set "PIPELINE_STAGE="
goto :eof

:run_name_preview
call :build_naming_export
set "CONFIG_POSIX=%CONFIG%"
set "CONFIG_POSIX=%CONFIG_POSIX:\=/%"
if "%DEBUG_PIPELINE%"=="1" echo [PREVIEW] resolving run name via container
call :compose run --rm --no-deps -e "HOST_COMPOSE_PROJECT=%HOST_COMPOSE_PROJECT%" -T %SERVICE% bash -lc "!NAMING_EXPORT!python3 -m src.cli.main run-name-preview --config '%CONFIG_POSIX%'"
set "STATUS=%ERRORLEVEL%"
if %STATUS% EQU 0 goto :preview_done
echo [WARN] Container preview failed (%STATUS%). Falling back to local Python...
python -m src.cli.main run-name-preview --config "%CONFIG%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
goto :eof

:preview_done
set "PIPELINE_EXIT_CODE=0"
goto :eof

:evaluate
set "PIPELINE_STAGE=eval-sft"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
if defined MERGED_DIR_CONTAINER echo [EVAL] model=!MERGED_DIR_CONTAINER!
if defined EVAL_DIR_CONTAINER echo [EVAL] output=!EVAL_DIR_CONTAINER!
set "CONFIG_POSIX=%CONFIG%"
set "CONFIG_POSIX=%CONFIG_POSIX:\=/%"
set "CMD=python3 -m src.cli.main eval-sft --config '%CONFIG_POSIX%'"
echo [CMD] %CMD%
call :compose up %COMPOSE_NO_BUILD_FLAG% -d %SERVICE%
call :compose exec %SERVICE% bash -lc "!RUN_ENV_EXPORT!%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs eval
set "PIPELINE_STAGE="
goto :eof

:serve
set "PIPELINE_STAGE=serve-vllm"
call :ensure_runtime
if errorlevel 1 (
    set "PIPELINE_STAGE="
    goto :eof
)
if not defined SERVE_PORT set "SERVE_PORT=8080"
set "SERVE_DISPLAY=localhost"
if defined SERVE_HOST (
    if /I not "%SERVE_HOST%"=="0.0.0.0" set "SERVE_DISPLAY=%SERVE_HOST%"
)
echo [SERVE] target=http://%SERVE_DISPLAY%:%SERVE_PORT%
if not defined SERVED_MODEL_PATH (
    echo [ERROR] SERVED_MODEL_PATH not available from runtime metadata.
    exit /b 2
)
set "SERVED_MODEL_PATH=%SERVED_MODEL_PATH%"
set "SERVED_MODEL_NAME=%SERVED_MODEL_NAME%"
set "SERVED_MODEL_MAX_LEN=%SERVED_MODEL_MAX_LEN%"
REM Bring up vLLM along with monitoring and UI
    call :compose up %COMPOSE_NO_BUILD_FLAG% -d %VLLM_SERVICE% dozzle open-webui
if errorlevel 1 goto :eof
echo [INFO] vLLM server exposed on http://%SERVE_DISPLAY%:%SERVE_PORT%
echo [INFO] Dozzle logs UI:      http://localhost:9999
echo [INFO] Open WebUI:          http://localhost:3000
set "PIPELINE_STAGE="
goto :eof

:showlast
if not exist logs\latest.txt (
    echo [WARN] logs/latest.txt not found yet.
    goto :eof
)
type logs\latest.txt
goto :eof

:clean
echo [CLEAN] Stopping project and removing orphans...
call :compose down --remove-orphans
echo [CLEAN] Pruning stopped containers, dangling images, volumes and networks...
docker container prune -f >nul 2>&1
docker image prune -f >nul 2>&1
docker volume prune -f >nul 2>&1
docker system prune -f >nul 2>&1
echo [CLEAN] Docker cleanup completed.
goto :eof

:ensure_runtime
if defined RUNTIME_LOADED goto :eof
call :load_runtime
if errorlevel 1 exit /b %ERRORLEVEL%
set "RUNTIME_LOADED=1"
echo [RUNTIME] COMPOSE_PROJECT=%COMPOSE_PROJECT%
if defined DATASET_NAME echo [RUNTIME] dataset=%DATASET_NAME% sample=%DATASET_SAMPLE_SIZE% base=%BASE_MODEL_NAME%
if defined RUN_NAME echo [RUNTIME] run=!RUN_NAME! (#%RUN_NUMBER%)
if "%DEBUG_PIPELINE%"=="1" echo [RUNTIME] debug pipeline mode enabled
:: Set compose flags depending on debug pipeline setting. When debug is off we
:: avoid triggering image rebuilds during `docker compose up` to suppress
:: verbose build output. The `:build` target still allows explicit builds.
if "%DEBUG_PIPELINE%"=="1" (
    set "COMPOSE_NO_BUILD_FLAG="
) else (
    set "COMPOSE_NO_BUILD_FLAG=--no-build"
)

set "RUN_ENV_EXPORT="
if defined RUN_NAME set "RUN_ENV_EXPORT=!RUN_ENV_EXPORT! RUN_NAME='!RUN_NAME!'"
if defined RUN_NUMBER set "RUN_ENV_EXPORT=!RUN_ENV_EXPORT! RUN_NUMBER='!RUN_NUMBER!'"
if defined RUN_NAME_PREFIX set "RUN_ENV_EXPORT=!RUN_ENV_EXPORT! RUN_NAME_PREFIX='!RUN_NAME_PREFIX!'"
if defined RUN_MODEL_SLUG set "RUN_ENV_EXPORT=!RUN_ENV_EXPORT! RUN_MODEL_SLUG='!RUN_MODEL_SLUG!'"
if defined RUN_DATASET_SLUG set "RUN_ENV_EXPORT=!RUN_ENV_EXPORT! RUN_DATASET_SLUG='!RUN_DATASET_SLUG!'"
if defined RUN_SIZE_SLUG set "RUN_ENV_EXPORT=!RUN_ENV_EXPORT! RUN_SIZE_SLUG='!RUN_SIZE_SLUG!'"
if defined RUN_OUTPUTS_DIR set "RUN_ENV_EXPORT=!RUN_ENV_EXPORT! RUN_OUTPUTS_DIR='!RUN_OUTPUTS_DIR!'"
if defined RUN_LOGS_DIR set "RUN_ENV_EXPORT=!RUN_ENV_EXPORT! RUN_LOGS_DIR='!RUN_LOGS_DIR!'"
if defined RUN_OUTPUTS_DIR_CONTAINER set "RUN_ENV_EXPORT=!RUN_ENV_EXPORT! RUN_OUTPUTS_DIR_CONTAINER='!RUN_OUTPUTS_DIR_CONTAINER!'"
if defined RUN_LOGS_DIR_CONTAINER set "RUN_ENV_EXPORT=!RUN_ENV_EXPORT! RUN_LOGS_DIR_CONTAINER='!RUN_LOGS_DIR_CONTAINER!'"
if defined RUN_DIR_NAME set "RUN_ENV_EXPORT=!RUN_ENV_EXPORT! RUN_DIR_NAME='!RUN_DIR_NAME!'"
if defined RUN_ENV_EXPORT (
    set "RUN_ENV_EXPORT=export!RUN_ENV_EXPORT! && "
)
goto :eof

:load_runtime
if "%DEBUG_PIPELINE%"=="1" echo [TRACE] entering load_runtime
if defined RUNTIME_FILE (
    if exist "%RUNTIME_FILE%" del "%RUNTIME_FILE%" >nul 2>&1
)
set "RUNTIME_FILE=%TEMP%\pipeline_runtime_%RANDOM%_%RANDOM%.env"
set "RUNTIME_RAW=%TEMP%\pipeline_runtime_%RANDOM%_%RANDOM%.raw"
set "CONFIG_POSIX=%CONFIG%"
set "CONFIG_POSIX=%CONFIG_POSIX:\=/%"
call :build_naming_export
set "RUNTIME_STAGE_ARG="
set "RUNTIME_STAGE_ARG_WIN="
if defined PIPELINE_STAGE (
    set "RUNTIME_STAGE_ARG= --stage '%PIPELINE_STAGE%'"
    set "RUNTIME_STAGE_ARG_WIN= --stage \"%PIPELINE_STAGE%\""
)
if "%DEBUG_PIPELINE%"=="1" echo [TRACE] invoking docker runtime probe for config=%CONFIG_POSIX%
call :compose run --rm --no-deps -e "HOST_COMPOSE_PROJECT=%HOST_COMPOSE_PROJECT%" -T %SERVICE% bash -lc "!NAMING_EXPORT!python3 -m src.cli.main print-runtime --config '%CONFIG_POSIX%' --format env!RUNTIME_STAGE_ARG!" >"%RUNTIME_RAW%"
set "STATUS=%ERRORLEVEL%"
if "%DEBUG_PIPELINE%"=="1" echo [TRACE] docker probe exit status=%STATUS%
if %STATUS% EQU 0 goto :runtime_filter

echo [WARN] Docker runtime probe failed (%STATUS%). Falling back to local Python...
python -m src.cli.main print-runtime --config "%CONFIG%" --format env%RUNTIME_STAGE_ARG_WIN% >"%RUNTIME_RAW%"
set "STATUS=%ERRORLEVEL%"
if "%DEBUG_PIPELINE%"=="1" echo [TRACE] local probe exit status=%STATUS%
if %STATUS% NEQ 0 goto :runtime_error

:runtime_filter
if "%DEBUG_PIPELINE%"=="1" echo [TRACE] filtering runtime probe output via python
python -m src.core.runtime_probe "%RUNTIME_RAW%" >"%RUNTIME_FILE%"
set "STATUS=%ERRORLEVEL%"
if %STATUS% NEQ 0 goto :runtime_error

if "%DEBUG_PIPELINE%"=="1" echo [TRACE] applying environment variables from filtered probe
for /f "usebackq tokens=1,* delims==" %%A in ("%RUNTIME_FILE%") do (
    set "%%A=%%B"
)

REM If the host exported a preferred compose project name, force it here so
REM all subsequent docker compose commands use a single, stable project.
if defined HOST_COMPOSE_PROJECT (
    if defined COMPOSE_PROJECT (
        set "HOST_COMPOSE_PROJECT=%COMPOSE_PROJECT%"
    ) else (
        set "COMPOSE_PROJECT=%HOST_COMPOSE_PROJECT%"
    )
) else (
    if not defined COMPOSE_PROJECT set "COMPOSE_PROJECT=sft"
    set "HOST_COMPOSE_PROJECT=%COMPOSE_PROJECT%"
)

if exist "%RUNTIME_FILE%" del "%RUNTIME_FILE%" >nul 2>&1
if exist "%RUNTIME_RAW%" del "%RUNTIME_RAW%" >nul 2>&1
if not defined COMPOSE_PROJECT (
    echo [ERROR] Runtime metadata missing COMPOSE_PROJECT.
    exit /b 2
)
exit /b 0

:runtime_error
if exist "%RUNTIME_FILE%" del "%RUNTIME_FILE%" >nul 2>&1
if exist "%RUNTIME_RAW%" del "%RUNTIME_RAW%" >nul 2>&1
exit /b %STATUS%

:build_naming_export
set "NAMING_EXPORT="
REM Derive a stable host project name to export so in-container probes return
REM the same compose project as the host. If the caller already set
REM HOST_COMPOSE_PROJECT, keep it; otherwise fall back to COMPOSE_PROJECT_NAME
REM or the SFT default.
if not defined HOST_COMPOSE_PROJECT (
    if defined COMPOSE_PROJECT_NAME (
        set "HOST_COMPOSE_PROJECT=%COMPOSE_PROJECT_NAME%"
    )
)
if not defined HOST_COMPOSE_PROJECT (
    set "HOST_COMPOSE_PROJECT=sft"
)
for %%V in (RUN_NAME FORCE_RUN_NAME RUN_INDEX FORCE_RUN_INDEX LEGACY_RUN_NAME USE_LEGACY_NAMING) do (
    if defined %%V (
        set "NAMING_EXPORT=!NAMING_EXPORT! %%V='!%%V!'"
    )
)
if defined NAMING_EXPORT (
    set "NAMING_EXPORT=export!NAMING_EXPORT! && "
)
exit /b 0

:ensure_host_compose_project
if defined COMPOSE_PROJECT goto :eof
if defined HOST_COMPOSE_PROJECT (
    set "COMPOSE_PROJECT=%HOST_COMPOSE_PROJECT%"
    goto :eof
)
for /f "usebackq tokens=* delims=" %%P in (`python -c "from pathlib import Path; name = Path(r'%_SCRIPT_DIR%').name.lower(); name = (name[:62] if name else '').rstrip('-'); name = name or 'pipeline'; name = name if name[0].isalpha() else (('proj-' + name)[:62].rstrip('-') or 'pipeline'); print(name)"`) do (
    set "COMPOSE_PROJECT=%%P"
)
if not defined COMPOSE_PROJECT set "COMPOSE_PROJECT=pipeline"
set "HOST_COMPOSE_PROJECT=%COMPOSE_PROJECT%"
exit /b 0

:run_post_stage
setlocal EnableExtensions EnableDelayedExpansion
set "NEXT=%~1"
if /I "!NEXT!"=="export-merged" (
    endlocal
    call :export
    exit /b %ERRORLEVEL%
)
if /I "!NEXT!"=="convert-awq" (
    endlocal
    call :convert_awq
    exit /b %ERRORLEVEL%
)
if /I "!NEXT!"=="eval-sft" (
    endlocal
    call :evaluate
    exit /b %ERRORLEVEL%
)
if /I "!NEXT!"=="serve-vllm" (
    endlocal
    call :serve
    exit /b %ERRORLEVEL%
)
echo [WARN] Unknown post-finetune target: !NEXT!
endlocal
exit /b 0

:compose
setlocal EnableExtensions EnableDelayedExpansion
set "__PROJECT=%COMPOSE_PROJECT%"
if not defined __PROJECT set "__PROJECT=%HOST_COMPOSE_PROJECT%"
if not defined __PROJECT set "__PROJECT=sft"
docker compose -p "!__PROJECT!" %*
set "__STATUS=%ERRORLEVEL%"
endlocal & exit /b %__STATUS%

:run_awq_cli
setlocal EnableExtensions EnableDelayedExpansion
set "CONFIG_ABS=%CONFIG%"
for %%I in ("%CONFIG%") do set "CONFIG_ABS=%%~fI"
set "CONFIG_REL=!CONFIG_ABS:%_SCRIPT_DIR%=%!"
if "!CONFIG_REL!"=="!CONFIG_ABS!" set "CONFIG_REL=%CONFIG%"
set "CONFIG_REL=!CONFIG_REL:\=/!"
set "CONFIG_REL=!CONFIG_REL:/./=!"   
if "%CONFIG_REL%"=="" set "CONFIG_REL=config.yaml"
set "CONFIG_CONTAINER=/workspace/%CONFIG_REL%"

set "CLI_BASE=python3 -m src.cli.awq_entry --config '!CONFIG_CONTAINER!'"
if defined FORCE_AWQ set "CLI_BASE=!CLI_BASE! --force"
if defined RUN_NAME set "CLI_BASE=!CLI_BASE! --run-name '!RUN_NAME!'"
if defined RUN_NUMBER set "CLI_BASE=!CLI_BASE! --run-index !RUN_NUMBER!"

if "%DEBUG_PIPELINE%"=="1" echo [TRACE] AWQ container command: !CLI_BASE!
echo [CMD] docker compose run --rm --no-deps -T awq-runner bash -lc "!RUN_ENV_EXPORT!!CLI_BASE!"
call :compose run --rm --no-deps -T awq-runner bash -lc "!RUN_ENV_EXPORT!!CLI_BASE!"
set "__STATUS=%ERRORLEVEL%"
endlocal & exit /b %__STATUS%

:capture_container_logs
setlocal EnableExtensions EnableDelayedExpansion
set "STAGE=%~1"
set "TARGET_SERVICE=%~2"
if not defined TARGET_SERVICE set "TARGET_SERVICE=%SERVICE%"
if not exist logs\latest.txt goto :capture_done
set "LATEST="
for /f "usebackq tokens=* delims=" %%L in ("logs\latest.txt") do set "LATEST=%%L"
if not defined LATEST goto :capture_done
set "HOST_PATH=!LATEST:/app/=!"
set "HOST_PATH=!HOST_PATH:/=\!"
if "!HOST_PATH:~0,1!"=="\" set "HOST_PATH=!HOST_PATH:~1!"
if not defined HOST_PATH goto :capture_done
if not exist "!HOST_PATH!" goto :capture_done
set "STAGE_PATH=!HOST_PATH!\%STAGE%"
if not exist "!STAGE_PATH!" mkdir "!STAGE_PATH!" >nul 2>&1
set "TARGET_FILE=!STAGE_PATH!\container.log"
echo [LOGS] Capturing docker logs for %STAGE% into !TARGET_FILE!
call :compose logs --no-color --timestamps --tail 2000 !TARGET_SERVICE! >"!TARGET_FILE!" 2>&1
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
echo    convert-awq          Run AWQ conversion using the awq-runner sidecar
echo    tensorboard-up       Start TensorBoard (port 6006)
echo    tensorboard-down     Stop TensorBoard service
echo    tensorboard-logs     Tail TensorBoard container logs
echo    show-last            Print path to latest log folder
echo    clean|prune          Stop project, remove orphans and prune unused Docker resources
echo.
echo Example:
echo    run_pipeline.bat preprocess-sft CONFIG=config.prod.yaml
exit /b 1

:: Manual convert-awq target removed (migrating AWQ to llm-compressor)

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
