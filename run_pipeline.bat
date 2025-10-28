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

set "CONFIG=config.yaml"
set "SERVICE=sft"
set "VLLM_SERVICE=vllm-server"
set "RUNTIME_LOADED="
set "RUNTIME_FILE="
set "COMPOSE_PROJECT="
set "SERVED_MODEL_PATH="
set "SERVED_MODEL_NAME="
set "SERVED_MODEL_MAX_LEN="
set "SERVE_PORT="

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
if /I "%TARGET%"=="clean"           goto :clean
if /I "%TARGET%"=="prune"           goto :clean
goto :help

REM ============================================================================
REM ============================= Commands =====================================
REM ============================================================================
:build
REM Build should not require runtime metadata; call docker compose directly
docker compose build
goto :eof

:up
call :ensure_runtime
if errorlevel 1 goto :eof
call :compose up -d %SERVICE%
goto :eof

:down
call :ensure_runtime
if errorlevel 1 goto :eof
call :compose down --remove-orphans
goto :eof

:bash
call :ensure_runtime
if errorlevel 1 goto :eof
call :compose exec %SERVICE% bash
goto :eof

:preprocess
call :ensure_runtime
if errorlevel 1 goto :eof
if defined PREPROCESS_DIR_CONTAINER echo [PREPROCESS] output=!PREPROCESS_DIR_CONTAINER!
set "CONFIG_POSIX=%CONFIG%"
set "CONFIG_POSIX=%CONFIG_POSIX:\=/%"
set "CMD=python3 -m src.cli.main preprocess-sft --config '%CONFIG_POSIX%'"
echo [CMD] %CMD%
call :compose up -d %SERVICE%
call :compose exec %SERVICE% bash -lc "%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs preprocess
goto :eof

:finetune
call :ensure_runtime
if errorlevel 1 goto :eof
if defined PREPROCESS_DIR_CONTAINER echo [FINETUNE] data=!PREPROCESS_DIR_CONTAINER!
if defined OUTPUT_DIR_CONTAINER echo [FINETUNE] output=!OUTPUT_DIR_CONTAINER!
set "CONFIG_POSIX=%CONFIG%"
set "CONFIG_POSIX=%CONFIG_POSIX:\=/%"
set "CMD=python3 -m src.cli.main finetune-sft --config '%CONFIG_POSIX%'"
echo [CMD] %CMD%
call :compose up -d %SERVICE%
call :compose exec %SERVICE% bash -lc "%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs finetune
goto :eof

:export
call :ensure_runtime
if errorlevel 1 goto :eof
if defined MERGED_DIR_CONTAINER echo [EXPORT] merged=!MERGED_DIR_CONTAINER!
set "CONFIG_POSIX=%CONFIG%"
set "CONFIG_POSIX=%CONFIG_POSIX:\=/%"
set "CMD=python3 -m src.cli.main export-merged --config '%CONFIG_POSIX%'"
echo [CMD] %CMD%
call :compose up -d %SERVICE%
call :compose exec %SERVICE% bash -lc "%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs export
goto :eof

:evaluate
call :ensure_runtime
if errorlevel 1 goto :eof
if defined MERGED_DIR_CONTAINER echo [EVAL] model=!MERGED_DIR_CONTAINER!
if defined EVAL_DIR_CONTAINER echo [EVAL] output=!EVAL_DIR_CONTAINER!
set "CONFIG_POSIX=%CONFIG%"
set "CONFIG_POSIX=%CONFIG_POSIX:\=/%"
set "CMD=python3 -m src.cli.main eval-sft --config '%CONFIG_POSIX%'"
echo [CMD] %CMD%
call :compose up -d %SERVICE%
call :compose exec %SERVICE% bash -lc "%CMD%"
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
call :capture_container_logs eval
goto :eof

:serve
call :ensure_runtime
if errorlevel 1 goto :eof
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
call :compose up -d %VLLM_SERVICE% dozzle open-webui
if errorlevel 1 goto :eof
echo [INFO] vLLM server exposed on http://%SERVE_DISPLAY%:%SERVE_PORT%
echo [INFO] Dozzle logs UI:      http://localhost:9999
echo [INFO] Open WebUI:          http://localhost:3000
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
docker compose down --remove-orphans
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
goto :eof

:load_runtime
if defined RUNTIME_FILE (
    if exist "%RUNTIME_FILE%" del "%RUNTIME_FILE%" >nul 2>&1
)
set "RUNTIME_FILE=%TEMP%\pipeline_runtime_%RANDOM%_%RANDOM%.tmp"
set "RUNTIME_RAW=%RUNTIME_FILE%.raw"
REM Compute runtime metadata inside Docker first to avoid local Python dependency
set "CONFIG_POSIX=%CONFIG%"
set "CONFIG_POSIX=%CONFIG_POSIX:\=/%"
set "PRINT_RUNTIME_CMD=python3 -m src.cli.main print-runtime --config '%CONFIG_POSIX%' --format env"
if defined DEBUG_PIPELINE (
    echo [TRACE] debug mode: filtering runtime output inside container
    set "PRINT_RUNTIME_CMD=python3 -m src.cli.main print-runtime --config '%CONFIG_POSIX%' --format env | grep -E '^[A-Z][A-Z0-9_]*='"
)
docker compose run --rm --no-deps -T sft bash -lc "%PRINT_RUNTIME_CMD%" >"%RUNTIME_RAW%"
set "STATUS=%ERRORLEVEL%"
if defined DEBUG_PIPELINE echo [DEBUG] docker probe status=%STATUS%
if defined DEBUG_PIPELINE echo [TRACE] after docker probe
if "%STATUS%"=="0" goto :after_runtime_probe

echo [RUNTIME] Docker runtime probe failed (%STATUS%). Falling back to local Python...
python -m src.cli.main print-runtime --config "%CONFIG%" --format env >"%RUNTIME_RAW%"
set "STATUS=%ERRORLEVEL%"
if defined DEBUG_PIPELINE echo [DEBUG] local probe status=%STATUS%
if not "%STATUS%"=="0" (
    if exist "%RUNTIME_RAW%" del "%RUNTIME_RAW%" >nul 2>&1
    if exist "%RUNTIME_FILE%" del "%RUNTIME_FILE%" >nul 2>&1
    exit /b %STATUS%
)

:after_runtime_probe
REM DEBUG: temporarily disable runtime raw dump to isolate issue
REM if defined DEBUG_PIPELINE call :debug_dump "runtime raw" "%RUNTIME_RAW%"
set "FILTER_EXIT="
if defined DEBUG_PIPELINE echo [TRACE] before findstr
findstr /R "^[A-Z][A-Z0-9_]*=" "%RUNTIME_RAW%" >"%RUNTIME_FILE%" 2>nul
set "FILTER_EXIT=%ERRORLEVEL%"
if defined DEBUG_PIPELINE echo [DEBUG] findstr errorlevel=%FILTER_EXIT%
if defined DEBUG_PIPELINE echo [TRACE] before filtered dump
if not "%FILTER_EXIT%"=="0" (
    if defined DEBUG_PIPELINE call :debug_dump "runtime raw (filter failed)" "%RUNTIME_RAW%"
)
if defined DEBUG_PIPELINE call :debug_dump "runtime filtered" "%RUNTIME_FILE%"
if defined DEBUG_PIPELINE echo [TRACE] before for-loop
for /f "usebackq tokens=* delims=" %%L in ("%RUNTIME_FILE%") do (
    if not "%%L"=="" (
        if defined DEBUG_PIPELINE echo [DEBUG] raw line: %%L
        echo(%%L| findstr /R "^[A-Z][A-Z0-9_]*=" >nul
        if not errorlevel 1 for /f "tokens=1,* delims==" %%A in ("%%L") do (
            set "%%A=%%B"
        )
    )
)
if exist "%RUNTIME_FILE%" del "%RUNTIME_FILE%" >nul 2>&1
if exist "%RUNTIME_RAW%" del "%RUNTIME_RAW%" >nul 2>&1
if not defined COMPOSE_PROJECT (
    echo [ERROR] Runtime metadata missing COMPOSE_PROJECT.
    exit /b 2
)
exit /b 0

:debug_dump
setlocal EnableExtensions EnableDelayedExpansion
set "_DESC=%~1"
set "_FILE=%~2"
echo [DEBUG] %_DESC% path=%_FILE%
if exist "%_FILE%" (
    for %%F in ("%_FILE%") do echo [DEBUG] %_DESC% size=%%~zF
    echo [DEBUG] %_DESC% content begin
    type "%_FILE%"
    echo [DEBUG] %_DESC% content end
) else (
    echo [DEBUG] %_DESC% missing
)
endlocal
goto :eof

:compose
if defined COMPOSE_PROJECT (
    docker compose -p "%COMPOSE_PROJECT%" %*
)
if not defined COMPOSE_PROJECT (
    docker compose %*
)
exit /b %ERRORLEVEL%

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
call :compose logs --no-color --timestamps --tail 2000 %SERVICE% >"!TARGET_FILE!" 2>&1
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
echo    clean|prune          Stop project, remove orphans and prune unused Docker resources
echo.
echo Example:
echo    run_pipeline.bat preprocess-sft CONFIG=config.prod.yaml
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
