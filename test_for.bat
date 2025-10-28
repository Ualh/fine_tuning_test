@echo off
setlocal EnableDelayedExpansion
set "_SOURCE=%TEMP%\manual_runtime.tmp"
echo reading !_SOURCE!
if not exist "!_SOURCE!" (
    echo source missing
    exit /b 1
)
echo --- file content via type ---
type "!_SOURCE!"
echo --- end content ---
for /f "usebackq tokens=* delims=" %%L in ("!_SOURCE!") do (
    echo line=%%L
)
