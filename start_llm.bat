@echo off
REM FADE Director — LLM server launcher
REM Starts llama-swap using tools\llama-swap.exe (built by scripts\build_llama.bat)
REM and llama-swap.yaml in the repo root (copy from llama-swap-config.example.yaml).
REM
REM If you use a different LLM server setup, replace this file with whatever
REM starts your OpenAI-compatible server on port 8000.

echo Checking LLM server (port 8000)...
powershell -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',8000)).Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo LLM server already running.
    exit /b 0
)

set ROOT=%~dp0
set SWAP_EXE=%ROOT%tools\llama-swap.exe
set SWAP_CFG=%ROOT%llama-swap.yaml

if not exist "%SWAP_EXE%" (
    echo ERROR: tools\llama-swap.exe not found.
    echo Run scripts\build_llama.bat to build it, or replace this file with your own server start command.
    exit /b 1
)

if not exist "%SWAP_CFG%" (
    echo ERROR: llama-swap.yaml not found.
    echo Copy llama-swap-config.example.yaml to llama-swap.yaml and fill in your model paths.
    exit /b 1
)

echo Starting llama-swap...
start /b "" "%SWAP_EXE%" --config "%SWAP_CFG%"

echo Waiting for LLM server to be ready...
:wait_llm
powershell -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',8000)).Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_llm
)
echo LLM server ready.
