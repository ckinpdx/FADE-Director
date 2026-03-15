@echo off
REM FADE Director — development launcher
REM Requires: uv (https://github.com/astral-sh/uv) and Node.js

cd /d "%~dp0"

if not exist .env (
    echo WARNING: .env not found. Copy .env.example to .env and fill in your paths.
)

REM ── 1. llama-swap ────────────────────────────────────────────────────────────
echo Checking llama-swap (port 8000)...
powershell -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',8000)).Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo Starting llama-swap...
    schtasks /run /tn "\openclaw\llama-swap" >nul 2>&1
    echo Waiting for llama-swap to be ready...
    :wait_llm
    powershell -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',8000)).Close(); exit 0 } catch { exit 1 }" >nul 2>&1
    if errorlevel 1 (
        timeout /t 1 /nobreak >nul
        goto wait_llm
    )
    echo llama-swap ready.
) else (
    echo llama-swap already running.
)

REM ── 2. Python venv (uv) ───────────────────────────────────────────────────────
if not exist .venv (
    echo Creating Python environment...
    uv venv
    if errorlevel 1 (
        echo ERROR: uv venv failed. Is uv installed? https://github.com/astral-sh/uv
        pause
        exit /b 1
    )
)
echo Installing/syncing Python dependencies...
uv pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

REM ── 3. Frontend ───────────────────────────────────────────────────────────────
echo Building frontend...
cd /d "%~dp0frontend"
if not exist node_modules (
    echo node_modules not found — running npm install...
    call npm install
    if errorlevel 1 (
        echo ERROR: npm install failed.
        pause
        exit /b 1
    )
)
call npm run build
if errorlevel 1 (
    echo ERROR: Frontend build failed.
    pause
    exit /b 1
)
cd /d "%~dp0"

REM ── 4. Kill any stale backend on port 8001 ───────────────────────────────────
FOR /F "tokens=5" %%P IN ('netstat -ano ^| findstr ":8001 " ^| findstr "LISTENING"') DO (
    powershell -Command "Stop-Process -Id %%P -Force -ErrorAction SilentlyContinue" 2>nul
)

REM ── 5. Open browser after server starts ──────────────────────────────────────
echo.
echo  FADE Director
echo  Open in browser: http://localhost:8001
echo.
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8001"

REM ── 6. Start backend (inside uv venv) ────────────────────────────────────────
uv run uvicorn backend.main:app ^
    --host 0.0.0.0 ^
    --port 8001 ^
    --log-level debug ^
    --reload ^
    --reload-dir backend
