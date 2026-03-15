@echo off
REM Music Director — development launcher
REM Starts llama-swap, waits for it, then starts the backend server.

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

REM ── 2. LTX-Desktop — started on-demand by FADE when video generation begins ──
REM      Do NOT start here: PyTorch CUDA context (~1-2GB) would occupy VRAM during
REM      analysis and LLM phases, reducing headroom for llama-swap models.
echo LTX-Desktop will start automatically when video generation is requested.

REM ── 3. Build frontend ────────────────────────────────────────────────────────
echo Building frontend...
cd /d "%~dp0frontend"
call npm run build
if errorlevel 1 (
    echo ERROR: Frontend build failed.
    pause
    exit /b 1
)
cd /d "%~dp0"

REM ── 4. Kill any stale backend on port 8001 ────────────────────────────────────
FOR /F "tokens=5" %%P IN ('netstat -ano ^| findstr ":8001 " ^| findstr "LISTENING"') DO (
    powershell -Command "Stop-Process -Id %%P -Force -ErrorAction SilentlyContinue" 2>nul
)

REM ── 5. Open browser after server starts ──────────────────────────────────────
echo.
echo  Music Director
echo  Open in browser: http://localhost:8001
echo.
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8001"

REM ── 6. Start backend ─────────────────────────────────────────────────────────
python -m uvicorn backend.main:app ^
    --host 0.0.0.0 ^
    --port 8001 ^
    --log-level debug ^
    --reload ^
    --reload-dir backend
