@echo off
REM FADE Director — production launcher (assumes venv + node_modules already set up)
REM Run run_dev.bat once first to install dependencies.

cd /d "%~dp0"

SET PATH=%LOCALAPPDATA%\Programs\uv;%PATH%

if not exist .env (
    echo WARNING: .env not found. Copy .env.example to .env and fill in your paths.
)

REM ── 1. Sanity checks ─────────────────────────────────────────────────────────
if not exist .venv (
    echo ERROR: .venv not found. Run run_dev.bat once to set up the environment.
    pause
    exit /b 1
)
if not exist frontend\node_modules (
    echo ERROR: frontend\node_modules not found. Run run_dev.bat once to set up the environment.
    pause
    exit /b 1
)

REM ── 2. LLM server ────────────────────────────────────────────────────────────
call "%~dp0start_llm.bat"
powershell -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',8000)).Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo ERROR: LLM server not reachable on port 8000. Check start_llm.bat.
    pause
    exit /b 1
)

REM ── 3. Build frontend ─────────────────────────────────────────────────────────
echo Building frontend...
cd /d "%~dp0frontend"
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

REM ── 6. Start backend ─────────────────────────────────────────────────────────
uv run uvicorn backend.main:app ^
    --host 0.0.0.0 ^
    --port 8001 ^
    --log-level info
