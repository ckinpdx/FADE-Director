@echo off
REM Creates a dedicated Python venv for ACEStep 1.5, installs ace-step into it,
REM then downloads all required model checkpoints with visible progress.
REM Run once before using the "Make a Song" page.
REM
REM Output is echoed to the console AND written to setup_acestep.log in the FADE-Director root.

setlocal
cd /d "%~dp0.."

set VENV_DIR=acestep_venv
set CHECKPOINT_DIR=checkpoints
set LOG=setup_acestep.log
set SCRIPT_DIR=%~dp0

echo. > "%LOG%"
call :log "=== ACEStep 1.5 setup started ==="

REM ── STEP 1: Install Python venv + packages ────────────────────────────────

if exist "%VENV_DIR%\Lib\site-packages\acestep\__init__.py" (
    call :log "ACEStep 1.5 package already installed — skipping venv setup."
    goto :download_models
)

REM ── Create venv if it doesn't exist ───────────────────────────────────────
if not exist "%VENV_DIR%\Scripts\python.exe" (
    call :log "Creating Python 3.11 venv at %VENV_DIR%\..."
    set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
    if not exist "%PYTHON_EXE%" (
        call :log "Falling back to py -3.11 launcher..."
        py -3.11 -m venv "%VENV_DIR%"
    ) else (
        "%PYTHON_EXE%" -m venv "%VENV_DIR%"
    )
    if errorlevel 1 (
        call :log "ERROR: venv creation failed. Make sure Python 3.11 is installed."
        goto :fail
    )
    call :log "Venv created."
) else (
    call :log "Venv already exists — skipping creation."
)

REM ── Install PyTorch CUDA 12.8 if not present ──────────────────────────────
if not exist "%VENV_DIR%\Lib\site-packages\torch\__init__.py" (
    call :log "Installing PyTorch with CUDA 12.8 support..."
    "%VENV_DIR%\Scripts\pip.exe" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 >> "%LOG%" 2>&1
    if errorlevel 1 (
        call :log "ERROR: pip install torch ^(CUDA^) failed. See %LOG% for details."
        goto :fail
    )
    call :log "PyTorch installed."
) else (
    call :log "PyTorch already installed — skipping."
)

REM ── Install ACE-Step 1.5 ──────────────────────────────────────────────────
REM nano-vllm has no Windows wheels — we use backend="pt" so it isn't needed.
REM Clone the repo, strip the nano-vllm dependency, then install from local clone.
call :log "Cloning ACE-Step 1.5 repository..."
set CLONE_DIR=%TEMP%\ace-step-15-src
if exist "%CLONE_DIR%" rmdir /s /q "%CLONE_DIR%"
git clone --quiet https://github.com/ace-step/ACE-Step-1.5.git "%CLONE_DIR%" >> "%LOG%" 2>&1
if errorlevel 1 (
    call :log "ERROR: git clone failed. Make sure git is installed and accessible."
    goto :fail
)
call :log "Patching pyproject.toml (removing nano-vllm + pinned torch version)..."
powershell -Command "(Get-Content '%CLONE_DIR%\pyproject.toml') | Where-Object {$_ -notmatch 'nano-vllm|torch==|torchaudio==|torchvision=='} | Set-Content '%CLONE_DIR%\pyproject.toml'" >> "%LOG%" 2>&1
if errorlevel 1 (
    call :log "ERROR: Failed to patch pyproject.toml."
    goto :fail
)
call :log "Installing ace-step 1.5 from local clone (this may take several minutes)..."
set PYTHONUTF8=1
"%VENV_DIR%\Scripts\pip.exe" install "%CLONE_DIR%" >> "%LOG%" 2>&1
if errorlevel 1 (
    call :log "ERROR: pip install ace-step 1.5 failed. See %LOG% for details."
    goto :fail
)
rmdir /s /q "%CLONE_DIR%"
call :log "ACE-Step 1.5 installed."

REM ── Install server dependencies ───────────────────────────────────────────
call :log "Installing server dependencies ^(fastapi uvicorn av torchcodec^)..."
"%VENV_DIR%\Scripts\pip.exe" install fastapi uvicorn av torchcodec >> "%LOG%" 2>&1
if errorlevel 1 (
    call :log "ERROR: pip install server deps failed. See %LOG% for details."
    goto :fail
)
call :log "Server dependencies installed."

REM ── Install nano-vllm + Triton (enables vLLM backend on Windows) ──────────
REM nano-vllm is bundled inside the ace-step package but not installed as a
REM Python package. We install it from its bundled path. It pulls in
REM triton-windows and flash-attn as dependencies automatically.
call :log "Installing nano-vllm ^(vLLM backend^) + triton-windows + flash-attn..."
for /f "delims=" %%P in ('"%VENV_DIR%\Scripts\python.exe" -c "import acestep, os; print(os.path.join(os.path.dirname(acestep.__file__), 'third_parts', 'nano-vllm'))"') do set NANO_VLLM_DIR=%%P
"%VENV_DIR%\Scripts\pip.exe" install "%NANO_VLLM_DIR%" >> "%LOG%" 2>&1
if errorlevel 1 (
    call :log "ERROR: nano-vllm install failed. See %LOG% for details."
    goto :fail
)
call :log "nano-vllm + triton-windows installed — vLLM backend enabled."

REM ── STEP 2: Download model checkpoints ───────────────────────────────────
:download_models
call :log ""
call :log "=== Downloading model checkpoints ==="
call :log "Target: %CD%\%CHECKPOINT_DIR%"
call :log "Progress is shown live below."
call :log ""

"%VENV_DIR%\Scripts\python.exe" "%SCRIPT_DIR%download_acestep_models.py" "%CHECKPOINT_DIR%"
if errorlevel 1 (
    call :log ""
    call :log "ERROR: Model download failed. Check output above for details."
    goto :fail
)

:done
call :log ""
call :log "=== ACEStep 1.5 setup complete ==="
call :log "Venv:        %CD%\%VENV_DIR%\"
call :log "Checkpoints: %CD%\%CHECKPOINT_DIR%\"
call :log "Full install log: %CD%\%LOG%"
echo.
pause
exit /b 0

:fail
call :log ""
call :log "=== Setup FAILED — check output above and %CD%\%LOG% ==="
echo.
pause
exit /b 1

REM ── Logging helper (console + file) ──────────────────────────────────────
:log
echo %~1
echo %~1 >> "%LOG%"
exit /b 0
