@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: build_llama.bat
:: Build llama-server (llama.cpp) and llama-swap from source.
::
:: Skips any component already found on PATH or in tools\.
:: Outputs: tools\llama-server.exe  and  tools\llama-swap.exe
::
:: Usage:
::   scripts\build_llama.bat           -- auto-detect GPU, build both
::   scripts\build_llama.bat 89        -- override CUDA arch (89 = Ada / RTX 4090, 86 = Ampere / RTX 3090)
::   scripts\build_llama.bat --clean   -- delete tools\build\ source/build tree and exit
:: ============================================================================

:: ── Clean flag ───────────────────────────────────────────────────────────────
set ROOT=%~dp0..
set TOOLS=%ROOT%\tools
set BUILD_TMP=%TOOLS%\build

if /i "%~1"=="--clean" (
    if exist "%BUILD_TMP%" (
        echo Removing %BUILD_TMP% ...
        rd /s /q "%BUILD_TMP%"
        echo Done.
    ) else (
        echo Nothing to clean — %BUILD_TMP% does not exist.
    )
    exit /b 0
)

:: Auto-detect GPU compute capability via nvidia-smi (e.g. "12.0" -> "120")
set CUDA_ARCH=
if "%~1"=="" (
    for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu=compute_cap --format=csv^,noheader 2^>nul') do set _MAJOR=%%a
    for /f "tokens=2 delims=." %%b in ('nvidia-smi --query-gpu=compute_cap --format=csv^,noheader 2^>nul') do set _MINOR=%%b
    if defined _MAJOR if defined _MINOR set CUDA_ARCH=!_MAJOR!!_MINOR!
)
if not "%~1"=="" set CUDA_ARCH=%~1
if not defined CUDA_ARCH (
    echo [WARN] Could not detect GPU compute capability — defaulting to 120 (Blackwell).
    echo        Override with: scripts\build_llama.bat 89
    set CUDA_ARCH=120
)

echo.
echo FADE llama.cpp + llama-swap builder
echo CUDA architecture: %CUDA_ARCH%
echo Output directory:  %TOOLS%
echo.

:: ── Prerequisite checks ─────────────────────────────────────────────────────

echo Checking prerequisites...

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git not found. Install Git for Windows: https://git-scm.com/
    exit /b 1
)
echo   git        OK

where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cmake not found. Install CMake: https://cmake.org/download/
    echo         Or install Visual Studio with C++ CMake tools.
    exit /b 1
)
echo   cmake      OK

nvcc --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvcc not found. Install CUDA Toolkit 12.6+: https://developer.nvidia.com/cuda-downloads
    exit /b 1
)
for /f "tokens=5 delims= " %%v in ('nvcc --version ^| findstr "release"') do set CUDA_VER=%%v
echo   nvcc       OK  (CUDA %CUDA_VER%)

where go >nul 2>&1
if errorlevel 1 (
    echo [WARN] go not found — llama-swap will be skipped.
    echo        Install Go: https://go.dev/dl/
    set SKIP_SWAP=1
) else (
    echo   go         OK
    set SKIP_SWAP=0
)

echo.
mkdir "%TOOLS%" 2>nul

:: ── llama-server ─────────────────────────────────────────────────────────────

set SERVER_FOUND=0

:: Check tools\ first
if exist "%TOOLS%\llama-server.exe" (
    echo [SKIP] llama-server found at tools\llama-server.exe
    set SERVER_FOUND=1
)

:: Check PATH
if !SERVER_FOUND!==0 (
    where llama-server >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%p in ('where llama-server') do echo [SKIP] llama-server found on PATH: %%p
        set SERVER_FOUND=1
    )
)

if !SERVER_FOUND!==0 (
    echo Building llama-server from source...
    echo   Cloning llama.cpp...
    if not exist "%BUILD_TMP%\llama.cpp" (
        git clone --depth 1 https://github.com/ggerganov/llama.cpp "%BUILD_TMP%\llama.cpp"
        if errorlevel 1 ( echo [ERROR] Clone failed. && exit /b 1 )
    ) else (
        echo   Using existing clone at build\llama.cpp
    )

    echo   Configuring CMake (CUDA arch %CUDA_ARCH%)...
    cmake -B "%BUILD_TMP%\llama.cpp\build" ^
          -S "%BUILD_TMP%\llama.cpp" ^
          -DGGML_CUDA=ON ^
          -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% ^
          -DLLAMA_BUILD_TESTS=OFF ^
          -DLLAMA_BUILD_EXAMPLES=OFF ^
          -DLLAMA_BUILD_SERVER=ON
    if errorlevel 1 ( echo [ERROR] CMake configure failed. && exit /b 1 )

    echo   Building (this takes a few minutes)...
    cmake --build "%BUILD_TMP%\llama.cpp\build" --config Release -j
    if errorlevel 1 ( echo [ERROR] Build failed. && exit /b 1 )

    copy "%BUILD_TMP%\llama.cpp\build\bin\Release\llama-server.exe" "%TOOLS%\llama-server.exe" >nul
    if errorlevel 1 (
        echo [ERROR] Could not copy llama-server.exe — check build output above.
        exit /b 1
    )
    echo   Built: tools\llama-server.exe
)

:: ── llama-swap ───────────────────────────────────────────────────────────────

if !SKIP_SWAP!==1 goto :skip_swap

set SWAP_FOUND=0

if exist "%TOOLS%\llama-swap.exe" (
    echo [SKIP] llama-swap found at tools\llama-swap.exe
    set SWAP_FOUND=1
)

if !SWAP_FOUND!==0 (
    where llama-swap >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%p in ('where llama-swap') do echo [SKIP] llama-swap found on PATH: %%p
        set SWAP_FOUND=1
    )
)

if !SWAP_FOUND!==0 (
    echo Building llama-swap from source...
    if not exist "%BUILD_TMP%\llama-swap" (
        git clone --depth 1 https://github.com/mostlygeek/llama-swap "%BUILD_TMP%\llama-swap"
        if errorlevel 1 ( echo [ERROR] Clone failed. && exit /b 1 )
    ) else (
        echo   Using existing clone at build\llama-swap
    )

    echo   Building...
    pushd "%BUILD_TMP%\llama-swap"
    go build -o "%TOOLS%\llama-swap.exe" ./cmd/server
    if errorlevel 1 ( popd && echo [ERROR] go build failed. && exit /b 1 )
    popd
    echo   Built: tools\llama-swap.exe
)

:skip_swap

:: ── Clean build tree ─────────────────────────────────────────────────────────
if exist "%BUILD_TMP%" (
    echo Cleaning build tree...
    rd /s /q "%BUILD_TMP%"
)

:: ── Summary ──────────────────────────────────────────────────────────────────

echo.
echo Done.
if exist "%TOOLS%\llama-server.exe" echo   tools\llama-server.exe
if exist "%TOOLS%\llama-swap.exe"   echo   tools\llama-swap.exe
echo.
echo Next steps:
echo   1. Add tools\ to your PATH, or reference the binaries directly in your
echo      llama-swap config (set "binary" to the full path of llama-server.exe).
echo   2. Configure llama-swap with your model paths and launch flags.
echo      See README.md for an example config.
echo   3. Start llama-swap before running the FADE backend.
echo.

endlocal
