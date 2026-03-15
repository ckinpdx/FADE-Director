@echo off
REM Music Director — start the backend server (INFO logging)
REM Reads settings from .env if present

cd /d "%~dp0"

if not exist .env (
    echo WARNING: .env not found. Copy .env.example to .env and fill in your paths.
)

python -m uvicorn backend.main:app ^
    --host 0.0.0.0 ^
    --port 8001 ^
    --log-level info
