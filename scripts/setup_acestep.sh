#!/usr/bin/env bash
# Creates a dedicated Python venv for ACEStep and installs ace-step into it.
# This venv is separate from FADE's venv — ace-step's dependencies are isolated.
# Run once before using the "Make a Song" page.

set -e
cd "$(dirname "$0")/.."

VENV_DIR="acestep_venv"

if [ -f "$VENV_DIR/bin/python" ]; then
    echo "ACEStep venv already exists at $VENV_DIR/ — nothing to do."
    echo "To reinstall, delete the $VENV_DIR folder and run this script again."
    exit 0
fi

echo "Creating ACEStep Python environment at $VENV_DIR/..."
python3 -m venv "$VENV_DIR"

echo "Installing ace-step 1.5 into $VENV_DIR/..."
"$VENV_DIR/bin/pip" install "git+https://github.com/ace-step/ACE-Step-1.5.git"

echo "Installing server dependencies into $VENV_DIR/..."
"$VENV_DIR/bin/pip" install fastapi uvicorn av torchcodec

echo ""
echo "ACEStep environment ready at $VENV_DIR/"
echo "You can now use the \"Make a Song\" page in FADE Director."
