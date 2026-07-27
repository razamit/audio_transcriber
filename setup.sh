#!/bin/bash
# Recreate the local environment for transcribe.py.
# Usage:  ./setup.sh
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required to decode audio. Install it with:  brew install ffmpeg"
    exit 1
fi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt

echo
echo "Done. Run the transcriber with:"
echo "    .venv/bin/python transcribe.py"
