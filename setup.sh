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

if [ "${1:-}" = "--with-diarization" ]; then
    echo
    echo "Installing speaker diarization extras (~2GB) ..."
    .venv/bin/pip install -r requirements-diarize.txt
    echo
    echo "Diarization installed. No account or token is needed; the model"
    echo "downloads on first run. Then run:"
    echo "    .venv/bin/python transcribe.py"
else
    echo
    echo "For speaker labels (who said what), re-run as:"
    echo "    ./setup.sh --with-diarization"
fi
