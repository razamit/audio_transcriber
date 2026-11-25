#!/bin/bash
# Recreate the local environment for transcribe.py on macOS or Linux.
# Usage:  ./setup.sh [--with-diarization] [--gpu]
# The work is done by install.py, which also runs on Windows (py install.py).
set -euo pipefail

cd "$(dirname "$0")"
exec python3 install.py "$@"
