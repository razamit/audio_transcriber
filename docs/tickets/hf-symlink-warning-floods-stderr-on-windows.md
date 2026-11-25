# huggingface_hub prints a multi-line symlink warning on every first download on Windows

Status: OPEN · Priority: LOW · Type: noisy output · Cost: none

> Found during the 2026-11-25 Windows GPU verification (see `docs/PROGRESS.md`).

## Symptom
On Windows without Developer Mode, the first model download prints a
`UserWarning` from `huggingface_hub/file_download.py:149` about symlinks, twice
(once for the hub folder, once for the model folder), each with a paragraph of
advice and a URL. It lands in the middle of the tool's own progress lines on
stderr.

## Evidence
From the stderr of the first Hebrew run:
```
...\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\<you>\.cache\huggingface\hub. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. ...
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. ...
```

## Why it matters
Cosmetic only: the download and the transcription succeed (exit 0). Not a
correctness or disk problem for this tool, which keeps one revision per model.
It does make the first run look like something went wrong.

## Scope to decide
- Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` in the process before the first
  download on Windows
  OR
- Leave it, and mention in the README that the warning is harmless.
