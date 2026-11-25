# Speaker labelling prints three library warnings on stderr

Status: OPEN · Priority: LOW · Type: noisy output · Cost: none

> Found during the 2026-11-25 Windows diarization run (see `docs/PROGRESS.md`).
> Same kind of problem as `hf-symlink-warning-floods-stderr-on-windows.md`,
> but a different source.

## Symptom
Each `--speakers` run on Windows (RTX 4090, torch 2.14.1+cu126,
pyannote.audio 4.0.7) prints three warnings among the tool's own progress
lines, even when the model is already cached:

1. From torch, at import: `triton not found; flop counting will not work for triton kernels`
2. From pyannote: a 7-line `ReproducibilityWarning` saying TF32 has been disabled
3. From pyannote: `UserWarning: std(): degrees of freedom is <= 0`

## Evidence
stderr of the cached run (output `diar_run2.txt`, exit 0, 590 turns):
```
Loading diarization pipeline ...
W1125 11:11:04.023000 33048 .venv\Lib\site-packages\torch\utils\flop_counter.py:113] triton not found; flop counting will not work for triton kernels
  Diarization running on: cuda
Detecting speaker turns ...
...\pyannote\audio\utils\reproducibility.py:74: ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.
...
...\pyannote\audio\models\blocks\pooling.py:103: UserWarning: std(): degrees of freedom is <= 0. ...
  590 turn(s) found.
```

## Why it matters
Cosmetic only: the run succeeds and the transcript is correct. The warnings
make a successful run look like something went wrong. The `std()` warning
comes from a very short speech segment and does not affect the result.

## Scope to decide
- Filter these three warnings by category and module inside
  `diarization.load_diarization_pipeline` / `extract_speaker_turns`
  (and the torch logger for triton)
  OR
- Leave them, and mention in the README that they are harmless.

Whichever is chosen, do the same for the huggingface symlink ticket.
