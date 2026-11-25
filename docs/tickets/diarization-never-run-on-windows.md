# Speaker labelling (diarization) has never been run on Windows

Status: CLOSED 2026-11-25 · Priority: MEDIUM · Type: verification gap · Cost: none

> **Resolution (2026-11-25).** Run for real; it was broken, and is fixed. The
> first `--speakers auto` run died at `import torch` with
> `OSError: [WinError 127] ... torch\lib\cudnn_cnn64_9.dll`: the GPU probe had
> already loaded cuDNN 9.28 from `nvidia-cudnn-cu12`, and torch's bundled
> cuDNN 9.10 bound to it. `candidate_library_folders()` now probes `torch/lib`
> first (decision row 3). After the fix: exit 0, both stages on `cuda`, 590
> turns, 5 speakers, 551 lines, 118.5 s cached (122.8 s with the pyannote
> download). Timings are in README's Speed section; see `docs/PROGRESS.md`.

> Found during the 2026-11-25 Windows GPU verification (see `docs/PROGRESS.md`).
> Unrelated to `hf-symlink-warning-floods-stderr-on-windows.md`.

## Symptom
On Windows 11 + RTX 4090 only plain transcription was run. The diarization
extras were not installed, so `py install.py --gpu --with-diarization` (CUDA 12
PyTorch from `download.pytorch.org/whl/cu126`, then pyannote.audio) and a
`--speakers` run have never executed on Windows.

## Evidence
```
> .venv/Scripts/python transcribe.py --capabilities
{"diarization": false, "engine": "faster-whisper", "languages": ["he", "en"], "version": "1.1.0"}
```
README states this under "What has been run for real".

## Why it matters
Windows users who choose speaker labels are on a path covered only by unit
tests with fakes: the CUDA torch install, the ffmpeg-based audio decoding for
pyannote and the torch device choice are unverified there. Plain transcription
is verified and is not affected.

## Fix
Run `py -3.13 install.py --gpu --with-diarization`, then
`.venv\Scripts\python transcribe.py --file <a call> --lang he --speakers auto --json`,
check that torch reports CUDA, record the timing in README's Speed section, and
close this ticket.
