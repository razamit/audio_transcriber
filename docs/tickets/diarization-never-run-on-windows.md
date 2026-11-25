# Speaker labelling (diarization) has never been run on Windows

Status: OPEN · Priority: MEDIUM · Type: verification gap · Cost: none

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
