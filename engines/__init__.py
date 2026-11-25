"""Transcription engines: the code that actually runs Whisper.

Two engines produce the same result from the same profile:

- mlx_engine.py runs mlx-whisper, which needs Apple Silicon.
- faster_whisper_engine.py runs faster-whisper (CTranslate2) on an NVIDIA GPU
  or the CPU, on Windows, Linux and Intel Macs.

selection.py picks one for this machine. Nothing in this package imports a
Whisper library at module level, so it stays importable on any install.
"""
