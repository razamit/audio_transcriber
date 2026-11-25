#!/usr/bin/env python3
"""Decoding an audio file into samples through the ffmpeg CLI, for pyannote and faster-whisper.

pyannote 4.x decodes files with torchcodec, which aborts the whole run on the
first packet it cannot decode. Real recordings often carry one: phone and
recorder apps commonly leave a truncated or junk frame at the very end of an
MP3. The ffmpeg CLI drops such a packet and keeps going, which is also what
mlx-whisper's own loader does, so decoding here keeps every model reading the
file the same way.

faster-whisper would otherwise decode through PyAV, whose API has moved under
it (faster-whisper 1.2.1 passes an argument PyAV 19 no longer accepts). It is
handed samples from here instead, decoded exactly as mlx-whisper decodes them,
so both transcription engines hear identical audio.

torch and numpy are imported inside the functions for the same reason as in
diarization.py: this module must stay importable without either installed.
"""

import subprocess

# The rate pyannote's segmentation and embedding models were trained at, and
# the rate Whisper expects. Decoding straight to it saves a resampling pass.
DIARIZATION_SAMPLE_RATE = 16000
WHISPER_SAMPLE_RATE = 16000


def decode_with_ffmpeg(audio_path, sample_rate, sample_format):
    """Raw mono samples as bytes, in the given ffmpeg sample format (f32le, s16le).

    Raises RuntimeError if ffmpeg cannot read the file at all, or if nothing
    decodable is in it.
    """
    command = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", audio_path,
        "-f", sample_format, "-ac", "1", "-ar", str(sample_rate),
        "-",
    ]
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0:
        details = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {audio_path}:\n{details}")
    if not completed.stdout:
        raise RuntimeError(f"ffmpeg decoded no audio from {audio_path}")
    return completed.stdout


def decode_to_waveform(audio_path, sample_rate=DIARIZATION_SAMPLE_RATE):
    """Decode a file to the in-memory mono waveform dict pyannote accepts."""
    import torch

    raw_samples = decode_with_ffmpeg(audio_path, sample_rate, "f32le")
    # bytearray, because torch.frombuffer wants a writable buffer.
    samples = torch.frombuffer(bytearray(raw_samples), dtype=torch.float32)
    return {"waveform": samples.unsqueeze(0), "sample_rate": sample_rate}


def decode_for_whisper(audio_path):
    """16 kHz mono float32 samples in [-1, 1), as mlx-whisper's loader produces them.

    16-bit PCM scaled by 1/32768, not float output, to match mlx-whisper
    sample for sample.
    """
    import numpy

    raw_samples = decode_with_ffmpeg(audio_path, WHISPER_SAMPLE_RATE, "s16le")
    return numpy.frombuffer(raw_samples, numpy.int16).astype(numpy.float32) / 32768.0
