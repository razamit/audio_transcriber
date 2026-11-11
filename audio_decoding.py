#!/usr/bin/env python3
"""Decoding an audio file into samples for pyannote, through the ffmpeg CLI.

pyannote 4.x decodes files with torchcodec, which aborts the whole run on the
first packet it cannot decode. Real recordings often carry one: phone and
recorder apps commonly leave a truncated or junk frame at the very end of an
MP3. The ffmpeg CLI drops such a packet and keeps going, which is also what
Whisper's own loader does, so decoding here keeps both models reading the file
the same way.

torch is imported inside the function for the same reason as in diarization.py:
this module must stay importable without the diarization extras.
"""

import subprocess

# The rate pyannote's segmentation and embedding models were trained at.
# Decoding straight to it saves pyannote a resampling pass.
DIARIZATION_SAMPLE_RATE = 16000


def decode_to_waveform(audio_path, sample_rate=DIARIZATION_SAMPLE_RATE):
    """Decode a file to the in-memory mono waveform dict pyannote accepts.

    Raises RuntimeError if ffmpeg cannot read the file at all, or if nothing
    decodable is in it.
    """
    import torch

    command = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", audio_path,
        "-f", "f32le", "-ac", "1", "-ar", str(sample_rate),
        "-",
    ]
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0:
        details = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {audio_path}:\n{details}")
    if not completed.stdout:
        raise RuntimeError(f"ffmpeg decoded no audio from {audio_path}")

    # bytearray, because torch.frombuffer wants a writable buffer.
    samples = torch.frombuffer(bytearray(completed.stdout), dtype=torch.float32)
    return {"waveform": samples.unsqueeze(0), "sample_rate": sample_rate}
