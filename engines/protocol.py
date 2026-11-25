"""What every transcription engine provides, so callers never care which one runs."""

from typing import Protocol


class TranscriptionEngine(Protocol):
    """Turns an audio file into Whisper-style text segments.

    ``key`` names the engine (``TRANSCRIBE_ENGINE`` uses it). ``runtime_key``
    names the model format it reads, which picks the model from the profile
    (see profiles.py). ``transcribe`` returns
    ``{"text": str, "segments": [{"start", "end", "text"}, ...]}``, the shape
    transcript_output.py and diarization.py consume.
    """

    key: str

    @property
    def runtime_key(self) -> str: ...

    def transcribe(self, audio_path, profile) -> dict: ...
