"""The Whisper decoding settings, defined once for every engine.

Both engines must decode the same way, or a transcript would depend on which
machine produced it. See the README's notes on the transcription settings
before changing any of these:

- An explicit language is mandatory because the Hebrew finetunes' language
  detection was degraded during training.
- condition_on_previous_text=False avoids the repetition-loop failure mode on
  long recordings.
- hallucination_silence_threshold only acts when word timestamps are on, in
  both mlx-whisper and faster-whisper, and neither engine turns them on. It is
  passed anyway so enabling word timestamps later activates it everywhere.
"""

CONDITION_ON_PREVIOUS_TEXT = False
HALLUCINATION_SILENCE_THRESHOLD = 2.0


def decoding_options(language):
    """Keyword arguments every engine passes to its transcribe call."""
    return {
        "language": language,
        "condition_on_previous_text": CONDITION_ON_PREVIOUS_TEXT,
        "hallucination_silence_threshold": HALLUCINATION_SILENCE_THRESHOLD,
    }
