#!/usr/bin/env python3
"""Transcription profiles: a model repo and its language code, kept together.

These two fields are not independent choices. The Hebrew profile points at an
ivrit-ai finetune of Whisper large-v3 whose language detection was degraded
during training, so it must be told language="he" and cannot be aimed at any
other language. Pointing it at English produces degraded text, sometimes
transliterated into Hebrew script, with no error to warn you.

Pairing them in one frozen object makes that mistake impossible: callers pass a
profile around, never a loose repo string.

Adding a language means adding one entry below — nothing else changes.
"""

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionProfile:
    """One (model, language) pairing, plus what the user needs told about it."""

    key: str
    label: str
    repo: str
    language: str
    download_size: str


PROFILES = {
    "he": TranscriptionProfile(
        key="he",
        label="Hebrew",
        repo="mlx-community/ivrit-ai-whisper-large-v3-mlx",
        language="he",
        download_size="~3GB",
    ),
    "en": TranscriptionProfile(
        key="en",
        label="English",
        # large-v3-turbo: a distilled decoder, roughly 4-5x faster than
        # large-v3 at effectively equal quality on clear conversational
        # English. Swap to mlx-community/whisper-large-v3-mlx (3GB) if your
        # recordings are heavily accented, noisy, or full of crosstalk.
        repo="mlx-community/whisper-large-v3-turbo",
        language="en",
        download_size="~1.6GB",
    ),
}

DEFAULT_PROFILE_KEY = "he"
PROFILE_ENV_VAR = "TRANSCRIBE_LANG"


def default_profile():
    return PROFILES[DEFAULT_PROFILE_KEY]


def profile_from_env():
    """The profile named by ``TRANSCRIBE_LANG``, or None if it is unset.

    An unrecognized value exits rather than falling back, because a silent
    fallback to Hebrew would be indistinguishable from a working override.
    """
    key = (os.environ.get(PROFILE_ENV_VAR) or "").strip().lower()
    if not key:
        return None
    if key not in PROFILES:
        sys.exit(
            f"{PROFILE_ENV_VAR}={key!r} is not a known language. "
            f"Choose one of: {', '.join(PROFILES)}"
        )
    return PROFILES[key]


def prompt_for_profile():
    """Ask which language to transcribe. Blank picks the default profile."""
    from_env = profile_from_env()
    if from_env is not None:
        print(f"Language: {from_env.label}  (from {PROFILE_ENV_VAR})")
        return from_env

    ordered = list(PROFILES.values())
    menu = "  ".join(f"[{n}] {p.label}" for n, p in enumerate(ordered, 1))
    fallback = default_profile()

    try:
        raw = input(f"Language: {menu}  (Enter = {fallback.label}): ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nCancelled.")

    if not raw:
        return fallback
    if raw.isdigit() and 1 <= int(raw) <= len(ordered):
        return ordered[int(raw) - 1]
    if raw.lower() in PROFILES:
        return PROFILES[raw.lower()]

    print(f"  Not one of the listed choices; using {fallback.label}.")
    return fallback
