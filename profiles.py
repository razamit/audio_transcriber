#!/usr/bin/env python3
"""Transcription profiles: a language code and the models that serve it, kept together.

These are not independent choices. The Hebrew profile points at ivrit-ai
finetunes of Whisper large-v3 whose language detection was degraded during
training, so they must be told language="he" and cannot be aimed at any other
language. Pointing one at English produces degraded text, sometimes
transliterated into Hebrew script, with no error to warn you.

Pairing them in one frozen object makes that mistake impossible: callers pass a
profile around, never a loose repo string.

A profile carries one model source per runtime, because the same model ships in
different formats: MLX weights for Apple Silicon, CTranslate2 weights for
faster-whisper everywhere else. The runtime keys below say which. Every source
in a profile is a build of a model meant for that profile's language, so the
pairing holds whichever runtime runs it.

Adding a language means adding one entry below; nothing else changes.
"""

import os
import sys
from dataclasses import dataclass, field

# Which model format a transcription engine reads. An engine reports one of
# these, and the profile hands back the matching source.
RUNTIME_MLX = "mlx"            # mlx-whisper on Apple Silicon
RUNTIME_CT2_CUDA = "ct2-cuda"  # faster-whisper on an NVIDIA GPU
RUNTIME_CT2_CPU = "ct2-cpu"    # faster-whisper on the CPU


@dataclass(frozen=True)
class ModelSource:
    """One downloadable model: its HuggingFace repo and how big it is."""

    repo: str
    download_size: str


@dataclass(frozen=True)
class TranscriptionProfile:
    """One language, the models built for it, and what the user needs told."""

    key: str
    label: str
    language: str
    # Keyed by runtime. Left out of the hash: a dict is not hashable, and the
    # key already identifies a profile.
    sources: dict = field(hash=False)

    def source_for(self, runtime_key):
        """The model this profile uses on the given runtime."""
        try:
            return self.sources[runtime_key]
        except KeyError:
            raise ValueError(
                f"The {self.label} profile has no model for runtime {runtime_key!r}"
            ) from None


PROFILES = {
    "he": TranscriptionProfile(
        key="he",
        label="Hebrew",
        language="he",
        sources={
            RUNTIME_MLX: ModelSource("mlx-community/ivrit-ai-whisper-large-v3-mlx", "~3GB"),
            RUNTIME_CT2_CUDA: ModelSource("ivrit-ai/whisper-large-v3-ct2", "~3.1GB"),
            # The turbo finetune on CPU: large-v3 is several times slower there,
            # too slow to be the default for hour-long recordings.
            RUNTIME_CT2_CPU: ModelSource("ivrit-ai/whisper-large-v3-turbo-ct2", "~1.6GB"),
        },
    ),
    "en": TranscriptionProfile(
        key="en",
        label="English",
        language="en",
        # large-v3-turbo: a distilled decoder, roughly 4-5x faster than
        # large-v3 at effectively equal quality on clear conversational
        # English. Swap the MLX source to mlx-community/whisper-large-v3-mlx
        # (3GB) if your recordings are heavily accented, noisy, or full of
        # crosstalk. The CTranslate2 repo is the one faster-whisper itself
        # resolves the name "large-v3-turbo" to.
        sources={
            RUNTIME_MLX: ModelSource("mlx-community/whisper-large-v3-turbo", "~1.6GB"),
            RUNTIME_CT2_CUDA: ModelSource("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "~1.6GB"),
            RUNTIME_CT2_CPU: ModelSource("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "~1.6GB"),
        },
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
