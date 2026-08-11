#!/usr/bin/env python3
"""Running Whisper: turning audio into timestamped text segments.

Knows nothing about speakers, file paths, or prompts — it takes a profile and
an audio file and hands back Whisper's raw result. The model to use comes from
the profile rather than a constant here, because the model and the language
code are one inseparable choice (see profiles.py).
"""

import os


def hf_cache_root():
    """Root of the HuggingFace hub cache, honoring the usual env overrides."""
    cache_root = os.environ.get("HF_HUB_CACHE")
    if cache_root:
        return cache_root
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.join(hf_home, "hub")


def model_is_cached(repo):
    """True if the given repo already looks present in the HuggingFace cache."""
    folder = "models--" + repo.replace("/", "--")
    return os.path.isdir(os.path.join(hf_cache_root(), folder))


def announce_first_run_download(profile):
    """Warn about the one-time model download so a slow start isn't a mystery.

    Each profile has its own model, so switching language for the first time
    triggers a fresh download even though the tool has been run before.
    """
    if not model_is_cached(profile.repo):
        print(
            f"First run for {profile.label}: downloading {profile.repo} "
            f"({profile.download_size}) into the HuggingFace cache. This takes "
            "a few minutes and only happens once per language."
        )


def run_transcription(audio_path, profile):
    """Transcribe with mlx-whisper using the given profile's model.

    The kwargs are deliberate: an explicit language is mandatory because the
    Hebrew finetune's language detection was degraded during training, and
    condition_on_previous_text=False avoids the repetition-loop failure mode
    on long recordings. See the README before changing them.
    """
    import mlx_whisper

    return mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=profile.repo,
        language=profile.language,
        verbose=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
    )
