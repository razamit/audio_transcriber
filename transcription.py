#!/usr/bin/env python3
"""Running Whisper: turning audio into timestamped text segments.

Knows nothing about speakers, file paths, or prompts: it takes a profile and
an audio file and hands back Whisper's raw result. The model to use comes from
the profile rather than a constant here, because the model and the language
code are one inseparable choice (see profiles.py).

Which library runs Whisper depends on the machine: mlx-whisper on Apple
Silicon, faster-whisper elsewhere (see engines/). The engine is chosen once per
process, and also decides which of the profile's models is used.
"""

import functools
import os

from engines.selection import select_engine


@functools.lru_cache(maxsize=1)
def active_engine():
    """The engine for this machine, chosen on first use and then kept."""
    return select_engine()


def hf_cache_root(environ=None, path_module=os.path, home=None):
    """Root of the HuggingFace hub cache, resolved the way huggingface_hub does.

    HF_HUB_CACHE wins, then HF_HOME, then XDG_CACHE_HOME, then ~/.cache. On
    Windows that last one is C:\\Users\\<name>\\.cache\\huggingface\\hub, which is
    where huggingface_hub downloads to as well.
    """
    environ = os.environ if environ is None else environ
    cache_root = environ.get("HF_HUB_CACHE")
    if cache_root:
        return path_module.expanduser(cache_root)
    hf_home = environ.get("HF_HOME")
    if not hf_home:
        home = path_module.expanduser("~") if home is None else home
        cache_home = environ.get("XDG_CACHE_HOME") or path_module.join(home, ".cache")
        hf_home = path_module.join(cache_home, "huggingface")
    return path_module.join(path_module.expanduser(hf_home), "hub")


def model_is_cached(repo):
    """True if the given repo already looks present in the HuggingFace cache."""
    folder = "models--" + repo.replace("/", "--")
    return os.path.isdir(os.path.join(hf_cache_root(), folder))


def announce_first_run_download(profile, engine=None):
    """Warn about the one-time model download so a slow start isn't a mystery.

    Each profile has its own model, so switching language for the first time
    triggers a fresh download even though the tool has been run before.
    """
    source = profile.source_for((engine or active_engine()).runtime_key)
    if not model_is_cached(source.repo):
        print(
            f"First run for {profile.label}: downloading {source.repo} "
            f"({source.download_size}) into the HuggingFace cache. This takes "
            "a few minutes and only happens once per language."
        )


def run_transcription(audio_path, profile, engine=None):
    """Transcribe with this machine's engine, using the profile's model for it.

    Returns ``{"text": str, "segments": [{"start", "end", "text"}, ...]}``.
    The decoding settings live in engines/decoding.py; see the README before
    changing them.
    """
    return (engine or active_engine()).transcribe(audio_path, profile)
