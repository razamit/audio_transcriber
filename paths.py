#!/usr/bin/env python3
"""Turning what a user types at a prompt into a usable file path.

Terminal drag-and-drop is the primary way files get into this tool, and it
does not hand over a clean path: it escapes spaces and parentheses with
backslashes, quoting styles vary, and ``~`` arrives unexpanded. This module
absorbs all of that so no caller has to think about it.

Also holds the ffmpeg check, since a missing decoder is the other way a run
dies before any model loads.
"""

import os
import re
import shutil
import sys

MAX_PROMPT_ATTEMPTS = 3


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def require_ffmpeg():
    """Exit early if ffmpeg is missing, rather than deep inside Whisper."""
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ffmpeg is required to decode audio but was not found on PATH.\n"
            "Install it with:  brew install ffmpeg"
        )


# --------------------------------------------------------------------------
# Input paths
# --------------------------------------------------------------------------

def _strip_surrounding_quotes(raw):
    """Remove one matching pair of surrounding single/double quotes."""
    for quote in ("'", '"'):
        if len(raw) >= 2 and raw.startswith(quote) and raw.endswith(quote):
            return raw[1:-1], True
    return raw, False


def sanitize_path_input(raw):
    """Turn whatever the user pasted or dragged in into a usable path.

    Handles surrounding quotes, the backslash escaping Terminal adds on
    drag-and-drop, and ``~``. Returns the best candidate; if unescaping
    produced a path that does not exist but the literal one does, the literal
    one wins (filenames may legitimately contain backslashes).
    """
    text = raw.strip()
    text, was_quoted = _strip_surrounding_quotes(text)
    text = text.strip()

    candidates = [text]
    if not was_quoted and "\\" in text:
        # Terminal escapes spaces, parens, ampersands... as \<char>.
        candidates.insert(0, re.sub(r"\\(.)", r"\1", text))

    candidates = [os.path.expanduser(c) for c in candidates]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def describe_path_problem(path):
    """Return a human-readable reason the path is unusable, or None if fine."""
    if not path:
        return "No path entered."
    if not os.path.exists(path):
        return f"No such file: {path}"
    if os.path.isdir(path):
        return f"That is a directory, not a file: {path}"
    if not os.access(path, os.R_OK):
        return f"File exists but is not readable: {path}"
    return None


def prompt_for_existing_file(label, give_up_hint=""):
    """Prompt until a readable file is given. Exits after too many failures."""
    for attempt in range(1, MAX_PROMPT_ATTEMPTS + 1):
        try:
            raw = input(label)
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nCancelled.")

        path = sanitize_path_input(raw)
        problem = describe_path_problem(path)
        if problem is None:
            return os.path.abspath(path)

        remaining = MAX_PROMPT_ATTEMPTS - attempt
        print(f"  {problem}")
        if remaining:
            print(f"  Try again ({remaining} attempt(s) left).")

    sys.exit(f"Giving up after {MAX_PROMPT_ATTEMPTS} attempts. {give_up_hint}".strip())


def prompt_for_audio_path():
    """Prompt for the recording to process."""
    return prompt_for_existing_file(
        "Path to audio file: ",
        "Drag the file into this window or paste its full path, then press Enter.",
    )


# --------------------------------------------------------------------------
# Output paths
# --------------------------------------------------------------------------

def build_output_path(audio_path, suffix=""):
    """`<stem><suffix>.txt` next to the input, bumped `_2`, `_3`... if taken.

    Never overwrites: a transcript is expensive to produce and the user may
    still want the previous one.
    """
    directory = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0] + suffix

    candidate = os.path.join(directory, f"{stem}.txt")
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_{counter}.txt")
        counter += 1
    return candidate
