#!/usr/bin/env python3
"""Local Hebrew audio transcription on Apple Silicon via mlx-whisper.

Interactive only: run it, paste or drag in an audio file, get a timestamped
transcript written next to the input file.
"""

import os
import re
import shutil
import sys
import time
import traceback

MODEL_REPO = "mlx-community/ivrit-ai-whisper-large-v3-mlx"
LANGUAGE = "he"
PARAGRAPH_GAP_SECONDS = 2.0
MAX_PROMPT_ATTEMPTS = 3


# --------------------------------------------------------------------------
# Input path handling
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


def prompt_for_audio_path():
    """Prompt until a readable file is given. Exits after too many failures."""
    for attempt in range(1, MAX_PROMPT_ATTEMPTS + 1):
        try:
            raw = input("Path to audio file: ")
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

    sys.exit(
        f"Giving up after {MAX_PROMPT_ATTEMPTS} attempts. "
        "Drag the file into this window or paste its full path, then press Enter."
    )


# --------------------------------------------------------------------------
# Output path and formatting
# --------------------------------------------------------------------------

def build_output_path(audio_path):
    """`<stem>.txt` next to the input, suffixed `_2`, `_3`... if taken."""
    directory = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]

    candidate = os.path.join(directory, f"{stem}.txt")
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_{counter}.txt")
        counter += 1
    return candidate


def format_timestamp(seconds):
    """Seconds -> HH:MM:SS."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_segments(segments):
    """One `[HH:MM:SS] text` line per segment, blank line on a long pause."""
    lines = []
    previous_end = None

    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue

        start = segment.get("start", 0.0)
        if previous_end is not None and start - previous_end > PARAGRAPH_GAP_SECONDS:
            lines.append("")
        lines.append(f"[{format_timestamp(start)}] {text}")
        previous_end = segment.get("end", start)

    return "\n".join(lines) + "\n" if lines else ""


def write_transcript(output_path, segments, fallback_text):
    """Write the rendered transcript as UTF-8 without a BOM."""
    body = render_segments(segments)
    if not body:
        # No usable segments; fall back to the flat blob rather than an empty file.
        body = (fallback_text or "").strip() + "\n"

    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)


# --------------------------------------------------------------------------
# Environment checks
# --------------------------------------------------------------------------

def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ffmpeg is required to decode audio but was not found on PATH.\n"
            "Install it with:  brew install ffmpeg"
        )


def model_is_cached():
    """True if the model repo already looks present in the HuggingFace cache."""
    cache_root = os.environ.get("HF_HUB_CACHE")
    if not cache_root:
        hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        cache_root = os.path.join(hf_home, "hub")

    folder = "models--" + MODEL_REPO.replace("/", "--")
    return os.path.isdir(os.path.join(cache_root, folder))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    require_ffmpeg()

    audio_path = prompt_for_audio_path()
    output_path = build_output_path(audio_path)

    if not model_is_cached():
        print(
            f"First run: downloading {MODEL_REPO} (~3GB) into the HuggingFace "
            "cache. This takes a few minutes and only happens once."
        )

    print(f"Transcribing {audio_path} ...")
    started = time.monotonic()

    try:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=MODEL_REPO,
            language=LANGUAGE,
            verbose=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
        )
    except Exception:
        print("\nTranscription failed:\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.monotonic() - started
    write_transcript(output_path, result.get("segments") or [], result.get("text"))

    print(f"\nTranscript written to: {os.path.abspath(output_path)}")
    print(f"Elapsed: {format_timestamp(elapsed)} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
