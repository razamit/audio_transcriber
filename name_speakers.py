#!/usr/bin/env python3
"""Put real names on an already-diarized transcript.

diarize.py offers this at the end of a run, but a transcript produced earlier
would otherwise need both models run again just to relabel it. This takes the
"<name>_speakers.txt" file directly and rewrites it in place.

Interactive, like the other scripts: run it, drag in the transcript.
"""

import os
import sys

from speaker_names import name_speakers_in_file
from transcribe import describe_path_problem, sanitize_path_input

MAX_PROMPT_ATTEMPTS = 3


def prompt_for_transcript_path():
    """Prompt until a readable .txt transcript is given, or give up."""
    for attempt in range(1, MAX_PROMPT_ATTEMPTS + 1):
        try:
            raw = input("Path to diarized transcript (_speakers.txt): ")
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

    sys.exit(f"Giving up after {MAX_PROMPT_ATTEMPTS} attempts.")


def main():
    path = prompt_for_transcript_path()
    if name_speakers_in_file(path):
        print(f"Updated: {path}")


if __name__ == "__main__":
    main()
