#!/usr/bin/env python3
"""Put real names on an already-diarized transcript.

transcribe.py offers this at the end of a run with speaker labelling, but a
transcript produced earlier would otherwise need both models run again just to
relabel it. This takes the "<name>_speakers.txt" file directly and rewrites it
in place — no audio, no models, none of the heavy dependencies.

Kept separate from transcribe.py for that reason: it is a text-editing job that
happens to share a vocabulary with transcription, not a transcription job.

    python name_speakers.py                       # asks for the transcript
    python name_speakers.py <name>_speakers.txt   # names that one

Given a path it ends by waiting for Enter, because it is then usually running
in a window of its own (the Claude Code plugin opens one), and that window
would otherwise close before the "Renamed" line could be read.
"""

import argparse
import os

from console_streams import use_utf8_when_redirected
from paths import describe_path_problem, prompt_for_existing_file
from speaker_names import name_speakers_in_file

CLOSE_PROMPT = "\nPress Enter to close this window."


def build_parser():
    parser = argparse.ArgumentParser(
        description="Replace SPEAKER_00 labels with names in a _speakers.txt transcript.")
    parser.add_argument("transcript", nargs="?",
                        help="the transcript to name; asked for when left out")
    return parser


def name_and_report(path):
    if name_speakers_in_file(path):
        print(f"Updated: {path}")


def name_given_transcript(raw_path):
    """Name the transcript given on the command line, then wait for Enter."""
    path = os.path.abspath(os.path.expanduser(raw_path))
    try:
        problem = describe_path_problem(path)
        if problem:
            print(problem)
        else:
            name_and_report(path)
    except Exception as error:  # the window must stay open long enough to read it
        print(f"Could not name the speakers: {error}")
    finally:
        wait_before_closing()


def wait_before_closing():
    try:
        input(CLOSE_PROMPT)
    except (EOFError, KeyboardInterrupt):
        print()


def main(argv=None):
    use_utf8_when_redirected()
    options = build_parser().parse_args(argv)
    if options.transcript:
        name_given_transcript(options.transcript)
        return
    path = prompt_for_existing_file(
        "Path to diarized transcript (_speakers.txt): ",
        "Drag the transcript into this window or paste its full path.",
    )
    name_and_report(path)


if __name__ == "__main__":
    main()
