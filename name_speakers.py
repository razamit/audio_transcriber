#!/usr/bin/env python3
"""Put real names on an already-diarized transcript.

transcribe.py offers this at the end of a run with speaker labelling, but a
transcript produced earlier would otherwise need both models run again just to
relabel it. This takes the "<name>_speakers.txt" file directly and rewrites it
in place — no audio, no models, none of the heavy dependencies.

Kept separate from transcribe.py for that reason: it is a text-editing job that
happens to share a vocabulary with transcription, not a transcription job.
"""

from paths import prompt_for_existing_file
from speaker_names import name_speakers_in_file


def main():
    path = prompt_for_existing_file(
        "Path to diarized transcript (_speakers.txt): ",
        "Drag the transcript into this window or paste its full path.",
    )
    if name_speakers_in_file(path):
        print(f"Updated: {path}")


if __name__ == "__main__":
    main()
