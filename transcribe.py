#!/usr/bin/env python3
"""Local audio transcription on Apple Silicon, optionally with speaker labels.

The single entry point. Run it, answer four prompts, get a transcript written
next to the input file:

    Language: [1] Hebrew  [2] English  (Enter = Hebrew): 2
    Path to audio file: call.m4a
    Label speakers? [y/N]: y
    Number of speakers (press Enter to auto-detect): 3

Everything below is orchestration only — prompts, ordering, and error
handling. The work lives in dedicated modules: profiles.py (which model for
which language), paths.py (input/output paths), transcription.py (Whisper),
diarization.py (pyannote), transcript_output.py (rendering), speaker_names.py
(putting real names on the labels).

Speaker labelling is optional because its dependencies are not: pyannote pulls
in roughly 1.2GB of PyTorch that plain transcription never touches, so it
installs separately and this script degrades gracefully without it.
"""

import os
import sys
import time
import traceback

import diarization
from diarization import extract_speaker_turns, label_segments, prompt_for_speaker_count
from paths import build_output_path, prompt_for_audio_path, require_ffmpeg
from profiles import prompt_for_profile
from speaker_names import name_speakers_in_file
from transcript_output import (
    format_timestamp,
    write_labelled_transcript,
    write_transcript,
)
from transcription import announce_first_run_download, run_transcription


def prompt_for_speaker_labelling():
    """Ask whether to identify who spoke. Exits with help if it cannot.

    Answering yes without the extras installed is a dead end, so it is caught
    here — before the audio path is used and long before any model loads.
    """
    try:
        raw = input("Label speakers? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nCancelled.")

    if raw not in ("y", "yes"):
        return False

    if not diarization.diarization_is_available():
        sys.exit(f"\n{diarization.INSTALL_HELP}")
    return True


def transcribe_only(audio_path, profile, output_path):
    """Whisper alone: text with timestamps, no speakers."""
    print(f"\nTranscribing {audio_path} as {profile.label} ...")
    result = run_transcription(audio_path, profile)
    write_transcript(output_path, result.get("segments") or [], result.get("text"))


def transcribe_with_speakers(audio_path, profile, output_path, speaker_count):
    """pyannote then Whisper, merged by timestamp.

    Diarization runs first so that a missing model or a bad audio file fails
    before spending minutes on transcription.
    """
    print("\nLoading diarization pipeline ...")
    pipeline = diarization.load_diarization_pipeline(diarization.optional_hf_token())

    print("Detecting speaker turns ...")
    turns = extract_speaker_turns(pipeline, audio_path, speaker_count)
    print(f"  {len(turns)} turn(s) found.")

    print(f"\nTranscribing {audio_path} as {profile.label} ...")
    result = run_transcription(audio_path, profile)

    labelled = label_segments(result.get("segments") or [], turns)
    if not labelled:
        sys.exit("No speech was transcribed; nothing to write.")
    write_labelled_transcript(output_path, labelled)


def main():
    require_ffmpeg()

    profile = prompt_for_profile()
    audio_path = prompt_for_audio_path()
    with_speakers = prompt_for_speaker_labelling()
    speaker_count = prompt_for_speaker_count() if with_speakers else None

    output_path = build_output_path(audio_path, "_speakers" if with_speakers else "")
    announce_first_run_download(profile)
    started = time.monotonic()

    try:
        if with_speakers:
            transcribe_with_speakers(audio_path, profile, output_path, speaker_count)
        else:
            transcribe_only(audio_path, profile, output_path)
    except Exception:
        print("\nFailed:\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.monotonic() - started
    print(f"\nTranscript written to: {os.path.abspath(output_path)}")
    print(f"Elapsed: {format_timestamp(elapsed)} ({elapsed:.1f}s)")

    # Naming happens after the file is safely on disk, never before: the
    # expensive work is already saved, so abandoning this prompt costs nothing
    # and the transcript survives exactly as diarization produced it.
    if with_speakers:
        name_speakers_in_file(output_path)


if __name__ == "__main__":
    main()
