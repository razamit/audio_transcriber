#!/usr/bin/env python3
"""Transcription with speaker labels ("who said what").

Runs two models over the same audio and merges them by timestamp:

  1. pyannote.audio  -> speaker turns  (who spoke when, no words)
  2. mlx-whisper     -> text segments  (what was said, no speaker)

Whisper has no concept of speakers and pyannote has no concept of words, so
neither can do this alone. The merge assigns each Whisper segment to whichever
speaker overlaps it most in time.

Speakers come out as SPEAKER_00, SPEAKER_01... — the model separates voices,
it does not know their names.

pyannote is language-agnostic, so only the Whisper half depends on the chosen
language profile.
"""

import os
import sys
import time
import traceback
from collections import defaultdict

from profiles import prompt_for_profile
from transcribe import (
    announce_first_run_download,
    build_output_path,
    format_timestamp,
    prompt_for_audio_path,
    require_ffmpeg,
    run_transcription,
)

DIARIZATION_REPO = "pyannote/speaker-diarization-community-1"
PARAGRAPH_GAP_SECONDS = 2.0
UNKNOWN_SPEAKER = "UNKNOWN"

# pyannote.audio 4.x bundles OpenTelemetry metrics that are enabled by default
# and POST to https://otel.pyannote.ai/v1/traces on every pipeline run. No
# audio, text or filename is transmitted — it is the model id, library
# version, a random session id, the audio duration and the speaker-count hint.
# This is still a local-first tool, so opt out by default. Export
# PYANNOTE_METRICS_ENABLED=true to send them anyway.
#
# setdefault, and before pyannote is imported: the package reads this variable
# at import time and falls back to its bundled config.yaml if it is unset.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

TOKEN_HELP = f"""\
A HuggingFace access token is required: the pyannote models are gated.

  1. Create a free account at https://huggingface.co/join
  2. Accept the conditions at:
       https://hf.co/{DIARIZATION_REPO}
       https://hf.co/pyannote/segmentation-3.0
  3. Create a token (type "Read") at https://hf.co/settings/tokens
  4. Store it once:
       .venv/bin/hf auth login
     or export it for this shell:
       export HF_TOKEN=hf_your_token_here
"""


# --------------------------------------------------------------------------
# Model setup
# --------------------------------------------------------------------------

def resolve_hf_token():
    """Find a token in the env or the stored HuggingFace login, or exit."""
    from huggingface_hub import get_token

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token = token or get_token()
    if not token:
        sys.exit(TOKEN_HELP)
    return token


def select_torch_device():
    """Prefer Apple's Metal backend, fall back to CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_diarization_pipeline(token):
    """Build the pyannote pipeline, moved onto the best available device.

    MPS is preferred but not every pyannote op has a Metal kernel, so a
    failure to move the pipeline falls back to CPU rather than dying.
    """
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_REPO, token=token)
    if pipeline is None:
        sys.exit(
            f"Could not load {DIARIZATION_REPO}. The token was found but access "
            f"was refused.\n\n{TOKEN_HELP}"
        )

    device = select_torch_device()
    try:
        pipeline.to(device)
    except Exception as exc:
        print(f"  Could not use {device} ({exc}); falling back to CPU.")
        device = __import__("torch").device("cpu")
        pipeline.to(device)

    print(f"  Diarization running on: {device}")
    return pipeline


def prompt_for_speaker_count():
    """Ask how many speakers to expect. Blank means let the model decide."""
    try:
        raw = input("Number of speakers (press Enter to auto-detect): ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nCancelled.")

    if not raw:
        return None
    if raw.isdigit() and int(raw) > 0:
        return int(raw)

    print("  Not a positive number; auto-detecting instead.")
    return None


# --------------------------------------------------------------------------
# Diarization and merging
# --------------------------------------------------------------------------

def extract_speaker_turns(pipeline, audio_path, speaker_count):
    """Run diarization and flatten it to (start, end, speaker) tuples."""
    kwargs = {"num_speakers": speaker_count} if speaker_count else {}
    annotation = pipeline(audio_path, **kwargs)

    turns = [
        (segment.start, segment.end, speaker)
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda turn: turn[0])
    return turns


def dominant_speaker(segment_start, segment_end, turns):
    """The speaker overlapping [start, end] the longest, or None."""
    overlap_by_speaker = defaultdict(float)

    for turn_start, turn_end, speaker in turns:
        if turn_start >= segment_end:
            break  # turns are sorted; nothing later can overlap
        overlap = min(segment_end, turn_end) - max(segment_start, turn_start)
        if overlap > 0:
            overlap_by_speaker[speaker] += overlap

    if not overlap_by_speaker:
        return None
    return max(overlap_by_speaker, key=overlap_by_speaker.get)


def label_segments(segments, turns):
    """Attach a speaker to each Whisper segment, dropping empty ones."""
    labelled = []

    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue

        start = segment.get("start", 0.0)
        end = segment.get("end", start)
        speaker = dominant_speaker(start, end, turns)
        labelled.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker or UNKNOWN_SPEAKER,
            }
        )

    return labelled


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def summarize_speakers(labelled):
    """Header lines listing each speaker and how long they talked."""
    talk_time = defaultdict(float)
    for entry in labelled:
        talk_time[entry["speaker"]] += entry["end"] - entry["start"]

    total = sum(talk_time.values()) or 1.0
    identified = [name for name in talk_time if name != UNKNOWN_SPEAKER]

    lines = [f"{len(identified)} speaker(s) detected:"]
    for speaker, seconds in sorted(talk_time.items(), key=lambda kv: -kv[1]):
        share = 100.0 * seconds / total
        note = "  (no speaker matched this span)" if speaker == UNKNOWN_SPEAKER else ""
        lines.append(
            f"  {speaker}  {format_timestamp(seconds)}  ({share:.0f}%){note}"
        )
    lines.append("")
    return lines


def render_labelled_segments(labelled):
    """`[HH:MM:SS] SPEAKER_00: text`, blank line on speaker change or pause."""
    lines = []
    previous_end = None
    previous_speaker = None

    for entry in labelled:
        speaker_changed = previous_speaker is not None and entry["speaker"] != previous_speaker
        long_pause = (
            previous_end is not None
            and entry["start"] - previous_end > PARAGRAPH_GAP_SECONDS
        )
        if speaker_changed or long_pause:
            lines.append("")

        stamp = format_timestamp(entry["start"])
        lines.append(f"[{stamp}] {entry['speaker']}: {entry['text']}")
        previous_end = entry["end"]
        previous_speaker = entry["speaker"]

    return lines


def write_labelled_transcript(output_path, labelled):
    """Write the merged transcript as UTF-8 without a BOM."""
    lines = summarize_speakers(labelled) + render_labelled_segments(labelled)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    require_ffmpeg()
    token = resolve_hf_token()

    profile = prompt_for_profile()
    audio_path = prompt_for_audio_path()
    speaker_count = prompt_for_speaker_count()
    output_path = build_output_path(audio_path, suffix="_speakers")

    announce_first_run_download(profile)
    started = time.monotonic()

    try:
        print("\nLoading diarization pipeline ...")
        pipeline = load_diarization_pipeline(token)

        print("Detecting speaker turns ...")
        turns = extract_speaker_turns(pipeline, audio_path, speaker_count)
        print(f"  {len(turns)} turn(s) found.")

        print(f"\nTranscribing {audio_path} as {profile.label} ...")
        result = run_transcription(audio_path, profile)
    except Exception:
        print("\nFailed:\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    labelled = label_segments(result.get("segments") or [], turns)
    if not labelled:
        sys.exit("No speech was transcribed; nothing to write.")

    elapsed = time.monotonic() - started
    write_labelled_transcript(output_path, labelled)

    print(f"\nTranscript written to: {os.path.abspath(output_path)}")
    print(f"Elapsed: {format_timestamp(elapsed)} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
