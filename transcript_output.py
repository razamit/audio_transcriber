"""Rendering transcripts to text, with and without speaker labels.

Whisper's ``result["text"]`` is a single unbroken blob, unpleasant to read for
anything longer than a minute. Everything here exists to break that into
something a person can scan: timestamps on every line, blank lines at long
pauses, and — when speakers are known — a blank line whenever the speaker
changes.

Both writers emit UTF-8 without a BOM and with Unix line endings, so Hebrew
transcripts open correctly everywhere rather than picking up a stray marker
that some editors surface as text.
"""

from collections import defaultdict

from diarization import UNKNOWN_SPEAKER

PARAGRAPH_GAP_SECONDS = 2.0


def format_timestamp(seconds):
    """Seconds -> HH:MM:SS."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _write_utf8(output_path, body):
    """One place that decides transcript encoding and line endings."""
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)


# --------------------------------------------------------------------------
# Plain transcripts
# --------------------------------------------------------------------------

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
    """Write a transcript with no speaker labels."""
    body = render_segments(segments)
    if not body:
        # No usable segments; fall back to the flat blob rather than an empty file.
        body = (fallback_text or "").strip() + "\n"
    _write_utf8(output_path, body)


# --------------------------------------------------------------------------
# Speaker-labelled transcripts
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
    """Write a transcript with a talk-time header and per-speaker lines."""
    lines = summarize_speakers(labelled) + render_labelled_segments(labelled)
    _write_utf8(output_path, "\n".join(lines) + "\n")
