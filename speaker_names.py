#!/usr/bin/env python3
"""Replace anonymous SPEAKER_00 labels with real names in a transcript.

Diarization separates voices but cannot name them, so the output arrives as
SPEAKER_00, SPEAKER_01... Those labels are also arbitrary: the numbering
reflects clustering order, not anything about the people, and it changes
between runs. Naming them is the step that makes a transcript readable.

The prompt shows each speaker's longest lines rather than just the label,
because "SPEAKER_01" on its own is not something anyone can answer.

Works on the transcript text, not on in-memory segments, so the same code
serves diarize.py right after a run and name_speakers.py on a file produced
days ago.
"""

import os
import re
import sys

# "[00:12:34] SPEAKER_01: some text" — the timestamp is consumed first so the
# colons inside it cannot be mistaken for the speaker/text separator.
SEGMENT_PATTERN = re.compile(r"^(\[\d{2}:\d{2}:\d{2}\] )([^:]+)(: )(.*)$")

# "  SPEAKER_01  00:19:35  (51%)" — the talk-time rows in the header summary.
SUMMARY_PATTERN = re.compile(r"^(\s+)(\S+)(\s+\d{2}:\d{2}:\d{2}\s+\(\d+%\).*)$")

UNKNOWN_SPEAKER = "UNKNOWN"
SAMPLES_PER_SPEAKER = 3
MIN_SAMPLE_CHARS = 25


# --------------------------------------------------------------------------
# Reading the transcript
# --------------------------------------------------------------------------

def parse_segments(text):
    """Every "[stamp] SPEAKER: text" line as a (stamp, speaker, text) tuple."""
    segments = []
    for line in text.splitlines():
        match = SEGMENT_PATTERN.match(line)
        if match:
            stamp, speaker, _, spoken = match.groups()
            segments.append((stamp.strip("[] "), speaker, spoken))
    return segments


def summarize_by_speaker(segments):
    """Per speaker: segment count and the most identifiable sample lines.

    Ordered by how much each speaker said, so the dominant voice is named
    first — that is usually the easiest one to recognize.
    """
    grouped = {}
    for stamp, speaker, spoken in segments:
        grouped.setdefault(speaker, []).append((stamp, spoken))

    summaries = []
    for speaker, lines in grouped.items():
        if speaker == UNKNOWN_SPEAKER:
            continue  # not a person; a span no speaker turn matched

        # Long lines identify a speaker far better than "Yeah." or "Okay."
        substantial = [ln for ln in lines if len(ln[1]) >= MIN_SAMPLE_CHARS]
        candidates = substantial or lines
        longest = sorted(candidates, key=lambda ln: -len(ln[1]))[:SAMPLES_PER_SPEAKER]
        summaries.append(
            {
                "speaker": speaker,
                "count": len(lines),
                "samples": sorted(longest),  # back into chronological order
            }
        )

    summaries.sort(key=lambda s: -s["count"])
    return summaries


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

def validate_name(raw):
    """Return (name, problem). A colon would corrupt the transcript format."""
    name = raw.strip()
    if not name:
        return None, None
    if ":" in name:
        return None, "A name cannot contain a colon; it would break the format."
    if len(name) > 40:
        return None, "That name is unreasonably long; keep it under 40 characters."
    return name, None


def prompt_for_names(summaries):
    """Ask for a name per speaker. Blank keeps the anonymous label."""
    if not summaries:
        return {}

    print("\nName the speakers. Press Enter to keep a label as it is.\n")
    mapping = {}

    for summary in summaries:
        speaker = summary["speaker"]
        print(f"{speaker} — {summary['count']} segment(s)")
        for stamp, spoken in summary["samples"]:
            excerpt = spoken if len(spoken) <= 90 else spoken[:87] + "..."
            print(f"  [{stamp}] {excerpt}")

        while True:
            try:
                raw = input(f"Name (Enter to keep {speaker}): ")
            except (EOFError, KeyboardInterrupt):
                print("\n  Stopping here; names chosen so far will be applied.")
                return mapping

            name, problem = validate_name(raw)
            if problem:
                print(f"  {problem}")
                continue
            if name and name != speaker:
                mapping[speaker] = name
            break
        print()

    return mapping


def warn_on_duplicates(mapping):
    """Two labels sharing a name merges them visually; say so rather than hide it."""
    used = {}
    for speaker, name in mapping.items():
        used.setdefault(name, []).append(speaker)

    for name, speakers in used.items():
        if len(speakers) > 1:
            print(f"  Note: {' and '.join(sorted(speakers))} are both named "
                  f"{name!r}; they stay separate rows in the summary.")


# --------------------------------------------------------------------------
# Rewriting
# --------------------------------------------------------------------------

def rename_in_text(text, mapping):
    """Apply the mapping to both the summary header and the segment lines."""
    if not mapping:
        return text

    output = []
    for line in text.splitlines():
        segment = SEGMENT_PATTERN.match(line)
        if segment and segment.group(2) in mapping:
            stamp, speaker, sep, spoken = segment.groups()
            output.append(f"{stamp}{mapping[speaker]}{sep}{spoken}")
            continue

        summary = SUMMARY_PATTERN.match(line)
        if summary and summary.group(2) in mapping:
            indent, speaker, rest = summary.groups()
            output.append(f"{indent}{mapping[speaker]}{rest}")
            continue

        output.append(line)

    return "\n".join(output) + "\n"


def name_speakers_in_file(path):
    """Prompt for names and rewrite the transcript in place. True if changed.

    The file is only reopened for writing once the rename has been built in
    memory, so an abandoned prompt leaves the original untouched.
    """
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    segments = parse_segments(text)
    if not segments:
        print(f"No speaker-labelled lines found in {os.path.basename(path)}.")
        return False

    summaries = summarize_by_speaker(segments)
    mapping = prompt_for_names(summaries)
    if not mapping:
        print("No names entered; the transcript is unchanged.")
        return False

    warn_on_duplicates(mapping)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(rename_in_text(text, mapping))

    renamed = ", ".join(f"{old} -> {new}" for old, new in sorted(mapping.items()))
    print(f"Renamed {renamed}")
    return True


if __name__ == "__main__":
    sys.exit("Run name_speakers.py to name an existing transcript.")
