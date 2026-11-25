"""Transcript rendering: invisible bidi control characters never reach the file."""

import os
import unittest

from tests.support import ScratchFolderTestCase
from transcript_output import (
    clean_text,
    render_labelled_segments,
    render_segments,
    write_transcript,
)

RLE = "‫"
PDF = "‬"
RLM = "‏"
LRI = "⁦"
PDI = "⁩"


class CleanTextTests(unittest.TestCase):
    def test_strips_controls_at_the_start_and_in_the_middle(self):
        # The shape the ivrit-ai model produces: RLE before a word mid-line.
        self.assertEqual(clean_text(f" {RLE}הקבוצה הזאת {RLE}שאנחנו{PDF} "),
                         "הקבוצה הזאת שאנחנו")

    def test_strips_marks_and_isolates(self):
        self.assertEqual(clean_text(f"{RLM}שלום {LRI}PMO{PDI}"), "שלום PMO")

    def test_text_without_controls_is_only_trimmed(self):
        self.assertEqual(clean_text("  hello, world  "), "hello, world")

    def test_none_becomes_empty(self):
        self.assertEqual(clean_text(None), "")


class RenderingTests(ScratchFolderTestCase):
    def test_plain_transcript_has_no_controls(self):
        body = render_segments([{"start": 0.0, "end": 1.0, "text": f"{RLE}בוקר טוב."}])
        self.assertEqual(body, "[00:00:00] בוקר טוב.\n")

    def test_a_segment_of_only_controls_is_dropped(self):
        body = render_segments([{"start": 0.0, "end": 1.0, "text": f"{RLE}{PDF}"},
                                {"start": 1.0, "end": 2.0, "text": "שלום"}])
        self.assertEqual(body, "[00:00:01] שלום\n")

    def test_labelled_transcript_has_no_controls(self):
        lines = render_labelled_segments(
            [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": f"{RLE}שלום"}])
        self.assertEqual(lines, ["[00:00:00] SPEAKER_00: שלום"])

    def test_a_labelled_segment_of_only_controls_is_dropped(self):
        lines = render_labelled_segments(
            [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": RLE},
             {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "שלום"}])
        self.assertEqual(lines, ["[00:00:01] SPEAKER_00: שלום"])

    def test_fallback_text_is_cleaned_too(self):
        path = os.path.join(self.folder, "call.txt")
        write_transcript(path, [], f"{RLE}שלום{PDF}")
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "שלום\n")


if __name__ == "__main__":
    unittest.main()
