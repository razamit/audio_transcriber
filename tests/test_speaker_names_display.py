"""Speaker naming shows Hebrew in visual order but stores names in logical order."""

import contextlib
import io
import os
import unittest
from unittest import mock

import speaker_names
from tests.support import ScratchFolderTestCase

SENTENCE = "אני חושב שהפגישה הייתה טובה מאוד היום"
SENTENCE_VISUAL = SENTENCE[::-1]
NAME = "דני"
NAME_VISUAL = NAME[::-1]

TRANSCRIPT = (
    f"[00:00:01] SPEAKER_00: {SENTENCE}\n"
    "[00:00:09] SPEAKER_01: Short English line that is long enough.\n"
)


class SpeakerNamesDisplayTests(ScratchFolderTestCase):
    def setUp(self):
        super().setUp()
        self.start_patch(mock.patch.dict(os.environ, {"TRANSCRIBE_BIDI": "on", "COLUMNS": "200"}))
        self.transcript_path = os.path.join(self.folder, "call_speakers.txt")
        with open(self.transcript_path, "w", encoding="utf-8") as handle:
            handle.write(TRANSCRIPT)

    def run_naming(self, answers):
        printed = io.StringIO()
        with mock.patch("builtins.input", side_effect=list(answers)), \
                contextlib.redirect_stdout(printed):
            changed = speaker_names.name_speakers_in_file(self.transcript_path)
        with open(self.transcript_path, encoding="utf-8") as handle:
            return changed, printed.getvalue(), handle.read()

    def test_samples_and_renamed_line_are_shown_in_visual_order(self):
        changed, printed, _ = self.run_naming([NAME, ""])
        self.assertTrue(changed)
        self.assertIn(f"  [00:00:01] {SENTENCE_VISUAL}\n", printed)
        self.assertNotIn(SENTENCE, printed)
        self.assertIn("  [00:00:09] Short English line that is long enough.\n", printed)
        self.assertIn(f"Renamed SPEAKER_00 -> {NAME_VISUAL}\n", printed)

    def test_names_are_stored_in_logical_order(self):
        _, _, written = self.run_naming([NAME, ""])
        self.assertIn(f"[00:00:01] {NAME}: {SENTENCE}\n", written)

    def test_prompt_mapping_keeps_the_typed_name(self):
        summaries = speaker_names.summarize_by_speaker(speaker_names.parse_segments(TRANSCRIPT))
        with mock.patch("builtins.input", side_effect=[NAME, ""]), \
                contextlib.redirect_stdout(io.StringIO()):
            mapping = speaker_names.prompt_for_names(summaries)
        self.assertEqual(mapping, {"SPEAKER_00": NAME})

    def test_long_sample_is_cut_in_logical_order_before_reordering(self):
        long_sentence = " ".join(["מילה"] * 30)
        summaries = [{"speaker": "SPEAKER_00", "count": 1,
                      "samples": [("00:00:01", long_sentence)]}]
        printed = io.StringIO()
        with mock.patch("builtins.input", side_effect=[""]), contextlib.redirect_stdout(printed):
            speaker_names.prompt_for_names(summaries)
        expected = (long_sentence[:87] + "...")
        self.assertIn(f"  [00:00:01] {speaker_names.for_display(expected, io.StringIO())}\n",
                      printed.getvalue())
        # The ellipsis ends the sentence, so it sits at the left end in visual order.
        self.assertIn("  [00:00:01] ...ימ הלימ", printed.getvalue())

    def test_duplicate_name_note_shows_the_name_in_visual_order(self):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            speaker_names.warn_on_duplicates({"SPEAKER_00": NAME, "SPEAKER_01": NAME})
        self.assertIn(f"both named '{NAME_VISUAL}'; they stay", printed.getvalue())

    def test_output_is_logical_when_reordering_is_off(self):
        with mock.patch.dict(os.environ, {"TRANSCRIBE_BIDI": "off"}):
            _, printed, _ = self.run_naming([NAME, ""])
        self.assertIn(f"  [00:00:01] {SENTENCE}\n", printed)
        self.assertIn(f"Renamed SPEAKER_00 -> {NAME}\n", printed)


if __name__ == "__main__":
    unittest.main()
