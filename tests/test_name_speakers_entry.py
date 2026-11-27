"""name_speakers.py: a transcript path as the first argument, and the pause before closing."""

import contextlib
import io
import os
import unittest
from unittest import mock

import name_speakers
from tests.support import ScratchFolderTestCase

TRANSCRIPT = (
    "2 speaker(s) detected:\n"
    "  SPEAKER_00  00:00:10  (60%)\n"
    "  SPEAKER_01  00:00:06  (40%)\n"
    "\n"
    "[00:00:01] SPEAKER_00: A first line that is long enough to be a sample.\n"
    "[00:00:05] SPEAKER_01: A second line that is long enough to be a sample.\n"
)
TRICKY_NAME = "2026-11-26 it's @ שיחה_speakers.txt"


class NameSpeakersEntryTests(ScratchFolderTestCase):
    def setUp(self):
        super().setUp()
        self.start_patch(mock.patch.dict(os.environ, {"TRANSCRIBE_BIDI": "off"}))
        self.transcript_path = os.path.join(self.folder, "My Drive", TRICKY_NAME)
        os.makedirs(os.path.dirname(self.transcript_path))
        with open(self.transcript_path, "w", encoding="utf-8") as handle:
            handle.write(TRANSCRIPT)

    def run_main(self, argv, answers):
        printed = io.StringIO()
        answers = list(answers)
        prompts = []

        def answer(prompt=""):
            prompts.append(prompt)
            if not answers:
                raise EOFError
            return answers.pop(0)

        with mock.patch("builtins.input", side_effect=answer), \
                contextlib.redirect_stdout(printed):
            name_speakers.main(argv)
        return printed.getvalue(), prompts

    def transcript_text(self):
        with open(self.transcript_path, encoding="utf-8") as handle:
            return handle.read()

    def test_path_argument_names_that_file_then_waits_for_enter(self):
        printed, prompts = self.run_main([self.transcript_path], ["Dana", "", ""])
        self.assertIn("[00:00:01] Dana: A first line", self.transcript_text())
        self.assertIn(f"Updated: {self.transcript_path}", printed)
        self.assertEqual(prompts[-1], name_speakers.CLOSE_PROMPT)
        self.assertEqual(len(prompts), 3, "two names, then the close prompt; no path prompt")

    def test_close_prompt_comes_after_the_renamed_line(self):
        printed, prompts = self.run_main([self.transcript_path], ["Dana", "Avi", ""])
        self.assertIn("Renamed SPEAKER_00 -> Dana, SPEAKER_01 -> Avi", printed)
        self.assertEqual(prompts[-1], name_speakers.CLOSE_PROMPT)

    def test_missing_path_is_reported_and_still_waits(self):
        missing = os.path.join(self.folder, "gone_speakers.txt")
        printed, prompts = self.run_main([missing], [""])
        self.assertIn(f"No such file: {missing}", printed)
        self.assertEqual(prompts, [name_speakers.CLOSE_PROMPT])

    def test_an_error_while_naming_is_shown_before_the_pause(self):
        with mock.patch.object(name_speakers, "name_speakers_in_file",
                               side_effect=PermissionError("read-only")):
            printed, prompts = self.run_main([self.transcript_path], [""])
        self.assertIn("Could not name the speakers: read-only", printed)
        self.assertEqual(prompts, [name_speakers.CLOSE_PROMPT])

    def test_closing_the_input_at_the_pause_is_not_an_error(self):
        printed, _ = self.run_main([self.transcript_path], ["", ""])  # EOF at the close prompt
        self.assertIn("No names entered; the transcript is unchanged.", printed)
        self.assertEqual(self.transcript_text(), TRANSCRIPT)

    def test_without_a_path_it_asks_for_one_and_does_not_pause(self):
        printed, prompts = self.run_main([], [self.transcript_path, "Dana", ""])
        self.assertTrue(prompts[0].startswith("Path to diarized transcript"))
        self.assertNotIn(name_speakers.CLOSE_PROMPT, prompts)
        self.assertIn(f"Updated: {self.transcript_path}", printed)


if __name__ == "__main__":
    unittest.main()
