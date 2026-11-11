"""No arguments means the interactive mode, exactly as before the CLI existed."""

import contextlib
import io
import os
import unittest
from unittest import mock

import transcribe
from tests.support import ScratchFolderTestCase

FAKE_WHISPER_RESULT = {
    "segments": [{"start": 0.0, "end": 1.0, "text": " hello"}],
    "text": "hello",
}


class InteractiveModeTests(ScratchFolderTestCase):
    def setUp(self):
        super().setUp()
        self.start_patch(mock.patch.dict(os.environ, {}, clear=False))
        os.environ.pop("TRANSCRIBE_LANG", None)
        self.start_patch(mock.patch.object(transcribe, "require_ffmpeg"))
        self.start_patch(mock.patch.object(transcribe, "announce_first_run_download"))
        self.start_patch(mock.patch.object(
            transcribe, "run_transcription", return_value=FAKE_WHISPER_RESULT))

    def run_main(self, argv, answers):
        prompts = mock.MagicMock(side_effect=list(answers))
        self.printed = io.StringIO()
        with mock.patch("builtins.input", prompts), \
                contextlib.redirect_stdout(self.printed), contextlib.redirect_stderr(io.StringIO()):
            transcribe.main(argv)
        return [call.args[0] for call in prompts.call_args_list]

    def test_empty_argv_runs_the_prompts_and_writes_next_to_audio(self):
        asked = self.run_main([], ["2", self.audio_path, "n"])
        self.assertEqual(len(asked), 3)
        self.assertTrue(asked[0].startswith("Language:"))
        self.assertEqual(asked[1], "Path to audio file: ")
        self.assertEqual(asked[2], "Label speakers? [y/N]: ")
        with open(os.path.join(self.folder, "call.txt"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "[00:00:00] hello\n")
        closing = self.printed.getvalue().splitlines()[-2:]
        self.assertEqual(closing[0], "Transcript written to: " + os.path.join(self.folder, "call.txt"))
        self.assertTrue(closing[1].startswith("Elapsed: 00:00:00 ("))

    def test_speaker_flow_still_asks_count_and_offers_naming(self):
        with mock.patch.object(transcribe.diarization, "diarization_is_available", return_value=True), \
                mock.patch.object(transcribe, "transcribe_with_speakers") as with_speakers, \
                mock.patch.object(transcribe, "name_speakers_in_file") as naming:
            asked = self.run_main([], ["1", self.audio_path, "y", ""])
        self.assertEqual(asked[3], "Number of speakers (press Enter to auto-detect): ")
        audio, profile, output, count = with_speakers.call_args.args
        self.assertEqual((profile.key, count), ("he", None))
        self.assertTrue(output.endswith("call_speakers.txt"))
        naming.assert_called_once_with(output)

    def test_no_sys_argv_arguments_means_interactive(self):
        with mock.patch.object(transcribe.sys, "argv", ["transcribe.py"]), \
                mock.patch.object(transcribe, "run_interactive") as interactive:
            transcribe.main()
        interactive.assert_called_once_with()

    def test_any_argument_never_prompts(self):
        with mock.patch.object(transcribe.sys, "argv", ["transcribe.py", "--capabilities"]), \
                mock.patch.object(transcribe, "run_interactive") as interactive, \
                mock.patch("builtins.input", side_effect=AssertionError("prompted")), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                transcribe.main()
        self.assertEqual(caught.exception.code, 0)
        interactive.assert_not_called()

    def test_flag_run_with_speakers_skips_naming(self):
        argv = ["--file", self.audio_path, "--lang", "en", "--speakers", "auto"]
        with mock.patch.object(transcribe.diarization, "diarization_is_available", return_value=True), \
                mock.patch("cli.shutil.which", return_value="/usr/bin/ffmpeg"), \
                mock.patch("cli.announce_first_run_download"), \
                mock.patch.object(transcribe, "transcribe_with_speakers", return_value=2) as with_speakers, \
                mock.patch.object(transcribe, "name_speakers_in_file") as naming, \
                mock.patch("builtins.input", side_effect=AssertionError("prompted")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                transcribe.main(argv)
        self.assertEqual(caught.exception.code, 0)
        with_speakers.assert_called_once()
        naming.assert_not_called()


if __name__ == "__main__":
    unittest.main()
