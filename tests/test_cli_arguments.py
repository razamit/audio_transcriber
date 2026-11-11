"""Argument parsing and the checks that run before any model loads."""

import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

import cli
from tests.support import ScratchFolderTestCase

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse(argv):
    return cli.build_parser().parse_args(argv)


def exit_code_of_bad_args(argv):
    """argparse exits on usage errors; capture its code and swallow its message."""
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        try:
            cli.run(argv, None, None)
        except SystemExit as exc:
            return exc.code
    raise AssertionError(f"expected a usage error for {argv}")


class ParseSpeakersTests(unittest.TestCase):
    def test_auto_in_any_case(self):
        self.assertEqual(cli.parse_speakers("auto"), "auto")
        self.assertEqual(cli.parse_speakers(" AUTO "), "auto")

    def test_positive_number(self):
        self.assertEqual(cli.parse_speakers("3"), 3)

    def test_rejects_zero_negative_and_words(self):
        for bad in ("0", "-2", "two", "", "2.5"):
            with self.subTest(bad=bad), self.assertRaises(Exception):
                cli.parse_speakers(bad)


class ParserTests(unittest.TestCase):
    def test_full_flag_set(self):
        args = parse(["--file", "a.m4a", "--lang", "EN", "--speakers", "auto",
                      "--out", "o.txt", "--json"])
        self.assertEqual((args.file, args.lang, args.speakers, args.out, args.json),
                         ("a.m4a", "en", "auto", "o.txt", True))

    def test_speakers_absent_means_none(self):
        self.assertIsNone(parse(["--file", "a.m4a"]).speakers)

    def test_usage_errors_exit_2(self):
        cases = [
            ["--file", "a.m4a", "--lang", "fr"],
            ["--file", "a.m4a", "--speakers", "many"],
            ["--file", "a.m4a", "--bogus"],
            ["--lang", "en"],  # no --file and no --capabilities
            ["--json"],
            ["--fi", "a.m4a", "--lang", "en"],  # abbreviations are not accepted
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                self.assertEqual(exit_code_of_bad_args(argv), 2)


class CapabilitiesTests(unittest.TestCase):
    def run_capabilities(self, available, argv=("--capabilities",)):
        stdout = io.StringIO()
        with mock.patch.object(cli.diarization, "diarization_is_available", return_value=available), \
                contextlib.redirect_stdout(stdout):
            code = cli.run(list(argv), None, None)
        return code, stdout.getvalue()

    def test_prints_one_json_line_and_exits_0(self):
        code, out = self.run_capabilities(True)
        self.assertEqual(code, 0)
        self.assertEqual(out.count("\n"), 1)
        self.assertEqual(json.loads(out), {
            "diarization": True, "languages": ["he", "en"], "version": cli.__version__,
        })

    def test_reports_missing_diarization(self):
        _, out = self.run_capabilities(False, ("--capabilities", "--json"))
        self.assertIs(json.loads(out)["diarization"], False)

    def test_missing_pyannote_parent_package_reads_as_not_installed(self):
        # find_spec("pyannote.audio") raises when "pyannote" itself is absent.
        no_parent = ModuleNotFoundError("No module named 'pyannote'")
        with mock.patch.object(cli.diarization.importlib.util, "find_spec", side_effect=no_parent):
            self.assertIs(cli.diarization_installed(), False)

    def test_base_install_capabilities_in_a_real_process(self):
        # -S drops site-packages, so neither pyannote nor its parent package exists.
        completed = subprocess.run(
            [sys.executable, "-S", "transcribe.py", "--capabilities"],
            cwd=REPO_ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIs(json.loads(completed.stdout)["diarization"], False)


class ValidationTests(ScratchFolderTestCase):
    """Each failure maps to its documented exit code, before any transcription."""

    def setUp(self):
        super().setUp()
        self.start_patch(mock.patch.object(cli.shutil, "which", return_value="/usr/bin/ffmpeg"))
        self.start_patch(mock.patch.dict(os.environ, {}, clear=False))
        os.environ.pop("TRANSCRIBE_LANG", None)
        self.diarization_available = self.start_patch(
            mock.patch.object(cli.diarization, "diarization_is_available", return_value=True))

    def code_for(self, argv):
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            return cli.run(argv, self.fail_if_called, self.fail_if_called)

    def fail_if_called(self, *args):
        self.fail("transcription must not start when validation fails")

    def test_missing_ffmpeg_exits_4(self):
        with mock.patch.object(cli.shutil, "which", return_value=None):
            self.assertEqual(self.code_for(["--file", self.audio_path, "--lang", "en"]), 4)

    def test_missing_or_directory_file_exits_2(self):
        missing = os.path.join(self.folder, "nope.m4a")
        self.assertEqual(self.code_for(["--file", missing, "--lang", "en"]), 2)
        self.assertEqual(self.code_for(["--file", self.folder, "--lang", "en"]), 2)

    def test_language_required_without_env(self):
        self.assertEqual(self.code_for(["--file", self.audio_path]), 2)

    def test_env_language_accepted_and_unknown_env_exits_2(self):
        os.environ["TRANSCRIBE_LANG"] = "en"
        self.assertEqual(cli.resolve_profile(None).key, "en")
        os.environ["TRANSCRIBE_LANG"] = "xx"
        self.assertEqual(self.code_for(["--file", self.audio_path]), 2)

    def test_lang_flag_beats_env(self):
        os.environ["TRANSCRIBE_LANG"] = "he"
        self.assertEqual(cli.resolve_profile("en").key, "en")

    def test_speakers_without_diarization_exits_3(self):
        self.diarization_available.return_value = False
        argv = ["--file", self.audio_path, "--lang", "en", "--speakers", "auto"]
        self.assertEqual(self.code_for(argv), 3)

    def test_speakers_with_pyannote_unimportable_exits_3(self):
        self.diarization_available.side_effect = ModuleNotFoundError("No module named 'pyannote'")
        argv = ["--file", self.audio_path, "--lang", "en", "--speakers", "2"]
        self.assertEqual(self.code_for(argv), 3)

    def test_out_pointing_at_a_folder_exits_2(self):
        self.assertEqual(
            self.code_for(["--file", self.audio_path, "--lang", "en", "--out", self.folder]), 2)

    def test_reserve_output_refuses_a_file_that_appeared_after_validation(self):
        late = os.path.join(self.folder, "late.txt")
        cli.reserve_output(late)
        with self.assertRaises(cli.CliError) as caught:
            cli.reserve_output(late)
        self.assertEqual(caught.exception.exit_code, 2)

    def test_existing_out_exits_2_and_is_untouched(self):
        out = os.path.join(self.folder, "taken.txt")
        with open(out, "w") as handle:
            handle.write("keep me")
        self.assertEqual(self.code_for(["--file", self.audio_path, "--lang", "en", "--out", out]), 2)
        with open(out) as handle:
            self.assertEqual(handle.read(), "keep me")

    def test_out_in_missing_folder_exits_2(self):
        out = os.path.join(self.folder, "no-such-dir", "t.txt")
        self.assertEqual(self.code_for(["--file", self.audio_path, "--lang", "en", "--out", out]), 2)

    def test_request_fields(self):
        request = cli.validate(parse(["--file", self.audio_path, "--lang", "en", "--speakers", "3"]))
        self.assertEqual(request.speaker_count, 3)
        self.assertTrue(request.output_path.endswith("call_speakers.txt"))
        auto = cli.validate(parse(["--file", self.audio_path, "--lang", "he", "--speakers", "auto"]))
        self.assertIsNone(auto.speaker_count)
        self.assertTrue(auto.with_speakers)


if __name__ == "__main__":
    unittest.main()
