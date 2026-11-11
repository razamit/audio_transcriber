"""A full non-interactive run with fake transcribers: result line, stdout, no prompts."""

import contextlib
import io
import json
import os
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

import cli
from tests.support import FakeTranscribers, ScratchFolderTestCase

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RunTests(ScratchFolderTestCase):
    def setUp(self):
        super().setUp()
        self.start_patch(mock.patch.object(cli.shutil, "which", return_value="/usr/bin/ffmpeg"))
        self.start_patch(mock.patch.object(cli, "announce_first_run_download"))
        self.start_patch(mock.patch.object(
            cli.diarization, "diarization_is_available", return_value=True))
        # Any prompt in non-interactive mode is a bug, so make one fail loudly.
        self.start_patch(mock.patch("builtins.input", side_effect=AssertionError("prompted")))
        self.fakes = FakeTranscribers(speaker_count=2, noise="segment noise")

    def run_cli(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.run(argv, self.fakes.plain, self.fakes.with_speakers)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_json_result_without_speakers(self):
        code, out, err = self.run_cli(["--file", self.audio_path, "--lang", "en", "--json"])
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertEqual(set(result), {"output", "language", "speakers", "elapsed_s"})
        self.assertEqual(result["output"], os.path.join(self.folder, "call.txt"))
        self.assertEqual(result["language"], "en")
        self.assertIsNone(result["speakers"])
        self.assertIsInstance(result["elapsed_s"], float)
        self.assertIn("segment noise", err)

    def test_json_result_with_speakers(self):
        argv = ["--file", self.audio_path, "--lang", "he", "--speakers", "3", "--json"]
        code, out, _ = self.run_cli(argv)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["speakers"], 2)
        self.assertEqual(self.fakes.calls[0][0], "speakers")
        self.assertEqual(self.fakes.calls[0][4], 3)
        self.assertTrue(json.loads(out)["output"].endswith("call_speakers.txt"))

    def test_plain_result_is_only_the_path(self):
        code, out, _ = self.run_cli(["--file", self.audio_path, "--lang", "en"])
        self.assertEqual(code, 0)
        self.assertEqual(out, os.path.join(self.folder, "call.txt") + "\n")

    def test_existing_default_output_is_not_overwritten(self):
        with open(os.path.join(self.folder, "call.txt"), "w") as handle:
            handle.write("older transcript")
        _, out, _ = self.run_cli(["--file", self.audio_path, "--lang", "en"])
        self.assertEqual(out.strip(), os.path.join(self.folder, "call_2.txt"))

    def test_out_is_used_exactly(self):
        target = os.path.join(self.folder, "exact name.txt")
        _, out, _ = self.run_cli(["--file", self.audio_path, "--lang", "en", "--out", target])
        self.assertEqual(out.strip(), target)
        self.assertTrue(os.path.exists(target))

    def test_relative_out_resolves_against_working_folder(self):
        previous = os.getcwd()
        os.chdir(self.folder)
        self.addCleanup(os.chdir, previous)
        _, out, _ = self.run_cli(["--file", "call.m4a", "--lang", "en", "--out", "rel.txt"])
        self.assertEqual(os.path.realpath(out.strip()), os.path.realpath(
            os.path.join(self.folder, "rel.txt")))

    def test_failed_run_leaves_no_empty_out_file(self):
        def crashes(*args):
            raise RuntimeError("boom")
        self.fakes.plain = crashes
        target = os.path.join(self.folder, "never.txt")
        code, _, _ = self.run_cli(["--file", self.audio_path, "--lang", "en", "--out", target])
        self.assertEqual(code, 1)
        self.assertFalse(os.path.exists(target))

    def test_sys_exit_inside_transcription_exits_1(self):
        def dies(*args):
            sys.exit("No speech was transcribed; nothing to write.")
        self.fakes.plain = dies
        code, out, err = self.run_cli(["--file", self.audio_path, "--lang", "en", "--json"])
        self.assertEqual((code, out), (1, ""))
        self.assertIn("No speech was transcribed", err)

    def test_unexpected_exception_exits_1(self):
        def crashes(*args):
            raise RuntimeError("boom")
        self.fakes.plain = crashes
        code, out, err = self.run_cli(["--file", self.audio_path, "--lang", "en"])
        self.assertEqual((code, out), (1, ""))
        self.assertIn("boom", err)


class RealStdoutTests(ScratchFolderTestCase):
    """In a real process, even fd-level writes must not reach stdout."""

    SCRIPT = textwrap.dedent("""
        import os, sys
        from unittest import mock
        import cli

        def noisy(audio_path, profile, output_path):
            print("python-level noise")
            os.write(1, b"fd-level noise\\n")
            open(output_path, "w").write("x")

        with mock.patch.object(cli, "announce_first_run_download"), \\
                mock.patch.object(cli.shutil, "which", return_value="/usr/bin/ffmpeg"):
            sys.exit(cli.run(sys.argv[1:], noisy, None))
    """)

    def test_stdout_carries_only_the_result_line(self):
        completed = subprocess.run(
            [sys.executable, "-c", self.SCRIPT, "--file", self.audio_path, "--lang", "en", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        lines = completed.stdout.splitlines()
        self.assertEqual(len(lines), 1, completed.stdout)
        self.assertEqual(json.loads(lines[0])["language"], "en")
        self.assertIn("python-level noise", completed.stderr)
        self.assertIn("fd-level noise", completed.stderr)


if __name__ == "__main__":
    unittest.main()
