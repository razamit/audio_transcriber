"""plugin_runner.py: venv choice, status, install commands and the run privacy rule.

Nothing is installed and no model runs. A fake HostSystem stands in for the
machine: its process launcher records each command and writes what a real
child would write. The repo is a scratch folder with requirements files.
"""

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import plugin_runner
from claude_plugin import requirements_snapshot, venv_location
from claude_plugin.host_system import HostSystem
from claude_plugin.transcription_run import build_run_command, with_json_flag

SEGMENT_TEXT = "[00:00.000 --> 00:04.000] the secret merger closes on Friday"
CAPABILITIES = {"diarization": False, "engine": "mlx", "languages": ["he", "en"],
                "version": "1.1.0"}


class FakeProcesses:
    """subprocess.run stand-in. `respond(command)` returns (exit_code, stdout, stderr)."""

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def __call__(self, command, **options):
        self.calls.append((command, options))
        exit_code, stdout, stderr = self.respond(command)
        target = options.get("stderr")
        if hasattr(target, "write"):
            target.write(stderr.encode("utf-8"))
        if options.get("stdout") is subprocess.PIPE:
            stdout = stdout.encode("utf-8")
        return subprocess.CompletedProcess(command, exit_code, stdout=stdout)


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.mkdtemp(prefix="runner-repo-")
        self.data_dir = tempfile.mkdtemp(prefix="runner-data-")
        for folder in (self.repo, self.data_dir):
            self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        for name, text in (("requirements.txt", "mlx-whisper==0.4.3\n"),
                           ("requirements-diarize.txt", "pyannote.audio==4.0.7\n")):
            self.write_file(os.path.join(self.repo, name), text)
        self.processes = FakeProcesses(lambda command: (0, json.dumps(CAPABILITIES) + "\n", ""))

    def host(self, platform_name="darwin", environ=None, tools=("ffmpeg",)):
        return HostSystem(
            platform_name=platform_name, version_info=(3, 12, 10), system_python="sys-python",
            environ=environ or {}, run_process=self.processes,
            which=lambda name: f"/usr/bin/{name}" if name in tools else None,
        )

    @staticmethod
    def write_file(path, text=""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def make_venv(self, folder, platform_name="darwin"):
        python = venv_location.python_in_venv(folder, platform_name)
        self.write_file(python)
        return python

    def repo_venv(self, platform_name="darwin"):
        return self.make_venv(os.path.join(self.repo, ".venv"), platform_name)

    def data_venv(self, platform_name="darwin"):
        return self.make_venv(os.path.join(self.data_dir, "venv"), platform_name)

    def run_main(self, argv, host=None):
        printed = io.StringIO()
        code = plugin_runner.main(argv, host=host or self.host(), repo_folder=self.repo,
                                  stdout=printed)
        output = printed.getvalue()
        self.assertEqual(output.count("\n"), 1, "stdout must be exactly one line")
        return code, json.loads(output), output


class VenvResolutionTests(RunnerTestCase):
    def test_nothing_installed(self):
        self.assertEqual(venv_location.resolve_venv(self.repo, self.data_dir, "darwin"),
                         (None, None))

    def test_data_venv_when_the_repo_has_none(self):
        python = self.data_venv()
        self.assertEqual(venv_location.resolve_venv(self.repo, self.data_dir, "darwin"),
                         (python, "data"))

    def test_repo_venv_wins_over_the_data_venv(self):
        self.data_venv()
        python = self.repo_venv()
        self.assertEqual(venv_location.resolve_venv(self.repo, self.data_dir, "darwin"),
                         (python, "repo"))

    def test_windows_looks_for_scripts_python_exe(self):
        self.data_venv(platform_name="darwin")  # bin/python means nothing on Windows
        self.assertEqual(venv_location.resolve_venv(self.repo, self.data_dir, "win32"),
                         (None, None))
        python = self.data_venv(platform_name="win32")
        self.assertTrue(python.endswith(os.path.join("venv", "Scripts", "python.exe")))
        self.assertEqual(venv_location.resolve_venv(self.repo, self.data_dir, "win32"),
                         (python, "data"))

    def test_install_target(self):
        self.assertEqual(venv_location.install_target(self.repo, self.data_dir, "linux"),
                         (os.path.join(self.data_dir, "venv"), "data"))
        self.repo_venv(platform_name="linux")
        self.assertEqual(venv_location.install_target(self.repo, self.data_dir, "linux"),
                         (None, "repo"))


class StatusTests(RunnerTestCase):
    def status(self, **host_options):
        code, report, _ = self.run_main(["status", "--data-dir", self.data_dir],
                                        host=self.host(**host_options))
        self.assertEqual(code, 0)
        return report

    def test_nothing_installed(self):
        report = self.status()
        self.assertFalse(report["installed"])
        self.assertIsNone(report["venv_python"])
        self.assertIsNone(report["venv_location"])
        self.assertIsNone(report["capabilities"])
        self.assertFalse(report["requirements_changed"])
        self.assertTrue(report["ffmpeg"])
        self.assertIsNone(report["ffmpeg_hint"])
        self.assertTrue(report["python_ok"])
        self.assertEqual(report["platform"], "darwin")
        self.assertEqual(self.processes.calls, [], "status must not start anything when not installed")

    def test_installed_reads_capabilities_from_the_venv_python(self):
        python = self.repo_venv()
        report = self.status()
        self.assertEqual((report["installed"], report["venv_location"]), (True, "repo"))
        self.assertEqual(report["capabilities"], CAPABILITIES)
        command, _ = self.processes.calls[0]
        self.assertEqual(command, [python, "-B", os.path.join(self.repo, "transcribe.py"),
                                   "--capabilities"])

    def test_capabilities_null_when_the_venv_fails(self):
        self.repo_venv()
        self.processes.respond = lambda command: (1, "", "Traceback")
        self.assertIsNone(self.status()["capabilities"])

    def test_missing_ffmpeg_gives_the_platform_hint(self):
        report = self.status(platform_name="win32", tools=())
        self.assertFalse(report["ffmpeg"])
        self.assertIn("winget install ffmpeg", report["ffmpeg_hint"])

    def test_default_language_only_when_valid(self):
        self.assertEqual(self.status(environ={"TRANSCRIBE_LANG": " EN "})["default_language"], "en")
        self.assertIsNone(self.status(environ={"TRANSCRIBE_LANG": "fr"})["default_language"])
        self.assertIsNone(self.status()["default_language"])

    def test_nvidia_gpu_only_counts_on_windows_and_linux(self):
        tools = ("ffmpeg", "nvidia-smi")
        self.assertTrue(self.status(platform_name="win32", tools=tools)["nvidia_gpu_detected"])
        self.assertTrue(self.status(platform_name="linux", tools=tools)["nvidia_gpu_detected"])
        self.assertFalse(self.status(platform_name="darwin", tools=tools)["nvidia_gpu_detected"])
        self.assertFalse(self.status(platform_name="linux")["nvidia_gpu_detected"])

    def test_old_python_is_reported(self):
        host = self.host()
        old = HostSystem(**{**host.__dict__, "version_info": (3, 9, 6)})
        _, report, _ = self.run_main(["status", "--data-dir", self.data_dir], host=old)
        self.assertFalse(report["python_ok"])


class RequirementsSnapshotTests(RunnerTestCase):
    def requirements_changed(self):
        _, report, _ = self.run_main(["status", "--data-dir", self.data_dir])
        return report["requirements_changed"]

    def test_data_venv_without_a_snapshot_counts_as_changed(self):
        self.data_venv()
        self.assertTrue(self.requirements_changed())

    def test_snapshot_matches_then_a_requirements_edit_is_noticed(self):
        self.data_venv()
        requirements_snapshot.save_snapshot(self.data_dir, self.repo)
        self.assertFalse(self.requirements_changed())
        self.write_file(os.path.join(self.repo, "requirements-diarize.txt"), "pyannote.audio==5\n")
        self.assertTrue(self.requirements_changed())

    def test_a_repo_venv_is_never_reported_as_changed(self):
        self.repo_venv()
        self.assertFalse(self.requirements_changed())


class InstallTests(RunnerTestCase):
    def install(self, *flags, exit_code=0, platform_name="darwin"):
        self.processes.respond = lambda command: (exit_code, "", "")
        argv = ["install", "--data-dir", self.data_dir, *flags]
        return self.run_main(argv, host=self.host(platform_name=platform_name))

    def test_first_install_goes_to_the_data_folder_and_saves_a_snapshot(self):
        code, report, _ = self.install("--with-diarization", "--gpu", platform_name="linux")
        command, options = self.processes.calls[0]
        self.assertEqual(command, ["sys-python", "-B", os.path.join(self.repo, "install.py"),
                                   "--with-diarization", "--gpu",
                                   "--venv-dir", os.path.join(self.data_dir, "venv")])
        self.assertEqual(options["cwd"], self.repo)
        self.assertEqual((code, report["ok"], report["venv_location"]), (0, True, "data"))
        self.assertEqual(report["log"], os.path.join(self.data_dir, "logs", "install.log"))
        self.assertNotIn("log_tail", report)
        self.assertIsNotNone(requirements_snapshot.load_snapshot(self.data_dir))

    def test_an_existing_repo_venv_is_upgraded_in_place(self):
        self.repo_venv()
        code, report, _ = self.install()
        command, _ = self.processes.calls[0]
        self.assertEqual(command, ["sys-python", "-B", os.path.join(self.repo, "install.py")])
        self.assertEqual((code, report["venv_location"]), (0, "repo"))
        self.assertIsNone(requirements_snapshot.load_snapshot(self.data_dir))

    def test_failure_reports_the_log_tail_and_saves_no_snapshot(self):
        self.processes.respond = lambda command: (1, "", "")
        code, report, _ = self.install(exit_code=1)
        self.assertEqual((code, report["ok"], report["exit_code"]), (1, False, 1))
        self.assertIn("install:", report["log_tail"])  # the start line, at least
        self.assertIsNone(requirements_snapshot.load_snapshot(self.data_dir))


class RunPrivacyTests(RunnerTestCase):
    """The rule plugin_runner exists for: transcript text never reaches stdout."""

    def setUp(self):
        super().setUp()
        self.python = self.repo_venv()

    def run_with_child(self, exit_code, child_stderr, child_stdout=""):
        self.processes.respond = lambda command: (exit_code, child_stdout, child_stderr)
        code, report, output = self.run_main(
            ["run", "--data-dir", self.data_dir, "--", "--file", "/a/call.m4a", "--lang", "en"])
        return code, report, output

    def log_text(self):
        with open(os.path.join(self.data_dir, "logs", "last-run.log"), encoding="utf-8") as handle:
            return handle.read()

    def test_success_reports_the_result_and_keeps_segments_in_the_log(self):
        result = {"output": "/a/call.txt", "language": "en", "speakers": None, "elapsed_s": 3.2}
        code, report, output = self.run_with_child(0, SEGMENT_TEXT + "\n", json.dumps(result))
        self.assertEqual((code, report["ok"], report["result"]), (0, True, result))
        self.assertNotIn("secret merger", output)
        self.assertIn("secret merger", self.log_text())
        self.assertTrue(self.log_text().startswith("["), "the log opens with the start line")

    def test_exit_1_shows_only_the_failure_section(self):
        stderr = f"{SEGMENT_TEXT}\n\nFailed:\n\nTraceback (most recent call last):\nRuntimeError: boom\n"
        code, report, output = self.run_with_child(1, stderr)
        self.assertEqual((code, report["ok"]), (1, False))
        self.assertTrue(report["message"].startswith("Failed:"))
        self.assertIn("RuntimeError: boom", report["message"])
        self.assertNotIn("secret merger", output)

    def test_exit_1_without_the_marker_gives_only_the_log_path(self):
        code, report, output = self.run_with_child(1, SEGMENT_TEXT + "\nModel download failed\n")
        self.assertEqual(code, 1)
        self.assertIn(os.path.join("logs", "last-run.log"), report["message"])
        self.assertNotIn("secret merger", output)
        self.assertNotIn("download failed", output)

    def test_a_killed_child_gives_only_the_log_path(self):
        code, report, output = self.run_with_child(-9, SEGMENT_TEXT + "\n")
        self.assertEqual((code, report["exit_code"]), (1, -9))
        self.assertNotIn("secret merger", output)

    def test_validation_exit_codes_show_the_child_message(self):
        # cli.py raises these before any model runs, so the stderr holds no transcript.
        for exit_code, message in ((2, "Not a file: /a/call.m4a"),
                                   (3, "Speaker labelling is not installed."),
                                   (4, "ffmpeg is required to decode audio")):
            with self.subTest(exit_code=exit_code):
                code, report, _ = self.run_with_child(exit_code, message + "\n")
                self.assertEqual((code, report["exit_code"], report["message"]),
                                 (exit_code, exit_code, message))

    def test_exit_0_without_a_json_line_never_echoes_stdout(self):
        code, report, output = self.run_with_child(0, "", SEGMENT_TEXT + "\n")
        self.assertEqual((code, report["ok"]), (1, False))
        self.assertNotIn("secret merger", output)

    def test_command_passes_flags_through_and_adds_json_once(self):
        self.run_with_child(0, "", json.dumps({"output": "/a/call.txt"}))
        command, options = self.processes.calls[0]
        self.assertEqual(command, [self.python, "-B", os.path.join(self.repo, "transcribe.py"),
                                   "--file", "/a/call.m4a", "--lang", "en", "--json"])
        self.assertEqual(options["env"]["PYTHONUNBUFFERED"], "1")

    def test_json_flag_is_appended_once(self):
        self.assertEqual(with_json_flag(["--lang", "en"]), ["--lang", "en", "--json"])
        self.assertEqual(with_json_flag(["--json", "--lang", "en"]), ["--json", "--lang", "en"])
        self.assertEqual(build_run_command("py", "/r", ["--json"]).count("--json"), 1)


class ArgumentTests(RunnerTestCase):
    def test_run_without_an_install_is_a_json_error(self):
        code, report, _ = self.run_main(["run", "--data-dir", self.data_dir, "--", "--file", "x"])
        self.assertEqual((code, report["error"]), (1, "not_installed"))

    def test_flags_pass_through_without_the_separator(self):
        self.repo_venv()
        self.processes.respond = lambda command: (0, json.dumps({"output": "/a.txt"}), "")
        self.run_main(["run", "--data-dir", self.data_dir, "--file", "/a b.m4a", "--lang", "he"])
        command, _ = self.processes.calls[0]
        self.assertEqual(command[3:], ["--file", "/a b.m4a", "--lang", "he", "--json"])

    def test_an_unsubstituted_data_dir_is_refused(self):
        code, report, _ = self.run_main(["status", "--data-dir", "${CLAUDE_PLUGIN_DATA}"])
        self.assertEqual((code, report["error"]), (2, "usage"))
        self.assertFalse(os.path.exists("${CLAUDE_PLUGIN_DATA}"))

    def test_bad_arguments_are_json_not_usage_text(self):
        code, report, _ = self.run_main(["status"])
        self.assertEqual((code, report["error"]), (2, "usage"))
        code, report, _ = self.run_main(["status", "--data-dir", self.data_dir, "--surprise"])
        self.assertEqual(code, 2)

    def test_exit_codes_the_os_cannot_carry_become_1(self):
        self.assertEqual(plugin_runner.process_exit_code(3), 3)
        self.assertEqual(plugin_runner.process_exit_code(-9), 1)


if __name__ == "__main__":
    unittest.main()
