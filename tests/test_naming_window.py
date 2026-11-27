"""plugin_runner.py name: the per-OS window commands and the fallback report.

No window opens. A fake HostSystem records the `open` call on macOS and the
Popen-style start on Windows and Linux. The one test that runs a generated
script replaces the naming program with printf, so it only echoes arguments.
"""

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest

from claude_plugin import naming_window, window_commands
from claude_plugin.host_system import HostSystem
from tests.test_plugin_runner import FakeProcesses, RunnerTestCase

TRICKY_PATH = ("/Users/x/Library/CloudStorage/GoogleDrive-a@b.com/My Drive/"
               "\u05e9\u05d9\u05d7\u05d5\u05ea/2026-11-26 it's \u05e9\u05d9\u05d7\u05d4_speakers.txt")
SECRET_LINE = "the secret merger closes on Friday"
SPEAKER_TRANSCRIPT = (
    "2 speaker(s) detected:\n"
    "  SPEAKER_00  00:00:10  (60%)\n"
    "  SPEAKER_01  00:00:06  (40%)\n"
    "\n"
    f"[00:00:01] SPEAKER_00: {SECRET_LINE}\n"
    "[00:00:05] SPEAKER_01: \u05d6\u05d4 \u05de\u05e9\u05e4\u05d8 \u05d1\u05d3\u05d9\u05e7\u05d4\n"
)


class FakeWindowProcess:
    """A started process: `exit_code` None means it is still running after the wait."""

    def __init__(self, exit_code):
        self.exit_code = exit_code

    def wait(self, timeout=None):
        if self.exit_code is None:
            raise subprocess.TimeoutExpired("window", timeout)
        return self.exit_code


class FakeStarter:
    """subprocess.Popen stand-in. `exit_codes` are used in order, one per start."""

    def __init__(self, exit_codes=(None,), error=None):
        self.exit_codes = list(exit_codes)
        self.error = error
        self.calls = []

    def __call__(self, argv, **options):
        self.calls.append((argv, options))
        if self.error:
            raise self.error
        return FakeWindowProcess(self.exit_codes.pop(0))


class WindowCommandTests(unittest.TestCase):
    def setUp(self):
        self.command = window_commands.naming_command("/v/bin/python", "/repo dir", TRICKY_PATH)

    def test_naming_command_passes_the_path_as_one_argument(self):
        self.assertEqual(self.command, ["/v/bin/python", "-B",
                                        os.path.join("/repo dir", "name_speakers.py"),
                                        TRICKY_PATH])

    def test_macos_script_quotes_every_part_for_sh(self):
        text = window_commands.macos_command_file_text(self.command)
        self.assertTrue(text.startswith("#!/bin/sh\n"))
        self.assertIn('rm -f -- "$0"\n', text)
        exec_line = [line for line in text.splitlines() if line.startswith("exec ")][0]
        self.assertEqual(shlex.split(exec_line)[1:], self.command)

    @unittest.skipUnless(os.name == "posix" and shutil.which("sh"), "needs /bin/sh")
    def test_macos_script_hands_sh_the_exact_arguments(self):
        folder = tempfile.mkdtemp(prefix="naming-script-")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        script = os.path.join(folder, "run.command")
        echo = ["printf", "%s\\n", "/repo dir/name_speakers.py", TRICKY_PATH]
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(window_commands.macos_command_file_text(echo))
        printed = subprocess.run(["sh", script], capture_output=True, check=True).stdout
        self.assertEqual(printed.decode("utf-8").splitlines(), echo[2:])
        self.assertFalse(os.path.exists(script), "the script deletes itself")

    def test_macos_opens_the_script_in_terminal_by_name(self):
        self.assertEqual(window_commands.macos_open_command("/d/naming/x.command"),
                         ["/usr/bin/open", "-a", "Terminal", "/d/naming/x.command"])

    def test_windows_uses_a_new_console_and_keeps_the_path_whole(self):
        command = [r"C:\Users\A B\venv\Scripts\python.exe", "-B", r"C:\p\name_speakers.py",
                   r"C:\Users\A B\My Drive\it's \u05e9\u05d9\u05d7\u05d4_speakers.txt"]
        argv, flags = window_commands.windows_launch(command)
        self.assertEqual((argv, flags), (command, 0x10))
        line = subprocess.list2cmdline(argv)
        self.assertIn(f'"{command[3]}"', line)
        if hasattr(subprocess, "CREATE_NEW_CONSOLE"):  # only defined on Windows
            self.assertEqual(flags, subprocess.CREATE_NEW_CONSOLE)

    def test_linux_tries_the_emulators_in_order_with_their_own_flag(self):
        found = {"xterm", "gnome-terminal", "x-terminal-emulator", "konsole", "xfce4-terminal"}
        candidates = window_commands.linux_launch_candidates(
            self.command, lambda name: f"/usr/bin/{name}" if name in found else None)
        self.assertEqual([(name, argv[1]) for name, argv in candidates],
                         [("x-terminal-emulator", "-e"), ("gnome-terminal", "--"),
                          ("konsole", "-e"), ("xfce4-terminal", "-x"), ("xterm", "-e")])
        for _, argv in candidates:
            self.assertEqual(argv[2:], self.command)

    def test_linux_without_emulators_has_no_candidates(self):
        self.assertEqual(window_commands.linux_launch_candidates(self.command, lambda name: None),
                         [])

    def test_manual_command_for_posix_shells(self):
        line = window_commands.manual_command(self.command, "darwin")
        self.assertEqual(shlex.split(line), self.command)

    def test_manual_command_for_powershell(self):
        line = window_commands.manual_command([r"C:\v\python.exe", "it's here"], "win32")
        self.assertEqual(line, r"& 'C:\v\python.exe' 'it''s here'")


class NamingWindowTestCase(RunnerTestCase):
    def setUp(self):
        super().setUp()
        self.python = self.repo_venv()
        self.starter = FakeStarter()
        self.transcript = os.path.join(self.data_dir, "My Drive", "it's \u05e9\u05d9\u05d7\u05d4_speakers.txt")
        self.write_file(self.transcript, SPEAKER_TRANSCRIPT)
        self.processes = FakeProcesses(self.record_command_file)
        self.command_file_text = None

    def record_command_file(self, command):
        """The fake `open`: keeps what the script said, as Terminal would run it."""
        with open(command[-1], encoding="utf-8") as handle:
            self.command_file_text = handle.read()
        return 0, "", ""

    def host(self, platform_name="darwin", environ=None, tools=()):
        return HostSystem(
            platform_name=platform_name, version_info=(3, 12, 10), system_python="sys-python",
            environ=environ or {}, run_process=self.processes, start_process=self.starter,
            which=lambda name: f"/usr/bin/{name}" if name in tools else None,
        )

    def name(self, transcript=None, separator=True, **host_options):
        argv = ["name", "--data-dir", self.data_dir]
        argv += (["--"] if separator else []) + ["--file", transcript or self.transcript]
        code, report, output = self.run_main(argv, host=self.host(**host_options))
        self.assertTrue(output.isascii(), "the JSON must be ASCII")
        self.assertNotIn(SECRET_LINE, output)
        return code, report

    def assert_fallback(self, report, reason):
        self.assertEqual((report["ok"], report["opened"], report["reason"]), (False, False, reason))
        self.assertIsNone(report["method"])


class MacosWindowTests(NamingWindowTestCase):
    def test_success_opens_a_command_file_in_terminal(self):
        code, report = self.name()
        self.assertEqual(code, 0)
        self.assertEqual((report["ok"], report["opened"], report["method"]),
                         (True, True, "macos_terminal_command_file"))
        command, options = self.processes.calls[0]
        self.assertEqual(command[:3], ["/usr/bin/open", "-a", "Terminal"])
        self.assertTrue(command[3].startswith(os.path.join(self.data_dir, "naming", "name-speakers-")))
        self.assertIs(options["stdout"], subprocess.DEVNULL)
        exec_line = self.command_file_text.splitlines()[-1]
        self.assertEqual(shlex.split(exec_line)[1:], [
            self.python, "-B", os.path.join(self.repo, "name_speakers.py"), self.transcript])
        self.assertEqual(shlex.split(report["manual_command"]), shlex.split(exec_line)[1:])
        self.assertEqual(self.starter.calls, [], "macOS never starts the program directly")

    def test_open_failing_falls_back_and_removes_the_script(self):
        self.processes.respond = lambda command: (1, "", "")
        code, report = self.name()
        self.assertEqual(code, 0)
        self.assert_fallback(report, "launch_failed")
        self.assertEqual(os.listdir(os.path.join(self.data_dir, "naming")), [])
        self.assertIn("name_speakers.py", report["manual_command"])

    def test_open_that_cannot_start_falls_back(self):
        def no_open(command):
            raise OSError("no such file")
        self.processes.respond = no_open
        _, report = self.name()
        self.assert_fallback(report, "launch_failed")

    def test_stale_scripts_are_cleared_and_fresh_ones_kept(self):
        folder = os.path.join(self.data_dir, "naming")
        stale, fresh = (os.path.join(folder, f"name-speakers-{tag}.command") for tag in ("a", "b"))
        for path in (stale, fresh):
            self.write_file(path, "#!/bin/sh\n")
        old = time.time() - 2 * naming_window.STALE_COMMAND_FILE_SECONDS
        os.utime(stale, (old, old))
        naming_window.remove_stale_command_files(folder)
        self.assertEqual(os.listdir(folder), ["name-speakers-b.command"])


class WindowsAndLinuxWindowTests(NamingWindowTestCase):
    def setUp(self):
        super().setUp()
        self.python = self.repo_venv(platform_name="win32")

    def test_windows_starts_a_new_console_without_the_old_terminal_markers(self):
        environ = {"WT_SESSION": "1", "TERM_PROGRAM": "vscode", "TRANSCRIBE_BIDI": "on", "PATH": "p"}
        _, report = self.name(platform_name="win32", environ=environ)
        self.assertEqual((report["opened"], report["method"]), (True, "windows_new_console"))
        argv, options = self.starter.calls[0]
        self.assertEqual(argv, [self.python, "-B", os.path.join(self.repo, "name_speakers.py"),
                                self.transcript])
        self.assertEqual(options["creationflags"], 0x10)
        self.assertEqual(options["env"], {"TRANSCRIBE_BIDI": "on", "PATH": "p"})
        self.assertTrue(report["manual_command"].startswith("& '"))

    def test_windows_child_that_dies_at_once_falls_back(self):
        self.starter.exit_codes = [1]
        _, report = self.name(platform_name="win32")
        self.assert_fallback(report, "launch_failed")

    def test_linux_moves_on_when_an_emulator_fails(self):
        self.starter.exit_codes = [1, None]
        tools = ("x-terminal-emulator", "gnome-terminal")
        _, report = self.name(platform_name="linux", environ={"DISPLAY": ":0"}, tools=tools)
        self.assertEqual((report["opened"], report["method"]), (True, "linux_gnome-terminal"))
        self.assertEqual([argv[0] for argv, _ in self.starter.calls],
                         ["/usr/bin/x-terminal-emulator", "/usr/bin/gnome-terminal"])
        _, options = self.starter.calls[1]
        self.assertTrue(options["start_new_session"])
        self.assertIs(options["stdout"], subprocess.DEVNULL)

    def test_linux_launcher_that_exits_0_counts_as_opened(self):
        self.starter.exit_codes = [0]
        _, report = self.name(platform_name="linux", environ={"WAYLAND_DISPLAY": "wayland-0"},
                              tools=("gnome-terminal",))
        self.assertEqual(report["method"], "linux_gnome-terminal")

    def test_linux_without_a_display(self):
        _, report = self.name(platform_name="linux", tools=("xterm",))
        self.assert_fallback(report, "no_display")
        self.assertEqual(self.starter.calls, [])

    def test_linux_without_an_emulator(self):
        _, report = self.name(platform_name="linux", environ={"DISPLAY": ":0"})
        self.assert_fallback(report, "no_terminal_emulator")

    def test_linux_when_every_emulator_fails(self):
        self.starter.error = OSError("exec format error")
        _, report = self.name(platform_name="linux", environ={"DISPLAY": ":0"}, tools=("xterm",))
        self.assert_fallback(report, "launch_failed")


class NamingFallbackTests(NamingWindowTestCase):
    def test_remote_session_never_tries_a_window(self):
        self.repo_venv(platform_name="win32")
        for platform_name in ("darwin", "win32", "linux"):
            with self.subTest(platform_name=platform_name):
                environ = {"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22", "DISPLAY": ":10"}
                _, report = self.name(platform_name=platform_name, environ=environ)
                self.assert_fallback(report, "remote_session")
        self.assertEqual((self.processes.calls, self.starter.calls), ([], []))

    def test_missing_file(self):
        _, report = self.name(transcript=os.path.join(self.data_dir, "gone_speakers.txt"))
        self.assert_fallback(report, "file_not_found")

    def test_transcript_without_speaker_labels(self):
        plain = os.path.join(self.data_dir, "call.txt")
        self.write_file(plain, f"[00:00:01] {SECRET_LINE}\n")
        _, report = self.name(transcript=plain)
        self.assert_fallback(report, "not_a_speaker_transcript")

    def test_not_installed_has_no_manual_command(self):
        os.remove(self.python)
        _, report = self.name()
        self.assert_fallback(report, "not_installed")
        self.assertIsNone(report["manual_command"])

    def test_a_host_that_cannot_start_processes(self):
        host = HostSystem(platform_name="win32", version_info=(3, 12, 10), system_python="py",
                          environ={}, which=lambda name: None, run_process=self.processes)
        self.repo_venv(platform_name="win32")
        _, report, _ = self.run_main(["name", "--data-dir", self.data_dir, "--",
                                      "--file", self.transcript], host=host)
        self.assert_fallback(report, "launch_failed")

    def test_flags_work_without_the_separator(self):
        _, report = self.name(separator=False)
        self.assertTrue(report["opened"])

    def test_missing_file_flag_is_a_usage_error(self):
        code, report, _ = self.run_main(["name", "--data-dir", self.data_dir, "--"],
                                        host=self.host())
        self.assertEqual((code, report["error"]), (2, "usage"))

    def test_hebrew_path_is_escaped_in_the_json_and_whole_in_the_command(self):
        self.processes.respond = lambda command: (1, "", "")
        code, report, output = self.run_main(
            ["name", "--data-dir", self.data_dir, "--", "--file", self.transcript])
        self.assertIn("\\u05e9", output)
        self.assertNotIn("ש", output)
        self.assertEqual(shlex.split(report["manual_command"])[-1], self.transcript)
        self.assertEqual(json.loads(output), report)


if __name__ == "__main__":
    unittest.main()
