"""`plugin_runner.py name`: open a terminal window that asks the user to name the speakers.

Naming means showing each speaker's longest lines, and transcript text must
not reach Claude. So the runner never runs the prompt itself: it opens a new
window on the user's own screen, running name_speakers.py on the transcript,
and returns at once. The report says only whether the window opened, how, and
a command the user can paste if it did not. It never holds transcript text;
the transcript is opened only to read its first line, the
"N speaker(s) detected:" header, which proves it is a labelled transcript.

A window that cannot be opened is not an error the skill has to handle
specially: the report has `"opened": false`, a short reason, and the manual
command, and the exit code is still 0.
"""

import glob
import os
import re
import subprocess
import tempfile
import time

from terminal_bidi import NON_REORDERING_MARKER_VARIABLES

from claude_plugin.window_commands import (
    linux_launch_candidates,
    macos_command_file_text,
    macos_open_command,
    manual_command,
    naming_command,
    windows_launch,
)

EXIT_OK = 0

# Set by sshd. A window would open on the remote machine's screen, if any.
REMOTE_SESSION_VARIABLES = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")
DISPLAY_VARIABLES = ("DISPLAY", "WAYLAND_DISPLAY")

# The new window is a different terminal from the one Claude Code runs in, so
# these must not tell terminal_bidi it is, say, VS Code's.
INHERITED_TERMINAL_VARIABLES = ("TERM_PROGRAM", "TERM_PROGRAM_VERSION") + tuple(
    NON_REORDERING_MARKER_VARIABLES)

SPEAKER_HEADER_PATTERN = re.compile(r"^\d+ speaker\(s\) detected:$")

COMMAND_FILE_FOLDER = "naming"
COMMAND_FILE_PREFIX = "name-speakers-"
COMMAND_FILE_SUFFIX = ".command"
STALE_COMMAND_FILE_SECONDS = 3600
OPEN_TIMEOUT_SECONDS = 30
# A launcher that fails usually exits at once; one still running, or exited 0
# after handing the window to a server (gnome-terminal), counts as opened.
LAUNCH_SETTLE_SECONDS = 1.0

METHOD_MACOS = "macos_terminal_command_file"
METHOD_WINDOWS = "windows_new_console"
METHOD_LINUX_PREFIX = "linux_"


class WindowNotOpened(Exception):
    """Why no window was opened, as a short ASCII reason code."""


def open_naming_window(data_dir, python, transcript_path, repo_folder, host):
    """(report, exit_code). Opens the window or explains why not; never raises for that."""
    manual = None
    try:
        if python is None:
            raise WindowNotOpened("not_installed")
        command = naming_command(python, repo_folder, transcript_path)
        manual = manual_command(command, host.platform_name)
        check_speaker_transcript(transcript_path)
        check_graphical_session(host)
        method = launch_window(command, data_dir, host)
    except WindowNotOpened as refusal:
        return {"ok": False, "opened": False, "method": None, "reason": str(refusal),
                "manual_command": manual}, EXIT_OK
    return {"ok": True, "opened": True, "method": method, "reason": None,
            "manual_command": manual}, EXIT_OK


def check_speaker_transcript(path):
    """Refuse a missing file or one without the labelled transcript's header line."""
    if not os.path.isfile(path):
        raise WindowNotOpened("file_not_found")
    try:
        with open(path, encoding="utf-8") as handle:
            first_line = handle.readline().strip()
    except (OSError, UnicodeDecodeError):
        raise WindowNotOpened("file_unreadable") from None
    if not SPEAKER_HEADER_PATTERN.match(first_line):
        raise WindowNotOpened("not_a_speaker_transcript")


def check_graphical_session(host):
    """Refuse a remote session, and a Unix desktop with no display to open a window on."""
    if any(host.environ.get(name) for name in REMOTE_SESSION_VARIABLES):
        raise WindowNotOpened("remote_session")
    if host.platform_name in ("darwin", "win32"):
        return
    if not any(host.environ.get(name) for name in DISPLAY_VARIABLES):
        raise WindowNotOpened("no_display")


def launch_window(command, data_dir, host):
    """Open the window for this platform. Returns the method name."""
    if host.platform_name == "darwin":
        return launch_on_macos(command, data_dir, host)
    if host.platform_name == "win32":
        return launch_on_windows(command, host)
    return launch_on_linux(command, host)


def launch_on_macos(command, data_dir, host):
    command_file = write_command_file(data_dir, command)
    try:
        completed = host.run_process(
            macos_open_command(command_file), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=OPEN_TIMEOUT_SECONDS)
        opened = completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        opened = False
    if not opened:
        remove_quietly(command_file)
        raise WindowNotOpened("launch_failed")
    return METHOD_MACOS


def write_command_file(data_dir, command):
    """A fresh executable script in <data dir>/naming, after clearing stale ones."""
    folder = os.path.join(data_dir, COMMAND_FILE_FOLDER)
    try:
        os.makedirs(folder, exist_ok=True)
        remove_stale_command_files(folder)
        handle, path = tempfile.mkstemp(prefix=COMMAND_FILE_PREFIX,
                                        suffix=COMMAND_FILE_SUFFIX, dir=folder)
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as script:
            script.write(macos_command_file_text(command))
        os.chmod(path, 0o700)
    except OSError:
        raise WindowNotOpened("launch_failed") from None
    return path


def remove_stale_command_files(folder, now=None):
    """Scripts left behind when Terminal never ran them (each deletes itself when it runs)."""
    now = time.time() if now is None else now
    pattern = os.path.join(folder, COMMAND_FILE_PREFIX + "*" + COMMAND_FILE_SUFFIX)
    for path in glob.glob(pattern):
        try:
            if now - os.path.getmtime(path) > STALE_COMMAND_FILE_SECONDS:
                os.remove(path)
        except OSError:
            pass


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def launch_on_windows(command, host):
    argv, creation_flags = windows_launch(command)
    options = {"creationflags": creation_flags, "env": window_environment(host.environ)}
    if not start_and_confirm(host, argv, options):
        raise WindowNotOpened("launch_failed")
    return METHOD_WINDOWS


def launch_on_linux(command, host):
    candidates = linux_launch_candidates(command, host.which)
    if not candidates:
        raise WindowNotOpened("no_terminal_emulator")
    options = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
               "stderr": subprocess.DEVNULL, "start_new_session": True,
               "env": window_environment(host.environ)}
    for name, argv in candidates:
        if start_and_confirm(host, argv, options):
            return METHOD_LINUX_PREFIX + name
    raise WindowNotOpened("launch_failed")


def window_environment(environ):
    """Ours, minus the variables that name the terminal Claude Code runs in."""
    return {name: value for name, value in environ.items()
            if name not in INHERITED_TERMINAL_VARIABLES}


def start_and_confirm(host, argv, options):
    """Start `argv` without waiting for it. False if it cannot start or fails at once."""
    if host.start_process is None:
        return False
    try:
        process = host.start_process(argv, **options)
    except (OSError, ValueError):
        return False
    try:
        exit_code = process.wait(timeout=LAUNCH_SETTLE_SECONDS)
    except subprocess.TimeoutExpired:
        return True
    return exit_code == 0
