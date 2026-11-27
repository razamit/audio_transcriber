"""The command lines that open a speaker-naming window, one pure function per OS.

Nothing here starts a process or touches a file, so every command can be
checked in a test on any machine. claude_plugin/naming_window.py decides which
one to use and runs it.

- macOS: a small `.command` shell script, opened in Terminal with
  `/usr/bin/open -a Terminal`. Opening a file needs no Automation permission,
  unlike `osascript` telling Terminal to `do script`, which macOS gates behind
  a one-time "control Terminal" prompt for whichever app hosts Claude Code.
  The paths go inside the script, quoted for /bin/sh, so Terminal never has to
  parse a path with spaces, quotes or Hebrew letters.
- Windows: the naming program itself, started with CREATE_NEW_CONSOLE.
- Linux and other Unix desktops: the first terminal emulator found, each with
  its own "run this program" flag (Debian's x-terminal-emulator and xterm and
  konsole take `-e`, xfce4-terminal `-x`, gnome-terminal `--`). Every one of
  them passes the following arguments to the program without a shell.
"""

import os
import shlex

from claude_plugin.host_system import NO_BYTECODE_FLAG

NAMING_SCRIPT = "name_speakers.py"

MACOS_OPEN = "/usr/bin/open"
MACOS_TERMINAL_APP = "Terminal"

# subprocess.CREATE_NEW_CONSOLE exists only on Windows; this is its documented value.
WINDOWS_CREATE_NEW_CONSOLE = 0x00000010

# Tried in this order. x-terminal-emulator is the user's chosen default on
# Debian and Ubuntu; xterm comes last because it is the plainest.
LINUX_EMULATORS = (
    ("x-terminal-emulator", "-e"),
    ("gnome-terminal", "--"),
    ("konsole", "-e"),
    ("xfce4-terminal", "-x"),
    ("xterm", "-e"),
)


def naming_command(python, repo_folder, transcript_path):
    """The venv's Python on name_speakers.py with the transcript as its argument."""
    return [python, NO_BYTECODE_FLAG, os.path.join(repo_folder, NAMING_SCRIPT), transcript_path]


def macos_command_file_text(command):
    """A /bin/sh script that deletes itself, then replaces itself with `command`.

    Deleting first keeps the plugin data folder clean; the shell already has
    the file open, so it reads on to the end regardless.
    """
    quoted = " ".join(shlex.quote(part) for part in command)
    return (
        "#!/bin/sh\n"
        "# Written by the audio-transcriber plugin to name the speakers of one transcript.\n"
        'rm -f -- "$0"\n'
        f"exec {quoted}\n"
    )


def macos_open_command(command_file):
    """Open the script in Terminal by name.

    Naming the app guards against `.command` files being set to open in an
    editor, which would show the script instead of running it.
    """
    return [MACOS_OPEN, "-a", MACOS_TERMINAL_APP, command_file]


def windows_launch(command):
    """(argv, creationflags) for a new console running `command`."""
    return list(command), WINDOWS_CREATE_NEW_CONSOLE


def linux_launch_candidates(command, which):
    """[(emulator name, argv)] for every known emulator `which` can find, in order."""
    candidates = []
    for name, run_flag in LINUX_EMULATORS:
        executable = which(name)
        if executable:
            candidates.append((name, [executable, run_flag, *command]))
    return candidates


def powershell_quote(text):
    """A PowerShell literal string: single quotes, with each ' doubled."""
    return "'" + text.replace("'", "''") + "'"


def manual_command(command, platform_name):
    """`command` as one line the user can paste into their own terminal.

    PowerShell on Windows, since that is what a Windows terminal opens to; a
    quoted program path needs the call operator `&` there.
    """
    if platform_name == "win32":
        return "& " + " ".join(powershell_quote(part) for part in command)
    return " ".join(shlex.quote(part) for part in command)
