#!/usr/bin/env python3
"""Install hints that differ by operating system.

One place for them, because the same advice is printed by the interactive
mode, the flag mode and the installer, and three copies drift apart.

Standard library only: install.py imports this with the system Python, before
any virtual environment exists.
"""

import sys

FFMPEG_INSTALL_HINTS = {
    "darwin": "brew install ffmpeg",
    # winget changes PATH for new terminals only, so the open one still fails.
    "win32": "winget install ffmpeg  (then open a new terminal)",
}
LINUX_FFMPEG_INSTALL_HINT = (
    "sudo apt install ffmpeg  (or your distribution's package manager, "
    "such as: sudo dnf install ffmpeg)"
)


def ffmpeg_install_hint(platform_name=None):
    """The command that installs ffmpeg on this (or the given) platform."""
    platform_name = sys.platform if platform_name is None else platform_name
    return FFMPEG_INSTALL_HINTS.get(platform_name, LINUX_FFMPEG_INSTALL_HINT)


def ffmpeg_missing_message(platform_name=None):
    """The full message for a missing ffmpeg, with the platform's install hint."""
    return (
        "ffmpeg is required to decode audio but was not found on PATH.\n"
        f"Install it with:  {ffmpeg_install_hint(platform_name)}"
    )


def installer_command(platform_name=None):
    """How this project's installer is started on this (or the given) platform."""
    platform_name = sys.platform if platform_name is None else platform_name
    return "py install.py" if platform_name == "win32" else "./setup.sh"


def venv_script_command(script_name, platform_name=None):
    """How to run a tool from the project's virtual environment, from the project folder."""
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name == "win32":
        return ".venv\\Scripts\\" + script_name
    return ".venv/bin/" + script_name


def venv_python_command(platform_name=None):
    """How to run the project's virtual-environment Python from the project folder."""
    return venv_script_command("python", platform_name)


def set_environment_variable_command(name, value, platform_name=None):
    """How to set an environment variable for the open terminal session only.

    Windows Terminal opens PowerShell by default, so that form comes first,
    with the Command Prompt form alongside it.
    """
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name == "win32":
        return f'$env:{name} = "{value}"    (Command Prompt: set {name}={value})'
    return f"export {name}={value}"
