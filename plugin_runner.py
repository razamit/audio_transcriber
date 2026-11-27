#!/usr/bin/env python3
"""The one script the Claude Code /transcribe skill calls.

    python3 plugin_runner.py status  --data-dir DIR
    python3 plugin_runner.py install --data-dir DIR [--with-diarization] [--gpu]
    python3 plugin_runner.py run     --data-dir DIR -- <transcribe.py flags>
    python3 plugin_runner.py name    --data-dir DIR -- --file <transcript>

(`py` in place of `python3` on Windows.) Each prints one JSON object on
stdout and nothing else, so the skill reads facts instead of parsing prose.
DIR is the plugin's data folder, `${CLAUDE_PLUGIN_DATA}`, which survives
plugin updates: it holds the venv for a plugin installed from GitHub, the
requirements snapshot and the logs.

Path rules, platform differences and the privacy rule live here, in tested
code, rather than in the skill's prose. The privacy rule: a transcription's
stderr carries the transcript, so `run` writes it to a log and never prints
it (see claude_plugin/transcription_run.py). `name` keeps the same rule for
speaker naming, which has to show transcript lines: it opens a terminal window
on the user's screen to do it, and reports only whether the window opened
(see claude_plugin/naming_window.py).

Standard library only, because it runs with the system Python. `--` before
the transcribe.py flags is optional, since some shells drop it. The JSON is
ASCII, with Hebrew paths escaped, so a Windows code page cannot break it.
"""

import argparse
import json
import os
import sys

sys.dont_write_bytecode = True  # nothing may be written into the plugin folder

from claude_plugin.host_system import REPO_FOLDER, real_host_system
from claude_plugin.install_step import run_install
from claude_plugin.naming_window import open_naming_window
from claude_plugin.status_report import build_status
from claude_plugin.transcription_run import run_transcription
from claude_plugin.venv_location import resolve_venv

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
PASSTHROUGH_SEPARATOR = "--"
PASSTHROUGH_COMMANDS = ("run", "name")  # the commands that take flags after `--`


class RunnerUsageError(Exception):
    """Bad arguments to the runner itself, reported as JSON instead of argparse's usage text."""


class JsonErrorParser(argparse.ArgumentParser):
    def error(self, message):
        raise RunnerUsageError(message)


def build_parser():
    parser = JsonErrorParser(prog="plugin_runner.py", allow_abbrev=False,
                             description="Status, install, run and name for the Claude Code plugin.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "install", "run", "name"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--data-dir", required=True,
                             help="the plugin data folder, ${CLAUDE_PLUGIN_DATA}")
        if name == "install":
            command.add_argument("--with-diarization", action="store_true")
            command.add_argument("--gpu", action="store_true")
    return parser


def parse_arguments(argv):
    """(options, passthrough_flags). Flags after `--` pass through untouched.

    Without `--`, anything `run` or `name` does not recognise is passed through instead.
    """
    if PASSTHROUGH_SEPARATOR in argv:
        split_at = argv.index(PASSTHROUGH_SEPARATOR)
        options = build_parser().parse_args(argv[:split_at])
        return options, argv[split_at + 1:]
    options, unknown = build_parser().parse_known_args(argv)
    if unknown and options.command not in PASSTHROUGH_COMMANDS:
        raise RunnerUsageError("unrecognized arguments: " + " ".join(unknown))
    return options, unknown


def checked_data_dir(raw):
    """The data folder as an absolute path. An unsubstituted ${...} is refused, not created."""
    if not raw.strip() or "${" in raw:
        raise RunnerUsageError(f"--data-dir must be the plugin data folder, got {raw!r}")
    return os.path.abspath(os.path.expanduser(raw))


def naming_transcript_path(flags):
    """The transcript given to `name` as `--file PATH`, as an absolute path."""
    parser = JsonErrorParser(prog="plugin_runner.py name", allow_abbrev=False)
    parser.add_argument("--file", required=True, help="the _speakers.txt transcript")
    transcript = parser.parse_args(flags).file
    return os.path.abspath(os.path.expanduser(transcript))


def run_command(options, passthrough_flags, host, repo_folder):
    """Dispatch one subcommand. Returns (report, exit_code)."""
    data_dir = checked_data_dir(options.data_dir)
    if options.command == "status":
        return build_status(data_dir, repo_folder, host), EXIT_OK
    if options.command == "install":
        return run_install(data_dir, options.with_diarization, options.gpu, repo_folder, host)
    python, _ = resolve_venv(repo_folder, data_dir, host.platform_name)
    if options.command == "name":
        transcript = naming_transcript_path(passthrough_flags)
        return open_naming_window(data_dir, python, transcript, repo_folder, host)
    if python is None:
        return {"ok": False, "exit_code": EXIT_ERROR, "error": "not_installed",
                "message": "The transcriber is not installed yet. Run install first."}, EXIT_ERROR
    return run_transcription(data_dir, python, passthrough_flags, repo_folder, host)


def process_exit_code(exit_code):
    """A code the OS can carry: a signal (negative) or out-of-range code becomes 1."""
    return exit_code if 0 <= exit_code <= 255 else EXIT_ERROR


def main(argv=None, host=None, repo_folder=REPO_FOLDER, stdout=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    host = real_host_system() if host is None else host
    stdout = sys.stdout if stdout is None else stdout
    try:
        options, passthrough_flags = parse_arguments(argv)
        report, exit_code = run_command(options, passthrough_flags, host, repo_folder)
    except RunnerUsageError as error:
        report, exit_code = {"ok": False, "exit_code": EXIT_USAGE, "error": "usage",
                             "message": str(error)}, EXIT_USAGE
    stdout.write(json.dumps(report) + "\n")  # ASCII escapes: safe on any console code page
    stdout.flush()
    return process_exit_code(exit_code)


if __name__ == "__main__":
    sys.exit(main())
