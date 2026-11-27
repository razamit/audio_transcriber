"""`plugin_runner.py run`: one transcription, with transcript text kept out of Claude's context.

This module exists for one rule. Whisper prints every transcribed segment to
stderr, and whatever the runner prints reaches Claude and so Anthropic. The
owner decided transcript text goes there only when they ask for it. So the
child's stderr is written to <data dir>/logs/last-run.log and never echoed,
and the report says only what is known to hold no transcript:

- exit 0: the parsed JSON result line (paths, language, counts, time).
- exit 2, 3, 4: the child's stderr. cli.py raises these while validating,
  before any model runs, so nothing has been transcribed yet.
- exit 1 or anything else: the traceback after cli.py's `Failed:` marker, or
  else only the log path. When in doubt, the path and not the content.
"""

import os
import subprocess

from claude_plugin.host_system import NO_BYTECODE_FLAG, child_environment
from claude_plugin.run_logs import RUN_LOG, log_path, open_log, read_log_from
from claude_plugin.status_report import parse_last_json_line

JSON_FLAG = "--json"
VALIDATION_EXIT_CODES = (2, 3, 4)  # cli.py: bad input, no diarization extras, no ffmpeg
FAILURE_MARKER = "Failed:"
FAILURE_SECTION_MAX_LINES = 60


def with_json_flag(transcribe_flags):
    flags = list(transcribe_flags)
    return flags if JSON_FLAG in flags else flags + [JSON_FLAG]


def build_run_command(python, repo_folder, transcribe_flags):
    return [python, NO_BYTECODE_FLAG, os.path.join(repo_folder, "transcribe.py"),
            *with_json_flag(transcribe_flags)]


def failure_section(log_text):
    """From the last line that is exactly `Failed:` to the end, or None without one."""
    lines = log_text.splitlines()
    marker_indexes = [index for index, line in enumerate(lines) if line.strip() == FAILURE_MARKER]
    if not marker_indexes:
        return None
    section = lines[marker_indexes[-1]:]
    if len(section) > FAILURE_SECTION_MAX_LINES:
        section = [FAILURE_MARKER, "..."] + section[-FAILURE_SECTION_MAX_LINES:]
    return "\n".join(section).strip()


def failure_message(exit_code, child_stderr, log):
    """What the report may say about a failed run. See the module docstring."""
    if exit_code in VALIDATION_EXIT_CODES:
        return child_stderr.strip()
    section = failure_section(child_stderr)
    if section:
        return f"{section}\n\nFull log: {log}"
    return f"Transcription failed. The log is at: {log}"


def build_report(exit_code, stdout_text, child_stderr, log):
    """(report, exit_code) from what the child left behind."""
    if exit_code != 0:
        return {"ok": False, "exit_code": exit_code, "result": None, "log": log,
                "message": failure_message(exit_code, child_stderr, log)}, exit_code
    result = parse_last_json_line(stdout_text)
    if result is None or "output" not in result:
        # Never echo stdout itself: if the one-line contract broke, it could hold anything.
        message = f"transcribe.py exited 0 but printed no JSON result. The log is at: {log}"
        return {"ok": False, "exit_code": 1, "result": None, "log": log,
                "message": message}, 1
    return {"ok": True, "exit_code": 0, "result": result, "log": log, "message": None}, 0


def run_transcription(data_dir, python, transcribe_flags, repo_folder, host):
    """Run transcribe.py from the given venv. Returns (report, exit_code)."""
    command = build_run_command(python, repo_folder, transcribe_flags)
    path = log_path(data_dir, RUN_LOG)
    log = open_log(path, "run: " + " ".join(command))
    child_output_starts_at = log.tell()
    try:
        completed = host.run_process(
            command, cwd=repo_folder, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=log, env=child_environment(host.environ),
        )
        exit_code, stdout_bytes = completed.returncode, completed.stdout or b""
    except OSError as error:
        log.write(f"Could not start transcribe.py: {error}\n".encode("utf-8"))
        exit_code, stdout_bytes = 1, b""
    finally:
        log.close()

    child_stderr = read_log_from(path, child_output_starts_at)
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    return build_report(exit_code, stdout_text, child_stderr, path)
