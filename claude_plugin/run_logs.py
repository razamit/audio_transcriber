"""The log files under <data dir>/logs: opening them with a start line, and reading them back.

Child output goes to these files instead of the runner's stdout, because
stdout is what Claude reads, and a transcription's stderr carries every
transcribed segment.
"""

import datetime
import os

LOGS_FOLDER = "logs"
INSTALL_LOG = "install.log"
RUN_LOG = "last-run.log"


def log_path(data_dir, log_name):
    return os.path.join(data_dir, LOGS_FOLDER, log_name)


def open_log(path, start_line):
    """Create (or overwrite) the log and write the start line.

    Returns the open binary handle, positioned after the start line, so
    whatever the child writes comes after it.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "wb")
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    handle.write(f"[{stamp}] {start_line}\n".encode("utf-8"))
    handle.flush()
    return handle


def read_log_from(path, offset):
    """The log's text from byte `offset` on, or "" when it cannot be read."""
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def last_lines(text, line_count):
    return "\n".join(text.splitlines()[-line_count:])
