"""`plugin_runner.py install`: run install.py into the right venv, with pip's output in a log.

pip prints hundreds of lines, none of them private, but none useful to Claude
either, so they go to <data dir>/logs/install.log. The report carries the last
lines only when the install failed.
"""

import os
import subprocess

from claude_plugin.host_system import NO_BYTECODE_FLAG, child_environment
from claude_plugin.requirements_snapshot import save_snapshot
from claude_plugin.run_logs import INSTALL_LOG, last_lines, log_path, open_log, read_log_from
from claude_plugin.venv_location import LOCATION_DATA, install_target

FAILURE_TAIL_LINES = 30


def build_install_command(system_python, repo_folder, with_diarization, gpu, venv_dir):
    """install.py with the matching flags; --venv-dir only when installing outside the repo."""
    command = [system_python, NO_BYTECODE_FLAG, os.path.join(repo_folder, "install.py")]
    if with_diarization:
        command.append("--with-diarization")
    if gpu:
        command.append("--gpu")
    if venv_dir:
        command += ["--venv-dir", venv_dir]
    return command


def run_install(data_dir, with_diarization, gpu, repo_folder, host):
    """Run the install. Returns (report, exit_code)."""
    venv_dir, location = install_target(repo_folder, data_dir, host.platform_name)
    command = build_install_command(host.system_python, repo_folder, with_diarization, gpu, venv_dir)
    path = log_path(data_dir, INSTALL_LOG)
    log = open_log(path, "install: " + " ".join(command))
    try:
        completed = host.run_process(
            command, cwd=repo_folder, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, env=child_environment(host.environ),
        )
        exit_code = completed.returncode
    except OSError as error:
        log.write(f"Could not start install.py: {error}\n".encode("utf-8"))
        exit_code = 1
    finally:
        log.close()

    report = {"ok": exit_code == 0, "exit_code": exit_code, "log": path,
              "venv_location": location}
    if exit_code == 0 and location == LOCATION_DATA:
        save_snapshot(data_dir, repo_folder)
    if exit_code != 0:
        report["log_tail"] = last_lines(read_log_from(path, 0), FAILURE_TAIL_LINES)
    return report, exit_code
