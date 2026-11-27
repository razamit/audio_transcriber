"""`plugin_runner.py status`: what the skill needs to know before it installs or runs.

Reports facts only. It never installs, creates or changes anything, so the
skill can call it at the start of every /transcribe.
"""

import json
import os
import subprocess

from install import python_version_problem
from platform_hints import ffmpeg_install_hint
from profiles import PROFILE_ENV_VAR, PROFILES

from claude_plugin.host_system import NO_BYTECODE_FLAG
from claude_plugin.requirements_snapshot import requirements_changed
from claude_plugin.venv_location import LOCATION_DATA, resolve_venv

CAPABILITIES_TIMEOUT_SECONDS = 90  # importing MLX or torch can take a while on a cold start


def parse_last_json_line(text):
    """The last non-empty line of `text` as a JSON object, or None."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        parsed = json.loads(lines[-1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def read_capabilities(python, repo_folder, host):
    """`transcribe.py --capabilities` from the given venv, or None if it fails."""
    command = [python, NO_BYTECODE_FLAG, os.path.join(repo_folder, "transcribe.py"),
               "--capabilities"]
    try:
        completed = host.run_process(
            command, cwd=repo_folder, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=CAPABILITIES_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return parse_last_json_line(completed.stdout)


def default_language(environ):
    """TRANSCRIBE_LANG when it names a known language, else None."""
    key = (environ.get(PROFILE_ENV_VAR) or "").strip().lower()
    return key if key in PROFILES else None


def nvidia_gpu_detected(host):
    """nvidia-smi on PATH, on the platforms where install.py --gpu applies."""
    supported = host.platform_name == "win32" or host.platform_name.startswith("linux")
    return bool(supported and host.which("nvidia-smi"))


def build_status(data_dir, repo_folder, host):
    python, location = resolve_venv(repo_folder, data_dir, host.platform_name)
    has_ffmpeg = host.which("ffmpeg") is not None
    return {
        "installed": python is not None,
        "venv_python": python,
        "venv_location": location,
        "capabilities": read_capabilities(python, repo_folder, host) if python else None,
        "ffmpeg": has_ffmpeg,
        "ffmpeg_hint": None if has_ffmpeg else ffmpeg_install_hint(host.platform_name),
        "default_language": default_language(host.environ),
        "nvidia_gpu_detected": nvidia_gpu_detected(host),
        "requirements_changed": (location == LOCATION_DATA
                                 and requirements_changed(data_dir, repo_folder)),
        "platform": host.platform_name,
        "python_ok": python_version_problem(host.version_info) is None,
    }
