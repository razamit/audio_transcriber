"""Everything the runner asks of the machine, in one injectable object.

Status, install and run take a HostSystem instead of calling subprocess,
shutil.which or os.environ directly, so tests hand in a fake one and nothing
is installed or transcribed. Files are still read and written for real; tests
point the runner at a scratch folder.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Mapping, Tuple

REPO_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Every Python the runner starts gets -B, so no __pycache__ is written into the
# plugin folder, which Claude Code replaces on update. pip still precompiles
# the packages it installs into the venv, so imports stay fast.
NO_BYTECODE_FLAG = "-B"


@dataclass(frozen=True)
class HostSystem:
    """The probes and the process launcher, plus facts about this machine."""

    platform_name: str
    version_info: Tuple[int, ...]
    system_python: str
    environ: Mapping[str, str]
    which: Callable
    run_process: Callable  # subprocess.run-compatible


def real_host_system():
    return HostSystem(
        platform_name=sys.platform,
        version_info=tuple(sys.version_info[:3]),
        system_python=sys.executable,
        environ=os.environ,
        which=shutil.which,
        run_process=subprocess.run,
    )


def child_environment(environ):
    """The child's environment: ours, with unbuffered output so a log can be followed live."""
    environment = dict(environ)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment
