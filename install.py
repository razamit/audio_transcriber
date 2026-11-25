#!/usr/bin/env python3
"""Create the local environment for transcribe.py, on macOS, Linux or Windows.

    python3 install.py [--with-diarization] [--gpu]     (macOS, Linux)
    py install.py [--with-diarization] [--gpu]          (Windows)

setup.sh runs this with the same arguments, so `./setup.sh` keeps working.

It checks Python and ffmpeg, creates .venv next to this file, and installs
requirements.txt (whose markers pick mlx-whisper or faster-whisper for the
machine). --with-diarization adds pyannote for speaker labels. --gpu adds the
NVIDIA libraries faster-whisper needs to use a CUDA GPU, and with
--with-diarization a CUDA build of PyTorch, so diarization uses the GPU too.
Any torch already installed is removed first, because pip would otherwise
call a CPU build "already satisfied" and keep it.

On a Python newer than faster-whisper's dependencies publish packages for,
it warns before installing (see new_python_warning) but carries on.

Standard library only, because it runs with the system Python before any
virtual environment exists. The pip commands are built by one function and run
by another, so the tests can check them without installing anything.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import sysconfig

from platform_hints import ffmpeg_install_hint, installer_command, venv_python_command

MINIMUM_PYTHON = (3, 10)
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
VENV_FOLDER = ".venv"

# cuBLAS for CUDA 12 and cuDNN 9: what CTranslate2 4.x loads to run on the GPU.
GPU_LIBRARY_PACKAGES = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*")
# A CUDA 12 build of PyTorch, matching the CUDA generation CTranslate2 4.x is
# built for. pytorch.org's install page now lists only CUDA 13 builds; the
# cu126 index still serves current releases for Windows and Linux.
TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu126"
TORCH_PACKAGES = ("torch", "torchaudio")

# faster-whisper depends on onnxruntime, which had published no package for
# Python 3.15 when this was written. Packages will follow, so it is a warning.
FIRST_PYTHON_WITHOUT_ONNXRUNTIME = (3, 15)
OLDER_PYTHON_COMMAND = {"win32": "py -3.13 install.py"}
OLDER_PYTHON_COMMAND_ELSEWHERE = "python3.13 install.py"


def parse_options(argv):
    parser = argparse.ArgumentParser(
        prog="install.py", description="Create .venv and install the transcriber."
    )
    parser.add_argument("--with-diarization", action="store_true",
                        help="also install speaker labelling (pyannote, about 2GB)")
    parser.add_argument("--gpu", action="store_true",
                        help="NVIDIA GPU support on Windows or Linux")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Checks: each returns a message to exit with, or None when all is well
# --------------------------------------------------------------------------

def python_version_problem(version_info):
    if tuple(version_info[:2]) >= MINIMUM_PYTHON:
        return None
    wanted = ".".join(str(part) for part in MINIMUM_PYTHON)
    found = ".".join(str(part) for part in version_info[:3])
    return f"Python {wanted} or newer is required; this is Python {found}."


def ffmpeg_problem(which, platform_name):
    if which("ffmpeg") is not None:
        return None
    return f"ffmpeg is required to decode audio. Install it with:  {ffmpeg_install_hint(platform_name)}"


def platform_problem(build_platform):
    """faster-whisper's CTranslate2 publishes no wheel for Windows on ARM."""
    if build_platform != "win-arm64":
        return None
    return (
        "This is an ARM64 build of Python. CTranslate2, which faster-whisper needs, "
        "publishes no Windows on ARM package, so the install would fail. An x64 "
        "build of Python may work under emulation; that has not been tested."
    )


def gpu_problem(gpu, platform_name):
    if not gpu or platform_name == "win32" or platform_name.startswith("linux"):
        return None
    return ("--gpu is for NVIDIA GPUs on Windows and Linux. On an Apple Silicon Mac, "
            "transcription already uses the GPU through MLX.")


def installs_faster_whisper(platform_name, machine):
    """Mirrors requirements.txt: every machine but an Apple Silicon Mac gets it."""
    return not (platform_name == "darwin" and machine == "arm64")


def new_python_warning(version_info, platform_name, machine):
    """A warning when faster-whisper may have no packages for this Python yet."""
    if not installs_faster_whisper(platform_name, machine):
        return None
    if tuple(version_info[:2]) < FIRST_PYTHON_WITHOUT_ONNXRUNTIME:
        return None
    found = ".".join(str(part) for part in version_info[:2])
    older = OLDER_PYTHON_COMMAND.get(platform_name, OLDER_PYTHON_COMMAND_ELSEWHERE)
    launcher_note = ""
    if platform_name == "win32":
        launcher_note = " (the py launcher picks the newest Python installed)"
    return (
        f"Warning: this is Python {found}{launcher_note}. faster-whisper needs onnxruntime, "
        "which may not publish prebuilt packages for it yet, so the install may fail. "
        f"If it does, run the installer with an older Python, such as:  {older}"
    )


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def venv_python_path(project_folder, platform_name):
    if platform_name == "win32":
        return os.path.join(project_folder, VENV_FOLDER, "Scripts", "python.exe")
    return os.path.join(project_folder, VENV_FOLDER, "bin", "python")


def pip_install(python, *arguments):
    return [python, "-m", "pip", "install", *arguments]


def pip_uninstall(python, *packages):
    return [python, "-m", "pip", "uninstall", "-y", *packages]


def build_install_commands(python, with_diarization, gpu):
    """Every pip command, in order. The CUDA torch goes in before pyannote, so
    pyannote's own torch requirement is already met and pip keeps the CUDA build.
    Any torch already there is uninstalled first: a CPU build would count as
    "already satisfied" and stay. pip uninstall skips packages that are absent."""
    commands = [
        pip_install(python, "--upgrade", "pip", "--quiet"),
        pip_install(python, "-r", "requirements.txt"),
    ]
    if gpu:
        commands.append(pip_install(python, *GPU_LIBRARY_PACKAGES))
    if with_diarization:
        if gpu:
            commands.append(pip_uninstall(python, *TORCH_PACKAGES))
            commands.append(pip_install(python, *TORCH_PACKAGES,
                                        "--index-url", TORCH_CUDA_INDEX_URL))
        commands.append(pip_install(python, "-r", "requirements-diarize.txt"))
    return commands


def build_venv_command(system_python):
    return [system_python, "-m", "venv", VENV_FOLDER]


def run_commands(commands, runner=subprocess.check_call, folder=PROJECT_FOLDER):
    for command in commands:
        runner(command, cwd=folder)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def first_problem(options, platform_name, which=shutil.which):
    checks = (
        python_version_problem(sys.version_info),
        platform_problem(sysconfig.get_platform()) if platform_name == "win32" else None,
        gpu_problem(options.gpu, platform_name),
        ffmpeg_problem(which, platform_name),
    )
    return next((problem for problem in checks if problem), None)


def closing_message(options, platform_name):
    run_line = f"    {venv_python_command(platform_name)} transcribe.py"
    lines = ["", "Done. Run the transcriber with:", run_line]
    if options.with_diarization:
        lines += ["", "Diarization installed. No account or token is needed; the model "
                      "downloads on first run."]
    else:
        lines += ["", "For speaker labels (who said what), re-run as:",
                  f"    {installer_command(platform_name)} --with-diarization"]
    return "\n".join(lines)


def main(argv=None, platform_name=None):
    platform_name = sys.platform if platform_name is None else platform_name
    options = parse_options(sys.argv[1:] if argv is None else argv)
    problem = first_problem(options, platform_name)
    if problem:
        sys.exit(problem)

    warning = new_python_warning(sys.version_info, platform_name, platform.machine())
    if warning:
        print(warning)

    python = venv_python_path(PROJECT_FOLDER, platform_name)
    try:
        run_commands([build_venv_command(sys.executable)])
        run_commands(build_install_commands(python, options.with_diarization, options.gpu))
    except subprocess.CalledProcessError as exc:
        sys.exit(f"\nInstall step failed (exit code {exc.returncode}): {' '.join(exc.cmd)}")
    print(closing_message(options, platform_name))


if __name__ == "__main__":
    main()
