"""Picking the transcription engine for this machine.

Apple Silicon Macs with mlx-whisper installed get the MLX engine; every other
machine gets faster-whisper. requirements.txt installs exactly one of the two
through environment markers, so the choice normally follows the install.

TRANSCRIBE_ENGINE=mlx|faster-whisper overrides it, which is how the
faster-whisper engine can be exercised on a Mac. An unknown value, or naming an
engine that is not installed, is an error rather than a silent fallback, for
the same reason as TRANSCRIBE_LANG: a fallback would look like a working
override.
"""

import importlib.util
import os
import platform
import sys

from platform_hints import installer_command, venv_python_command

ENGINE_ENV_VAR = "TRANSCRIBE_ENGINE"
MLX_ENGINE_KEY = "mlx"
FASTER_WHISPER_ENGINE_KEY = "faster-whisper"

# The Python module each engine needs, and the pip requirement that provides it.
ENGINE_MODULES = {MLX_ENGINE_KEY: "mlx_whisper", FASTER_WHISPER_ENGINE_KEY: "faster_whisper"}
ENGINE_REQUIREMENTS = {
    MLX_ENGINE_KEY: "mlx-whisper==0.4.3",
    FASTER_WHISPER_ENGINE_KEY: "faster-whisper==1.2.1",
}


class EngineUnavailable(Exception):
    """No usable engine, with a message saying how to get one."""


def module_importable(module_name):
    """True if the module could be imported, without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def is_apple_silicon(platform_name, machine):
    # Under Rosetta, platform.machine() reports x86_64, and MLX cannot run there.
    return platform_name == "darwin" and machine == "arm64"


def no_engine_message(platform_name):
    return (
        "No transcription engine is installed.\n"
        f"Install the requirements with:  {installer_command(platform_name)}"
    )


def missing_override_message(engine_key, platform_name):
    pip = venv_python_command(platform_name) + " -m pip"
    return (
        f"{ENGINE_ENV_VAR}={engine_key} but {ENGINE_MODULES[engine_key]} is not installed.\n"
        f"Install it with:  {pip} install {ENGINE_REQUIREMENTS[engine_key]}"
    )


def checked_override(engine_key, importable, platform_name):
    """The engine named by TRANSCRIBE_ENGINE, if it is known and installed."""
    if engine_key not in ENGINE_MODULES:
        raise EngineUnavailable(
            f"{ENGINE_ENV_VAR}={engine_key!r} is not a known engine. "
            f"Choose one of: {', '.join(ENGINE_MODULES)}"
        )
    if not importable(ENGINE_MODULES[engine_key]):
        raise EngineUnavailable(missing_override_message(engine_key, platform_name))
    return engine_key


def choose_engine_key(environ=None, platform_name=None, machine=None,
                      importable=module_importable):
    """The key of the engine to use. Raises EngineUnavailable if there is none."""
    environ = os.environ if environ is None else environ
    platform_name = sys.platform if platform_name is None else platform_name
    machine = platform.machine() if machine is None else machine

    override = (environ.get(ENGINE_ENV_VAR) or "").strip().lower()
    if override:
        return checked_override(override, importable, platform_name)
    if is_apple_silicon(platform_name, machine) and importable(ENGINE_MODULES[MLX_ENGINE_KEY]):
        return MLX_ENGINE_KEY
    if importable(ENGINE_MODULES[FASTER_WHISPER_ENGINE_KEY]):
        return FASTER_WHISPER_ENGINE_KEY
    raise EngineUnavailable(no_engine_message(platform_name))


def build_engine(engine_key):
    """Construct the engine. Each engine module is imported only when chosen."""
    if engine_key == MLX_ENGINE_KEY:
        from engines.mlx_engine import MlxWhisperEngine

        return MlxWhisperEngine()
    from engines.faster_whisper_engine import FasterWhisperEngine

    return FasterWhisperEngine()


def select_engine(**machine_facts):
    """The engine for this machine, or exit with how to install one."""
    try:
        return build_engine(choose_engine_key(**machine_facts))
    except EngineUnavailable as exc:
        sys.exit(str(exc))


def engine_key_or_none(**machine_facts):
    """The engine that would run, or None, for reporting without exiting."""
    try:
        return choose_engine_key(**machine_facts)
    except EngineUnavailable:
        return None
