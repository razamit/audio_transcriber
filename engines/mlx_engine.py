"""mlx-whisper: the engine for Apple Silicon Macs.

MLX runs Whisper on the Mac's GPU through Metal. It has no CPU or CUDA
fallback, which is why the faster-whisper engine exists for everything else.
mlx_whisper is imported only when a transcription starts.
"""

import importlib

from engines.decoding import decoding_options
from profiles import RUNTIME_MLX
from terminal_bidi import reordering_stdout

ENGINE_KEY = "mlx"


def import_mlx_whisper():
    return importlib.import_module("mlx_whisper")


class MlxWhisperEngine:
    """Transcribes with mlx_whisper.transcribe, printing its own progress lines."""

    key = ENGINE_KEY
    runtime_key = RUNTIME_MLX

    def __init__(self, load_mlx_whisper=import_mlx_whisper,
                 stdout_wrapper=reordering_stdout):
        self._load_mlx_whisper = load_mlx_whisper
        self._stdout_wrapper = stdout_wrapper

    def transcribe(self, audio_path, profile):
        """mlx-whisper's own result dict, which already has text and segments."""
        mlx_whisper = self._load_mlx_whisper()
        source = profile.source_for(self.runtime_key)
        # verbose=True prints each segment as it is decoded. The wrapper
        # reorders those lines for terminals that cannot show Hebrew.
        with self._stdout_wrapper():
            return mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=source.repo,
                verbose=True,
                **decoding_options(profile.language),
            )
