"""faster-whisper (CTranslate2): the engine for Windows, Linux and Intel Macs.

Runs the same Whisper models as the MLX engine, converted to CTranslate2
format, on an NVIDIA GPU when one works and on the CPU otherwise (see
device_plan.py). Decoding settings are shared with the MLX engine through
decoding.py, plus a beam size of 1 set here so both decode greedily. Audio is
decoded by the ffmpeg CLI, as mlx-whisper does it, rather than by faster-whisper's
PyAV loader (see audio_decoding.py).

If the GPU fails while loading the model or transcribing (running out of
memory on a small card is the usual cause, more so when speaker labelling
has already used the card), the file is transcribed again on the CPU, with
the profile's CPU model, and the engine stays on the CPU from then on.

faster_whisper is imported only when a transcription starts, and the device is
chosen the first time anyone asks for the runtime key, so constructing this
engine costs nothing.
"""

import importlib
import os

from audio_decoding import decode_for_whisper
from engines.decoding import decoding_options
from engines.device_plan import cpu_plan, detect_device_plan
from engines.segment_progress import format_segment_line
from terminal_bidi import for_display

ENGINE_KEY = "faster-whisper"

# faster-whisper runs a beam search of 5 by default; mlx-whisper decodes
# greedily and has no beam search to turn on. A beam of 1 is greedy decoding,
# so both engines decode the same way and CPU runs do not pay five times over
# for a search the MLX engine never does. It is not in decoding.py because
# mlx_whisper.transcribe, as called there, does not accept beam_size.
GREEDY_BEAM_SIZE = 1


def import_faster_whisper():
    return importlib.import_module("faster_whisper")


def print_progress_line(line):
    """Print one segment line as it arrives, like mlx-whisper's verbose mode.

    Passed through for_display so Hebrew reads correctly on terminals that
    cannot show right-to-left text (see terminal_bidi.py).
    """
    print(for_display(line), flush=True)


def cpu_fallback_plan():
    return cpu_plan(os.cpu_count())


def gpu_failure_message(error, cpu_repo):
    """One line saying the GPU failed, why, and what happens instead."""
    reason = (str(error).strip().splitlines() or [type(error).__name__])[0]
    return (f"  The GPU could not run the model ({reason}). "
            f"Transcribing again on the CPU with {cpu_repo}.")


class FasterWhisperEngine:
    """Transcribes with faster_whisper.WhisperModel on the best working device."""

    key = ENGINE_KEY

    def __init__(self, choose_device_plan=detect_device_plan,
                 load_whisper_module=import_faster_whisper,
                 write_progress_line=print_progress_line,
                 decode_audio=decode_for_whisper,
                 make_cpu_plan=cpu_fallback_plan,
                 report=print):
        self._choose_device_plan = choose_device_plan
        self._decode_audio = decode_audio
        self._load_whisper_module = load_whisper_module
        self._write_progress_line = write_progress_line
        self._make_cpu_plan = make_cpu_plan
        self._report = report
        self._device_plan = None

    @property
    def device_plan(self):
        """Chosen once, on first use, then reused (unless the GPU later fails)."""
        if self._device_plan is None:
            self._device_plan = self._choose_device_plan()
        return self._device_plan

    @property
    def runtime_key(self):
        return self.device_plan.runtime_key

    def transcribe(self, audio_path, profile):
        """Run Whisper and return ``{"text", "segments"}`` like mlx-whisper does."""
        samples = self._decode_audio(audio_path)
        plan = self.device_plan
        try:
            return self._transcribe_on(plan, samples, profile)
        except Exception as error:  # CTranslate2 raises RuntimeError for CUDA failures
            if plan.device != "cuda":
                raise
            self._switch_to_cpu(error, profile)
        return self._transcribe_on(self.device_plan, samples, profile)

    def _switch_to_cpu(self, error, profile):
        """Use the CPU plan from now on, and say so in one line."""
        self._device_plan = self._make_cpu_plan()
        cpu_source = profile.source_for(self._device_plan.runtime_key)
        self._report(gpu_failure_message(error, cpu_source.repo))

    def _transcribe_on(self, plan, samples, profile):
        """Load the profile's model for `plan` and decode the samples with it."""
        source = profile.source_for(plan.runtime_key)
        whisper_module = self._load_whisper_module()
        model = whisper_module.WhisperModel(
            source.repo,
            device=plan.device,
            compute_type=plan.compute_type,
            cpu_threads=plan.cpu_threads,
        )
        options = {**decoding_options(profile.language), "beam_size": GREEDY_BEAM_SIZE}
        # segments is a generator: decoding happens while it is consumed.
        segments, _ = model.transcribe(samples, **options)
        return self._collect(segments)

    def _collect(self, segments):
        """Turn Segment objects into plain dicts, printing each as it is decoded."""
        converted = []
        for segment in segments:
            self._write_progress_line(format_segment_line(segment.start, segment.end, segment.text))
            converted.append({"start": segment.start, "end": segment.end, "text": segment.text})
        return {"text": "".join(entry["text"] for entry in converted), "segments": converted}
