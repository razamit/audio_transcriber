"""The faster-whisper engine against a fake faster_whisper module, plus the MLX engine's call.

No model loads. The fakes record what the engines pass, so the decoding
settings, the model choice per device and the result shape are all checked.
"""

import contextlib
import importlib.util
import os
import shutil
import subprocess
import types
import unittest

from audio_decoding import decode_for_whisper
from engines.decoding import decoding_options
from engines.device_plan import choose_device, cpu_plan, cpu_thread_count, detect_device_plan
from engines.faster_whisper_engine import FasterWhisperEngine
from engines.mlx_engine import MlxWhisperEngine
from engines.segment_progress import format_progress_timestamp, format_segment_line
from profiles import PROFILES
from tests.support import ScratchFolderTestCase

SHARED_SETTINGS = {"condition_on_previous_text": False, "hallucination_silence_threshold": 2.0}


class FakeWhisperModel:
    """Records its construction and transcribe kwargs; yields fixed segments."""

    instances = []

    def __init__(self, repo, **options):
        self.repo = repo
        self.options = options
        self.transcribe_calls = []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **options):
        self.transcribe_calls.append((audio, options))
        segments = (
            types.SimpleNamespace(start=0.0, end=2.5, text=" שלום לכולם."),
            types.SimpleNamespace(start=2.5, end=61.25, text=" Second line."),
        )
        return (segment for segment in segments), types.SimpleNamespace(language="he")


FAKE_FASTER_WHISPER = types.SimpleNamespace(WhisperModel=FakeWhisperModel)


class CudaOutOfMemoryOnLoad(FakeWhisperModel):
    """Fails to build on the GPU, as CTranslate2 does when the card is too small."""

    def __init__(self, repo, **options):
        super().__init__(repo, **options)
        if options["device"] == "cuda":
            raise RuntimeError("CUDA failed with error out of memory\nmore detail")


class CudaOutOfMemoryMidRun(FakeWhisperModel):
    """Builds, then fails on the GPU after the first segment has been decoded."""

    def transcribe(self, audio, **options):
        segments, info = super().transcribe(audio, **options)
        if self.options["device"] != "cuda":
            return segments, info

        def failing_segments():
            yield next(segments)
            raise RuntimeError("CUDA failed with error out of memory")

        return failing_segments(), info


class FakeProbe:
    """A CudaLibraryProbe stand-in: `missing` is the library that will not load."""

    def __init__(self, missing=None):
        self.missing = missing
        self.registered = False
        self.checked = False

    def register_library_folders(self):
        self.registered = True

    def first_unloadable_library(self):
        self.checked = True
        return self.missing


def fake_ctranslate2(gpu_count):
    return types.SimpleNamespace(get_cuda_device_count=lambda: gpu_count)


class EngineHarness(unittest.TestCase):
    def setUp(self):
        FakeWhisperModel.instances = []
        self.progress_lines = []
        self.reports = []

    def engine_for(self, gpu_count, missing_library=None, logical_cores=16,
                   model_class=FakeWhisperModel):
        self.probe = FakeProbe(missing_library)

        def plan():
            return detect_device_plan(self.probe, lambda: fake_ctranslate2(gpu_count),
                                      logical_cores, self.reports.append)

        whisper_module = types.SimpleNamespace(WhisperModel=model_class)
        return FasterWhisperEngine(plan, lambda: whisper_module, self.progress_lines.append,
                                   decode_audio=lambda path: ("samples of", path),
                                   make_cpu_plan=lambda: cpu_plan(logical_cores),
                                   report=self.reports.append)

    def model(self):
        self.assertEqual(len(FakeWhisperModel.instances), 1)
        return FakeWhisperModel.instances[0]


class FasterWhisperTranscribeTests(EngineHarness):
    def test_decoding_kwargs_are_exactly_the_shared_settings(self):
        self.engine_for(gpu_count=0).transcribe("call.m4a", PROFILES["he"])
        audio, options = self.model().transcribe_calls[0]
        # The model gets samples decoded by the ffmpeg CLI, not the path.
        self.assertEqual(audio, ("samples of", "call.m4a"))
        # Exact match: no VAD, greedy decoding like mlx-whisper, nothing else.
        self.assertEqual(options, {"language": "he", **SHARED_SETTINGS, "beam_size": 1})

    def test_segments_become_plain_dicts_and_text_is_joined(self):
        result = self.engine_for(gpu_count=0).transcribe("call.m4a", PROFILES["he"])
        self.assertEqual(result["segments"], [
            {"start": 0.0, "end": 2.5, "text": " שלום לכולם."},
            {"start": 2.5, "end": 61.25, "text": " Second line."},
        ])
        self.assertEqual(result["text"], " שלום לכולם. Second line.")

    def test_each_segment_is_printed_in_mlx_whisper_form(self):
        self.engine_for(gpu_count=0).transcribe("call.m4a", PROFILES["en"])
        self.assertEqual(self.progress_lines, [
            "[00:00.000 --> 00:02.500]  שלום לכולם.",
            "[00:02.500 --> 01:01.250]  Second line.",
        ])

    def test_cpu_uses_int8_the_turbo_hebrew_model_and_half_the_cores(self):
        self.engine_for(gpu_count=0, logical_cores=16).transcribe("call.m4a", PROFILES["he"])
        model = self.model()
        self.assertEqual(model.repo, "ivrit-ai/whisper-large-v3-turbo-ct2")
        self.assertEqual(model.options, {"device": "cpu", "compute_type": "int8", "cpu_threads": 8})

    def test_working_gpu_uses_float16_and_full_large_v3(self):
        self.engine_for(gpu_count=1).transcribe("call.m4a", PROFILES["he"])
        model = self.model()
        self.assertEqual(model.repo, "ivrit-ai/whisper-large-v3-ct2")
        self.assertEqual((model.options["device"], model.options["compute_type"]), ("cuda", "float16"))
        self.assertEqual(self.reports, ["  Transcription running on: cuda (float16)"])

    def test_gpu_without_its_libraries_falls_back_to_cpu_and_says_so(self):
        engine = self.engine_for(gpu_count=1, missing_library="cudnn_ops64_9.dll")
        engine.transcribe("call.m4a", PROFILES["he"])
        self.assertEqual(self.model().options["device"], "cpu")
        self.assertEqual(engine.runtime_key, "ct2-cpu")
        self.assertEqual(len(self.reports), 2)
        self.assertIn("cudnn_ops64_9.dll", self.reports[0])
        self.assertIn("--gpu", self.reports[0])
        self.assertTrue(self.reports[1].startswith("  Transcription running on: cpu (int8"))

    def test_libraries_are_registered_first_and_only_checked_when_a_gpu_exists(self):
        engine = self.engine_for(gpu_count=0)
        self.assertEqual(engine.runtime_key, "ct2-cpu")
        self.assertTrue(self.probe.registered)
        self.assertFalse(self.probe.checked)

    def test_device_is_chosen_once(self):
        calls = []
        engine = FasterWhisperEngine(lambda: calls.append(1) or choose_device(
            lambda: 0, lambda: None, 8, report=lambda line: None))
        self.assertEqual((engine.runtime_key, engine.runtime_key), ("ct2-cpu", "ct2-cpu"))
        self.assertEqual(len(calls), 1)


class GpuFailureFallbackTests(EngineHarness):
    def test_model_that_will_not_load_on_the_gpu_is_retried_on_the_cpu(self):
        engine = self.engine_for(gpu_count=1, model_class=CudaOutOfMemoryOnLoad)
        result = engine.transcribe("call.m4a", PROFILES["he"])
        gpu_try, cpu_try = FakeWhisperModel.instances
        self.assertEqual((gpu_try.repo, gpu_try.options["device"]), ("ivrit-ai/whisper-large-v3-ct2", "cuda"))
        # The CPU plan has its own runtime key, so the profile's CPU model is used.
        self.assertEqual((cpu_try.repo, cpu_try.options["device"]),
                         ("ivrit-ai/whisper-large-v3-turbo-ct2", "cpu"))
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(engine.runtime_key, "ct2-cpu")

    def test_the_fallback_is_announced_in_one_line(self):
        self.engine_for(gpu_count=1, model_class=CudaOutOfMemoryOnLoad).transcribe("call.m4a", PROFILES["he"])
        self.assertEqual(self.reports[-1],
                         "  The GPU could not run the model (CUDA failed with error out of memory). "
                         "Transcribing again on the CPU with ivrit-ai/whisper-large-v3-turbo-ct2.")

    def test_failure_partway_through_decoding_starts_over_on_the_cpu(self):
        engine = self.engine_for(gpu_count=1, model_class=CudaOutOfMemoryMidRun)
        result = engine.transcribe("call.m4a", PROFILES["en"])
        self.assertEqual([model.options["device"] for model in FakeWhisperModel.instances], ["cuda", "cpu"])
        self.assertEqual([entry["start"] for entry in result["segments"]], [0.0, 2.5])

    def test_later_files_go_straight_to_the_cpu(self):
        engine = self.engine_for(gpu_count=1, model_class=CudaOutOfMemoryOnLoad)
        engine.transcribe("first.m4a", PROFILES["he"])
        engine.transcribe("second.m4a", PROFILES["he"])
        self.assertEqual([model.options["device"] for model in FakeWhisperModel.instances], ["cuda", "cpu", "cpu"])

    def test_a_cpu_failure_is_not_retried(self):
        class BrokenEverywhere(FakeWhisperModel):
            def __init__(self, repo, **options):
                super().__init__(repo, **options)
                raise RuntimeError("model files are corrupt")

        engine = self.engine_for(gpu_count=0, model_class=BrokenEverywhere)
        with self.assertRaisesRegex(RuntimeError, "corrupt"):
            engine.transcribe("call.m4a", PROFILES["he"])
        self.assertEqual(len(FakeWhisperModel.instances), 1)


class CpuThreadTests(unittest.TestCase):
    def test_half_the_cores_but_never_below_ctranslate2s_default(self):
        self.assertEqual(cpu_thread_count(16), 8)
        self.assertEqual(cpu_thread_count(4), 4)
        self.assertEqual(cpu_thread_count(None), 4)


class FakeMlxWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, **options):
        self.calls.append((audio_path, options))
        return {"text": " hi", "segments": []}


class MlxEngineTests(unittest.TestCase):
    def test_mlx_call_uses_the_profile_repo_and_the_shared_settings(self):
        fake = FakeMlxWhisper()
        wrapped = []

        @contextlib.contextmanager
        def recording_wrapper():
            wrapped.append("entered")
            yield

        MlxWhisperEngine(lambda: fake, recording_wrapper).transcribe("call.m4a", PROFILES["he"])
        audio_path, options = fake.calls[0]
        self.assertEqual(audio_path, "call.m4a")
        self.assertEqual(options, {
            "path_or_hf_repo": "mlx-community/ivrit-ai-whisper-large-v3-mlx",
            "verbose": True, "language": "he", **SHARED_SETTINGS,
        })
        self.assertEqual(wrapped, ["entered"])

    def test_both_engines_draw_on_one_definition(self):
        self.assertEqual(decoding_options("en"), {"language": "en", **SHARED_SETTINGS})


@unittest.skipUnless(shutil.which("ffmpeg") and importlib.util.find_spec("numpy"),
                     "needs the ffmpeg CLI and numpy")
class WhisperAudioDecodingTests(ScratchFolderTestCase):
    def test_half_a_second_of_tone_becomes_8000_float_samples(self):
        tone = os.path.join(self.folder, "tone.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=0.5:sample_rate=44100", tone], check=True)
        samples = decode_for_whisper(tone)
        self.assertEqual((samples.dtype.name, samples.shape), ("float32", (8000,)))
        self.assertTrue(0.05 < float(abs(samples).max()) <= 1.0)

    def test_undecodable_file_raises(self):
        with self.assertRaises(RuntimeError):
            decode_for_whisper(self.audio_path)  # "not really audio"


class ProgressFormatTests(unittest.TestCase):
    def test_timestamps_match_mlx_whisper(self):
        self.assertEqual(format_progress_timestamp(0), "00:00.000")
        self.assertEqual(format_progress_timestamp(65.4321), "01:05.432")
        self.assertEqual(format_progress_timestamp(3661.5), "01:01:01.500")

    def test_line_shape(self):
        self.assertEqual(format_segment_line(1, 2, " hi"), "[00:01.000 --> 00:02.000]  hi")


if __name__ == "__main__":
    unittest.main()
