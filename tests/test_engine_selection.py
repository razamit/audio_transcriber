"""Which engine runs on which machine, the TRANSCRIBE_ENGINE override, and the wiring."""

import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import transcription
from engines import selection
from engines.faster_whisper_engine import FasterWhisperEngine
from engines.mlx_engine import MlxWhisperEngine
from profiles import PROFILES, RUNTIME_CT2_CPU, RUNTIME_MLX


def importable_set(*module_names):
    """A stand-in for module_importable that knows only the given modules."""
    return lambda module_name: module_name in module_names


BOTH_ENGINES = importable_set("mlx_whisper", "faster_whisper")


def choose(platform_name, machine, importable=BOTH_ENGINES, environ=None):
    return selection.choose_engine_key(
        environ=environ or {}, platform_name=platform_name, machine=machine, importable=importable
    )


class AutomaticSelectionTests(unittest.TestCase):
    def test_apple_silicon_with_mlx_gets_mlx(self):
        self.assertEqual(choose("darwin", "arm64"), "mlx")

    def test_apple_silicon_without_mlx_falls_to_faster_whisper(self):
        self.assertEqual(choose("darwin", "arm64", importable_set("faster_whisper")), "faster-whisper")

    def test_intel_mac_gets_faster_whisper_even_with_mlx_installed(self):
        self.assertEqual(choose("darwin", "x86_64"), "faster-whisper")

    def test_windows_and_linux_get_faster_whisper(self):
        self.assertEqual(choose("win32", "AMD64"), "faster-whisper")
        self.assertEqual(choose("linux", "x86_64"), "faster-whisper")

    def test_windows_with_only_mlx_is_unavailable(self):
        with self.assertRaises(selection.EngineUnavailable):
            choose("win32", "AMD64", importable_set("mlx_whisper"))

    def test_no_engine_message_names_the_platform_installer(self):
        nothing = importable_set()
        with self.assertRaises(selection.EngineUnavailable) as on_windows:
            choose("win32", "AMD64", nothing)
        self.assertIn("py install.py", str(on_windows.exception))
        with self.assertRaises(selection.EngineUnavailable) as on_mac:
            choose("darwin", "arm64", nothing)
        self.assertIn("./setup.sh", str(on_mac.exception))


class OverrideTests(unittest.TestCase):
    def test_override_picks_faster_whisper_on_apple_silicon(self):
        environ = {"TRANSCRIBE_ENGINE": "faster-whisper"}
        self.assertEqual(choose("darwin", "arm64", environ=environ), "faster-whisper")

    def test_override_is_trimmed_and_case_insensitive(self):
        self.assertEqual(choose("darwin", "arm64", environ={"TRANSCRIBE_ENGINE": " MLX "}), "mlx")

    def test_blank_override_means_automatic(self):
        self.assertEqual(choose("darwin", "arm64", environ={"TRANSCRIBE_ENGINE": "  "}), "mlx")

    def test_unknown_override_is_an_error_not_a_fallback(self):
        with self.assertRaises(selection.EngineUnavailable) as caught:
            choose("darwin", "arm64", environ={"TRANSCRIBE_ENGINE": "whisper-cpp"})
        self.assertIn("mlx, faster-whisper", str(caught.exception))

    def test_override_naming_a_missing_engine_says_how_to_install_it(self):
        with self.assertRaises(selection.EngineUnavailable) as caught:
            choose("win32", "AMD64", importable_set("mlx_whisper"),
                   environ={"TRANSCRIBE_ENGINE": "faster-whisper"})
        message = str(caught.exception)
        self.assertIn("faster-whisper==1.2.1", message)
        self.assertIn(r".venv\Scripts\python -m pip", message)


class EngineConstructionTests(unittest.TestCase):
    def test_build_engine_returns_matching_engines(self):
        self.assertIsInstance(selection.build_engine("mlx"), MlxWhisperEngine)
        self.assertIsInstance(selection.build_engine("faster-whisper"), FasterWhisperEngine)
        self.assertEqual(selection.build_engine("mlx").runtime_key, RUNTIME_MLX)

    def test_select_engine_exits_with_the_install_message(self):
        with self.assertRaises(SystemExit) as caught:
            selection.select_engine(environ={}, platform_name="linux", machine="x86_64",
                                    importable=importable_set())
        self.assertIn("No transcription engine is installed", str(caught.exception.code))

    def test_engine_key_or_none_reports_without_exiting(self):
        self.assertIsNone(selection.engine_key_or_none(
            environ={}, platform_name="linux", machine="x86_64", importable=importable_set()))
        self.assertEqual(selection.engine_key_or_none(
            environ={}, platform_name="darwin", machine="arm64", importable=BOTH_ENGINES), "mlx")


class FakeEngine:
    key = "fake"
    runtime_key = RUNTIME_CT2_CPU

    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, profile):
        self.calls.append((audio_path, profile.key))
        return {"text": " hi", "segments": [{"start": 0.0, "end": 1.0, "text": " hi"}]}


class TranscriptionWiringTests(unittest.TestCase):
    """transcription.py keeps its signatures and asks the engine which model to use."""

    def setUp(self):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        self.cache_root = cache.name
        patcher = mock.patch.dict(os.environ, {"HF_HUB_CACHE": self.cache_root})
        patcher.start()
        self.addCleanup(patcher.stop)

    def announce(self, profile, engine):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            transcription.announce_first_run_download(profile, engine)
        return printed.getvalue()

    def test_announcement_names_the_runtime_specific_model(self):
        message = self.announce(PROFILES["he"], FakeEngine())
        self.assertIn("ivrit-ai/whisper-large-v3-turbo-ct2", message)
        self.assertIn("~1.6GB", message)

    def test_no_announcement_once_that_model_is_cached(self):
        os.makedirs(os.path.join(self.cache_root, "models--ivrit-ai--whisper-large-v3-turbo-ct2"))
        self.assertEqual(self.announce(PROFILES["he"], FakeEngine()), "")

    def test_run_transcription_delegates_to_the_engine(self):
        engine = FakeEngine()
        result = transcription.run_transcription("call.m4a", PROFILES["en"], engine)
        self.assertEqual(engine.calls, [("call.m4a", "en")])
        self.assertEqual(result["segments"][0]["text"], " hi")

    def test_default_engine_is_chosen_once_per_process(self):
        transcription.active_engine.cache_clear()
        self.addCleanup(transcription.active_engine.cache_clear)
        engine = FakeEngine()
        with mock.patch.object(transcription, "select_engine", return_value=engine) as chooser:
            transcription.run_transcription("a.m4a", PROFILES["en"])
            transcription.run_transcription("b.m4a", PROFILES["en"])
        chooser.assert_called_once_with()
        self.assertEqual(len(engine.calls), 2)


if __name__ == "__main__":
    unittest.main()
