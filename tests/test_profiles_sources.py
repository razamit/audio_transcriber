"""Each profile carries one model per runtime, all built for that profile's language."""

import unittest

from profiles import PROFILES, RUNTIME_CT2_CPU, RUNTIME_CT2_CUDA, RUNTIME_MLX

ALL_RUNTIMES = {RUNTIME_MLX, RUNTIME_CT2_CUDA, RUNTIME_CT2_CPU}


class ProfileSourceTests(unittest.TestCase):
    def test_every_profile_covers_every_runtime(self):
        for profile in PROFILES.values():
            with self.subTest(profile=profile.key):
                self.assertEqual(set(profile.sources), ALL_RUNTIMES)
                for source in profile.sources.values():
                    self.assertTrue(source.download_size.startswith("~"))

    def test_hebrew_sources(self):
        hebrew = PROFILES["he"]
        self.assertEqual(hebrew.language, "he")
        self.assertEqual(hebrew.source_for(RUNTIME_MLX).repo,
                         "mlx-community/ivrit-ai-whisper-large-v3-mlx")
        self.assertEqual(hebrew.source_for(RUNTIME_CT2_CUDA).repo, "ivrit-ai/whisper-large-v3-ct2")
        self.assertEqual(hebrew.source_for(RUNTIME_CT2_CPU).repo,
                         "ivrit-ai/whisper-large-v3-turbo-ct2")

    def test_every_hebrew_source_is_an_ivrit_ai_finetune(self):
        # The pairing guarantee: no runtime may fall back to stock Whisper for Hebrew.
        for source in PROFILES["he"].sources.values():
            self.assertIn("ivrit-ai", source.repo)

    def test_english_ct2_sources_are_the_repo_faster_whisper_calls_large_v3_turbo(self):
        english = PROFILES["en"]
        self.assertEqual(english.source_for(RUNTIME_MLX).repo, "mlx-community/whisper-large-v3-turbo")
        for runtime in (RUNTIME_CT2_CUDA, RUNTIME_CT2_CPU):
            self.assertEqual(english.source_for(runtime).repo,
                             "mobiuslabsgmbh/faster-whisper-large-v3-turbo")

    def test_unknown_runtime_is_a_clear_error(self):
        with self.assertRaises(ValueError) as caught:
            PROFILES["he"].source_for("onnx")
        self.assertIn("onnx", str(caught.exception))

    def test_profiles_stay_hashable(self):
        self.assertEqual(len({PROFILES["he"], PROFILES["en"]}), 2)


if __name__ == "__main__":
    unittest.main()
