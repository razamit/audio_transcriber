"""Shared test helpers: a scratch folder with a fake audio file, and fake transcribers.

Nothing here loads a model. The fake transcribers write a transcript file the
way the real ones do, so tests can check paths and the result line without
Whisper or pyannote.
"""

import os
import shutil
import tempfile
import unittest


class ScratchFolderTestCase(unittest.TestCase):
    """Gives each test an empty folder holding `call.m4a` (not real audio)."""

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="transcriber-test-")
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        self.audio_path = os.path.join(self.folder, "call.m4a")
        with open(self.audio_path, "wb") as handle:
            handle.write(b"not really audio")

    def start_patch(self, patcher):
        """Start a mock patcher for this test only (enterContext needs 3.11)."""
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started


class FakeTranscribers:
    """Stand-ins for transcribe_only / transcribe_with_speakers that record calls."""

    def __init__(self, speaker_count=2, noise=""):
        self.speaker_count = speaker_count
        self.noise = noise
        self.calls = []

    def plain(self, audio_path, profile, output_path):
        self.calls.append(("plain", audio_path, profile.key, output_path, None))
        self._write(output_path)

    def with_speakers(self, audio_path, profile, output_path, speaker_count):
        self.calls.append(("speakers", audio_path, profile.key, output_path, speaker_count))
        self._write(output_path)
        return self.speaker_count

    def _write(self, output_path):
        if self.noise:
            print(self.noise)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("[00:00:00] hello\n")
