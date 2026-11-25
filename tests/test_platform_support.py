"""Small per-platform behaviour: paths, ffmpeg advice, torch device, streams, HF cache."""

import io
import ntpath
import os
import posixpath
import types
import unittest
from unittest import mock

import paths
import platform_hints
from console_streams import use_utf8_when_redirected
from diarization import download_help, install_help, select_torch_device
from tests.support import ScratchFolderTestCase
from transcription import hf_cache_root


class WindowsPathInputTests(ScratchFolderTestCase):
    def test_windows_keeps_backslashes_as_separators(self):
        typed = r"C:\Users\amit\Recordings\call (1).m4a"
        self.assertEqual(paths.sanitize_path_input(typed, platform_name="win32"), typed)

    def test_windows_strips_the_quotes_a_terminal_adds(self):
        typed = r'"C:\Users\amit\My Recordings\call.m4a"'
        self.assertEqual(paths.sanitize_path_input(typed, platform_name="win32"),
                         r"C:\Users\amit\My Recordings\call.m4a")

    def test_mac_still_unescapes_drag_and_drop(self):
        # Forward slashes so the test also runs on Windows, where the scratch
        # folder's own backslashes would otherwise be unescaped away.
        spaced = self.folder.replace("\\", "/") + "/my call.m4a"
        with open(spaced, "wb") as handle:
            handle.write(b"x")
        dragged = spaced.replace(" ", "\\ ")
        self.assertEqual(paths.sanitize_path_input(dragged, platform_name="darwin"), spaced)


class FfmpegHintTests(unittest.TestCase):
    def test_one_hint_per_platform(self):
        self.assertEqual(platform_hints.ffmpeg_install_hint("darwin"), "brew install ffmpeg")
        self.assertTrue(platform_hints.ffmpeg_install_hint("win32").startswith("winget install ffmpeg"))
        self.assertIn("apt install ffmpeg", platform_hints.ffmpeg_install_hint("linux"))

    def test_message_carries_the_hint(self):
        message = platform_hints.ffmpeg_missing_message("win32")
        self.assertIn("not found on PATH", message)
        self.assertIn("winget install ffmpeg", message)

    def test_interactive_check_uses_the_shared_message(self):
        with mock.patch.object(paths.shutil, "which", return_value=None), \
                self.assertRaises(SystemExit) as caught:
            paths.require_ffmpeg()
        self.assertEqual(caught.exception.code, platform_hints.ffmpeg_missing_message())

    def test_installer_and_venv_commands(self):
        self.assertEqual(platform_hints.installer_command("win32"), "py install.py")
        self.assertEqual(platform_hints.installer_command("linux"), "./setup.sh")
        self.assertEqual(platform_hints.venv_python_command("win32"), r".venv\Scripts\python")
        self.assertEqual(platform_hints.venv_python_command("darwin"), ".venv/bin/python")

    def test_venv_tools_and_environment_variables(self):
        self.assertEqual(platform_hints.venv_script_command("hf", "win32"), r".venv\Scripts\hf")
        self.assertEqual(platform_hints.venv_script_command("hf", "linux"), ".venv/bin/hf")
        self.assertEqual(platform_hints.set_environment_variable_command("A", "b", "darwin"), "export A=b")
        windows = platform_hints.set_environment_variable_command("A", "b", "win32")
        self.assertTrue(windows.startswith('$env:A = "b"'))
        self.assertIn("set A=b", windows)


class DiarizationHelpTests(unittest.TestCase):
    def test_windows_help_uses_windows_commands(self):
        text = install_help("win32") + download_help("win32")
        self.assertIn(r".venv\Scripts\python -m pip install -r requirements-diarize.txt", text)
        self.assertIn("py install.py --with-diarization", text)
        self.assertIn(r".venv\Scripts\hf auth login", text)
        self.assertIn('$env:HF_TOKEN = "hf_your_token_here"', text)
        self.assertNotIn(".venv/bin", text)
        self.assertNotIn("export ", text)

    def test_mac_and_linux_help_keep_the_unix_commands(self):
        text = install_help("linux") + download_help("linux")
        self.assertIn(".venv/bin/python -m pip install -r requirements-diarize.txt", text)
        self.assertIn("./setup.sh --with-diarization", text)
        self.assertIn(".venv/bin/hf auth login", text)
        self.assertIn("export HF_TOKEN=hf_your_token_here", text)


def fake_torch(cuda, mps):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        device=lambda name: name,
    )


class TorchDeviceTests(unittest.TestCase):
    def test_cuda_then_mps_then_cpu(self):
        self.assertEqual(select_torch_device(fake_torch(cuda=True, mps=True)), "cuda")
        self.assertEqual(select_torch_device(fake_torch(cuda=False, mps=True)), "mps")
        self.assertEqual(select_torch_device(fake_torch(cuda=False, mps=False)), "cpu")


class FakeStream:
    def __init__(self, encoding, console):
        self.encoding = encoding
        self.console = console
        self.reconfigured_to = None

    def isatty(self):
        return self.console

    def reconfigure(self, encoding):
        self.reconfigured_to = encoding


class Utf8StreamTests(unittest.TestCase):
    def test_redirected_legacy_stream_on_windows_becomes_utf8(self):
        stream = FakeStream("cp1255", console=False)
        use_utf8_when_redirected("win32", [stream])
        self.assertEqual(stream.reconfigured_to, "utf-8")

    def test_output_to_nul_counts_as_a_console_but_still_becomes_utf8(self):
        # `> NUL` gives a stream isatty() calls a terminal, on the legacy code page.
        stream = FakeStream("cp1252", console=True)
        use_utf8_when_redirected("win32", [stream])
        self.assertEqual(stream.reconfigured_to, "utf-8")

    def test_utf8_streams_and_other_platforms_are_left_alone(self):
        cases = [("win32", FakeStream("utf-8", console=True)),  # a real Windows console
                 ("win32", FakeStream("UTF-8", console=False)),
                 ("darwin", FakeStream("cp1255", console=False))]
        for platform_name, stream in cases:
            with self.subTest(platform=platform_name, encoding=stream.encoding):
                use_utf8_when_redirected(platform_name, [stream])
                self.assertIsNone(stream.reconfigured_to)

    def test_a_real_cp1252_stream_can_print_hebrew_afterwards(self):
        raw = io.BytesIO()
        # newline="\n" so Windows does not turn the line break into \r\n.
        stream = io.TextIOWrapper(raw, encoding="cp1252", newline="\n")
        with self.assertRaises(UnicodeEncodeError):
            stream.write("שלום")
            stream.flush()
        use_utf8_when_redirected("win32", [stream])
        stream.write("שלום\n")
        stream.flush()
        self.assertTrue(raw.getvalue().endswith("שלום\n".encode("utf-8")))


class HuggingFaceCacheTests(unittest.TestCase):
    def test_windows_default_is_a_clean_path_under_the_profile(self):
        root = hf_cache_root(environ={}, path_module=ntpath, home=r"C:\Users\amit")
        self.assertEqual(root, r"C:\Users\amit\.cache\huggingface\hub")

    def test_overrides_in_huggingface_hub_order(self):
        self.assertEqual(hf_cache_root({"HF_HUB_CACHE": "/c/hub", "HF_HOME": "/h"}, posixpath), "/c/hub")
        self.assertEqual(hf_cache_root({"HF_HOME": "/h"}, posixpath), "/h/hub")
        self.assertEqual(hf_cache_root({"XDG_CACHE_HOME": "/x"}, posixpath, home="/home/amit"),
                         "/x/huggingface/hub")
        self.assertEqual(hf_cache_root({}, posixpath, home="/home/amit"),
                         "/home/amit/.cache/huggingface/hub")


if __name__ == "__main__":
    unittest.main()
