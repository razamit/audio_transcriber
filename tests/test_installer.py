"""install.py: the checks and the pip commands it builds. Nothing is installed."""

import os
import subprocess
import sys
import types
import unittest

import install

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "venv-python"


def pip(*arguments):
    return [PY, "-m", "pip", "install", *arguments]


class InstallCommandTests(unittest.TestCase):
    def test_base_install(self):
        self.assertEqual(install.build_install_commands(PY, with_diarization=False, gpu=False), [
            pip("--upgrade", "pip", "--quiet"),
            pip("-r", "requirements.txt"),
        ])

    def test_diarization_adds_its_requirements_last(self):
        commands = install.build_install_commands(PY, with_diarization=True, gpu=False)
        self.assertEqual(commands[-1], pip("-r", "requirements-diarize.txt"))
        self.assertEqual(len(commands), 3)

    def test_gpu_adds_cublas_and_cudnn(self):
        commands = install.build_install_commands(PY, with_diarization=False, gpu=True)
        self.assertEqual(commands[-1], pip("nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*"))

    def test_gpu_with_diarization_puts_cuda_torch_before_pyannote(self):
        commands = install.build_install_commands(PY, with_diarization=True, gpu=True)
        self.assertEqual(commands, [
            pip("--upgrade", "pip", "--quiet"),
            pip("-r", "requirements.txt"),
            pip("nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*"),
            # A CPU torch already installed would count as "already satisfied".
            [PY, "-m", "pip", "uninstall", "-y", "torch", "torchaudio"],
            pip("torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu126"),
            pip("-r", "requirements-diarize.txt"),
        ])

    def test_torch_is_only_uninstalled_for_a_gpu_diarization_install(self):
        for with_diarization, gpu in ((True, False), (False, True), (False, False)):
            with self.subTest(with_diarization=with_diarization, gpu=gpu):
                commands = install.build_install_commands(PY, with_diarization, gpu)
                self.assertFalse(any("uninstall" in command for command in commands))

    def test_venv_python_location_per_platform(self):
        windows = install.venv_python_path("proj", "win32")
        self.assertEqual(windows, os.path.join("proj", ".venv", "Scripts", "python.exe"))
        self.assertEqual(install.venv_python_path("proj", "linux"),
                         os.path.join("proj", ".venv", "bin", "python"))

    def test_venv_is_made_by_the_python_running_the_installer(self):
        self.assertEqual(install.build_venv_command("py3"), ["py3", "-m", "venv", ".venv"])

    def test_commands_run_in_order_in_the_project_folder(self):
        ran = []
        install.run_commands([["a"], ["b"]], runner=lambda command, cwd: ran.append((command, cwd)),
                             folder="proj")
        self.assertEqual(ran, [(["a"], "proj"), (["b"], "proj")])


class InstallCheckTests(unittest.TestCase):
    def test_python_version(self):
        self.assertIn("3.10 or newer", install.python_version_problem((3, 9, 7)))
        self.assertIsNone(install.python_version_problem((3, 10, 0)))
        self.assertIsNone(install.python_version_problem((3, 13, 1)))

    def test_python_315_warns_where_faster_whisper_is_installed(self):
        windows = install.new_python_warning((3, 15, 0), "win32", "AMD64")
        self.assertIn("Python 3.15", windows)
        self.assertIn("may fail", windows)
        self.assertIn("py -3.13 install.py", windows)
        self.assertIn("python3.13 install.py", install.new_python_warning((3, 16, 1), "linux", "x86_64"))
        self.assertIsNotNone(install.new_python_warning((3, 15, 0), "darwin", "x86_64"))  # Intel Mac

    def test_no_warning_for_older_python_or_apple_silicon(self):
        self.assertIsNone(install.new_python_warning((3, 14, 2), "win32", "AMD64"))
        self.assertIsNone(install.new_python_warning((3, 15, 0), "darwin", "arm64"))

    def test_ffmpeg_hint_follows_the_platform(self):
        missing = lambda name: None
        self.assertIn("winget install ffmpeg", install.ffmpeg_problem(missing, "win32"))
        self.assertIn("brew install ffmpeg", install.ffmpeg_problem(missing, "darwin"))
        self.assertIsNone(install.ffmpeg_problem(lambda name: "/bin/ffmpeg", "win32"))

    def test_gpu_flag_is_for_windows_and_linux_only(self):
        self.assertIsNone(install.gpu_problem(True, "win32"))
        self.assertIsNone(install.gpu_problem(True, "linux"))
        self.assertIsNone(install.gpu_problem(False, "darwin"))
        self.assertIn("Windows and Linux", install.gpu_problem(True, "darwin"))

    def test_windows_on_arm_python_is_refused_up_front(self):
        self.assertIn("ARM64", install.platform_problem("win-arm64"))
        self.assertIsNone(install.platform_problem("win-amd64"))

    def test_closing_message_uses_the_platform_commands(self):
        options = types.SimpleNamespace(with_diarization=False, gpu=False)
        windows = install.closing_message(options, "win32")
        self.assertIn(r".venv\Scripts\python transcribe.py", windows)
        self.assertIn("py install.py --with-diarization", windows)
        self.assertIn(".venv/bin/python transcribe.py", install.closing_message(options, "darwin"))

    def test_arguments(self):
        options = install.parse_options(["--with-diarization", "--gpu"])
        self.assertEqual((options.with_diarization, options.gpu), (True, True))


@unittest.skipUnless(sys.platform == "darwin", "the refusal tested here is Mac-specific")
class SetupWrapperTests(unittest.TestCase):
    def test_setup_sh_passes_its_arguments_to_install_py(self):
        # --gpu is refused on a Mac before anything is created or installed.
        completed = subprocess.run(["bash", "setup.sh", "--gpu"], cwd=REPO_ROOT,
                                   capture_output=True, text=True, stdin=subprocess.DEVNULL)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("--gpu is for NVIDIA GPUs", completed.stderr)


if __name__ == "__main__":
    unittest.main()
