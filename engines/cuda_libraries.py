"""Finding and test-loading the NVIDIA libraries CTranslate2 needs for the GPU.

CTranslate2 4.x runs on an NVIDIA GPU only with cuBLAS for CUDA 12 and cuDNN 9
present. When they are missing it does not raise a Python error: on Windows a
missing cuDNN kills the whole process partway through a run. So before the GPU
is chosen, this module loads every needed library itself, with ctypes, where a
failure is an ordinary OSError.

The libraries usually come from pip packages (nvidia-cublas-cu12 and
nvidia-cudnn-cu12, which `install.py --gpu` installs) or from a CUDA build of
PyTorch, which bundles them in torch/lib. Neither location is on the default
library search path, so their folders are added first: with
os.add_dll_directory on Windows, and by loading each library from its full
path on Linux. A library loaded once stays loaded, so when CTranslate2 later
asks for it by name it gets the copy already in memory.

A CUDA toolkit and cuDNN installed system-wide are found through PATH on
Windows, by full path, because ctypes leaves PATH out of its DLL search. The
cuDNN 9 parts outside the required list are loaded too when they sit beside
the required ones, since cuDNN would otherwise open them later by name.
"""

import ctypes
import fnmatch
import importlib.util
import os
import sys

# Loaded in this order: each cuBLAS and cuDNN part needs the one before it.
CUDA_LIBRARY_NAMES = {
    "win32": (
        "cublasLt64_12.dll", "cublas64_12.dll",
        "cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll",
    ),
    "linux": (
        "libcublasLt.so.12", "libcublas.so.12",
        "libcudnn.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9",
    ),
}

# Every cuDNN 9 library file, required or not, as a lower-case file name pattern.
OPTIONAL_CUDNN_PATTERN = {"win32": "cudnn*64_9.dll", "linux": "libcudnn*.so.9"}

# The folder inside each nvidia.* pip package that holds its libraries.
NVIDIA_LIBRARY_SUBFOLDER = {"win32": "bin", "linux": "lib"}


def package_folders(package_name, find_spec=importlib.util.find_spec):
    """Where an installed package lives, without importing it. Empty if absent."""
    try:
        spec = find_spec(package_name)
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    return list(spec.submodule_search_locations or [])


def candidate_library_folders(platform_name, find_spec=importlib.util.find_spec,
                              list_folder=os.listdir, is_folder=os.path.isdir):
    """Existing folders that may hold the CUDA libraries, most specific first."""
    subfolder = NVIDIA_LIBRARY_SUBFOLDER.get(platform_name, "lib")
    folders = []
    for nvidia_root in package_folders("nvidia", find_spec):
        try:
            children = sorted(list_folder(nvidia_root))
        except OSError:
            continue
        folders.extend(os.path.join(nvidia_root, child, subfolder) for child in children)
    for torch_root in package_folders("torch", find_spec):
        folders.append(os.path.join(torch_root, "lib"))
    return [folder for folder in folders if is_folder(folder)]


def is_cudnn_library(file_name):
    return file_name.lower().startswith(("cudnn", "libcudnn"))


def load_shared_library(path):
    """Load one library globally, so later lookups by name find this copy."""
    return ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_GLOBAL", 0))


class CudaLibraryProbe:
    """Makes the CUDA library folders searchable, then checks every library loads.

    Every operating-system call is injected, so tests can play Windows or
    Linux on any machine.
    """

    def __init__(self, platform_name=None, load_library=load_shared_library,
                 add_dll_directory=None, find_folders=candidate_library_folders,
                 path_exists=os.path.exists, list_folder=os.listdir, search_path=None):
        self.platform_name = sys.platform if platform_name is None else platform_name
        self._load_library = load_library
        self._add_dll_directory = add_dll_directory or getattr(os, "add_dll_directory", None)
        self._find_folders = find_folders
        self._path_exists = path_exists
        self._list_folder = list_folder
        self._search_path = os.environ.get("PATH", "") if search_path is None else search_path
        self._folders = None
        # Kept so the libraries and DLL folders stay registered for the process.
        self._loaded_libraries = []
        self._dll_directory_handles = []

    def register_library_folders(self):
        """Find the candidate folders and, on Windows, add them to the DLL search."""
        if self._folders is not None:
            return self._folders
        self._folders = self._find_folders(self.platform_name)
        if self.platform_name == "win32" and self._add_dll_directory is not None:
            for folder in self._folders:
                try:
                    self._dll_directory_handles.append(self._add_dll_directory(folder))
                except OSError:
                    continue
        return self._folders

    def first_unloadable_library(self):
        """The first needed library that will not load, or None if all of them do.

        Once the required ones load, the other cuDNN 9 parts found beside them
        are loaded too, so none is left for cuDNN to load lazily mid-run.
        """
        names = CUDA_LIBRARY_NAMES.get(self.platform_name)
        if names is None:
            return "CUDA libraries (no NVIDIA support on this platform)"
        folders = self.register_library_folders() + self._path_folders()
        cudnn_folders = []
        for name in names:
            loaded_from = self._load_from_anywhere(name, folders)
            if loaded_from is None:
                return name
            if is_cudnn_library(name) and os.path.dirname(loaded_from):
                cudnn_folders.append(os.path.dirname(loaded_from))
        self._preload_optional_cudnn(dict.fromkeys(cudnn_folders), set(names))
        return None

    def _path_folders(self):
        """PATH folders, searched on Windows only.

        ctypes on Windows loads a bare DLL name with a search that leaves PATH
        out, so a CUDA toolkit and cuDNN installed system-wide and put on PATH
        would never be found by name. Each PATH folder is tried by full path.
        """
        if self.platform_name != "win32":
            return []
        return [folder.strip().strip('"') for folder in self._search_path.split(";")
                if folder.strip()]

    def _load_from_anywhere(self, name, folders):
        """The path or name that loaded: each folder's copy, then the bare name."""
        full_paths = [os.path.join(folder, name) for folder in folders]
        candidates = [path for path in full_paths if self._path_exists(path)] + [name]
        for candidate in candidates:
            if self._try_load(candidate):
                return candidate
        return None

    def _try_load(self, path):
        try:
            self._loaded_libraries.append(self._load_library(path))
            return True
        except OSError:
            return False

    def _preload_optional_cudnn(self, folders, required_names):
        """Load every other cuDNN 9 file in `folders`; failures are ignored.

        cuDNN 9 is split into more libraries than the required list (graph,
        engines, heuristic, adv and so on) and opens some of them by plain
        name only when an operation first needs them. That late lookup does
        not search the folders found here, so it would fail and end the
        process mid-run. Loaded now, they are already in memory when asked for.
        """
        required = {name.lower() for name in required_names}
        for folder in folders:
            for file_name in self._optional_cudnn_files(folder, required):
                self._try_load(os.path.join(folder, file_name))

    def _optional_cudnn_files(self, folder, required):
        """cuDNN 9 file names in `folder` that are not already required, sorted."""
        pattern = OPTIONAL_CUDNN_PATTERN[self.platform_name]
        try:
            file_names = sorted(self._list_folder(folder))
        except OSError:
            return []
        return [name for name in file_names
                if name.lower() not in required and fnmatch.fnmatchcase(name.lower(), pattern)]
