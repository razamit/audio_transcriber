"""Finding and test-loading the CUDA libraries, with Windows and Linux simulated."""

import os
import types
import unittest

from engines.cuda_libraries import CUDA_LIBRARY_NAMES, CudaLibraryProbe, candidate_library_folders


class FakeLoader:
    """Loads anything except names (or paths ending in names) listed as broken."""

    def __init__(self, broken=()):
        self.broken = set(broken)
        self.loaded = []

    def __call__(self, path):
        if os.path.basename(path) in self.broken:
            raise OSError(f"cannot load {path}")
        self.loaded.append(path)
        return object()


class PathOnlyLoader(FakeLoader):
    """Loads full paths only: a bare name fails, as with PATH-only DLLs on Windows."""

    def __call__(self, path):
        if os.path.basename(path) == path:
            raise OSError(f"cannot find {path}")
        return super().__call__(path)


def probe_for(platform_name, folders, loader, existing_paths=(), added=None,
              search_path="", folder_contents=None):
    folder_contents = folder_contents or {}

    def list_folder(folder):
        if folder not in folder_contents:
            raise OSError(f"no folder {folder}")
        return folder_contents[folder]

    return CudaLibraryProbe(
        platform_name=platform_name,
        load_library=loader,
        add_dll_directory=(added.append if added is not None else lambda folder: None),
        find_folders=lambda platform_name: list(folders),
        path_exists=lambda path: path in existing_paths,
        list_folder=list_folder,
        search_path=search_path,
    )


class WindowsProbeTests(unittest.TestCase):
    FOLDERS = [os.path.join("site", "nvidia", "cudnn", "bin"), os.path.join("site", "torch", "lib")]

    def test_folders_are_added_to_the_dll_search_and_all_libraries_load(self):
        added, loader = [], FakeLoader()
        probe = probe_for("win32", self.FOLDERS, loader, added=added)
        self.assertIsNone(probe.first_unloadable_library())
        self.assertEqual(added, self.FOLDERS)
        self.assertEqual(loader.loaded, list(CUDA_LIBRARY_NAMES["win32"]))

    def test_a_folder_copy_is_preferred_over_the_system_search(self):
        cudnn_copy = os.path.join(self.FOLDERS[0], "cudnn64_9.dll")
        loader = FakeLoader()
        probe_for("win32", self.FOLDERS, loader, existing_paths={cudnn_copy}).first_unloadable_library()
        self.assertIn(cudnn_copy, loader.loaded)

    def test_missing_cudnn_is_named(self):
        probe = probe_for("win32", self.FOLDERS, FakeLoader(broken={"cudnn_ops64_9.dll"}))
        self.assertEqual(probe.first_unloadable_library(), "cudnn_ops64_9.dll")

    def test_a_system_install_on_path_is_loaded_by_full_path(self):
        cuda_bin, cudnn_bin = r"C:\CUDA\v12.6\bin", r"C:\cuDNN\bin\12.6"
        system_copies = {os.path.join(cuda_bin, name) for name in ("cublasLt64_12.dll", "cublas64_12.dll")}
        system_copies |= {os.path.join(cudnn_bin, name)
                          for name in CUDA_LIBRARY_NAMES["win32"] if name.startswith("cudnn")}
        loader = PathOnlyLoader()
        probe = probe_for("win32", [], loader, existing_paths=system_copies,
                          search_path=f'C:\\Windows;{cuda_bin}; ;"{cudnn_bin}"')
        self.assertIsNone(probe.first_unloadable_library())
        self.assertEqual(set(loader.loaded), system_copies)

    def test_without_path_a_system_install_is_not_found(self):
        probe = probe_for("win32", [], PathOnlyLoader(), search_path="")
        self.assertEqual(probe.first_unloadable_library(), "cublasLt64_12.dll")

    def test_path_is_not_searched_outside_windows(self):
        folder = "/usr/local/cuda/lib64"
        copy = os.path.join(folder, "libcublas.so.12")
        loader = FakeLoader()
        probe_for("linux", [], loader, existing_paths={copy}, search_path=folder).first_unloadable_library()
        self.assertNotIn(copy, loader.loaded)

    def test_other_cudnn_parts_beside_the_required_ones_are_preloaded(self):
        folder = self.FOLDERS[0]
        files = ["cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_graph64_9.dll",
                 "cudnn_engines_precompiled64_9.dll", "cudnn_adv64_9.dll", "readme.txt", "cudnn64_8.dll"]
        existing = {os.path.join(folder, name) for name in files}
        loader = FakeLoader(broken={"cudnn_adv64_9.dll"})  # an optional failure is ignored
        probe = probe_for("win32", self.FOLDERS, loader, existing_paths=existing,
                          folder_contents={folder: files})
        self.assertIsNone(probe.first_unloadable_library())
        optional = loader.loaded[len(CUDA_LIBRARY_NAMES["win32"]):]
        self.assertEqual(optional, [os.path.join(folder, "cudnn_engines_precompiled64_9.dll"),
                                    os.path.join(folder, "cudnn_graph64_9.dll")])

    def test_folders_are_registered_only_once(self):
        added = []
        probe = probe_for("win32", self.FOLDERS, FakeLoader(), added=added)
        probe.register_library_folders()
        probe.first_unloadable_library()
        self.assertEqual(len(added), len(self.FOLDERS))


class LinuxAndMacProbeTests(unittest.TestCase):
    def test_linux_loads_sonames_without_touching_the_dll_search(self):
        added, loader = [], FakeLoader()
        probe = probe_for("linux", ["lib"], loader, added=added)
        self.assertIsNone(probe.first_unloadable_library())
        self.assertEqual(added, [])
        self.assertIn("libcudnn_ops.so.9", loader.loaded)

    def test_linux_preloads_other_cudnn_parts_beside_the_required_ones(self):
        folder = os.path.join("site", "nvidia", "cudnn", "lib")
        files = ["libcudnn.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9", "libcudnn_heuristic.so.9",
                 "libcudnn_engines_runtime_compiled.so.9", "libcudnn_graph.so.9.1.0", "libcublas.so.12"]
        loader = FakeLoader()
        probe = probe_for("linux", [folder], loader, existing_paths={os.path.join(folder, name) for name in files},
                          folder_contents={folder: files})
        self.assertIsNone(probe.first_unloadable_library())
        self.assertEqual(loader.loaded[-2:], [os.path.join(folder, "libcudnn_engines_runtime_compiled.so.9"),
                                              os.path.join(folder, "libcudnn_heuristic.so.9")])

    def test_no_optional_preload_when_cudnn_came_from_the_system_search(self):
        loader = FakeLoader()
        probe_for("linux", [], loader).first_unloadable_library()
        self.assertEqual(loader.loaded, list(CUDA_LIBRARY_NAMES["linux"]))

    def test_linux_missing_cublas_is_named(self):
        probe = probe_for("linux", [], FakeLoader(broken={"libcublas.so.12"}))
        self.assertEqual(probe.first_unloadable_library(), "libcublas.so.12")

    def test_a_platform_without_nvidia_support_never_counts_as_loadable(self):
        self.assertIsNotNone(probe_for("darwin", [], FakeLoader()).first_unloadable_library())


class CandidateFolderTests(unittest.TestCase):
    def fake_spec(self, locations):
        return lambda name: (types.SimpleNamespace(submodule_search_locations=locations[name])
                             if name in locations else None)

    def test_nvidia_package_folders_then_torch_lib(self):
        locations = {"nvidia": [os.path.join("site", "nvidia")], "torch": [os.path.join("site", "torch")]}
        expected_windows = [
            os.path.join("site", "nvidia", "cublas", "bin"),
            os.path.join("site", "nvidia", "cudnn", "bin"),
            os.path.join("site", "torch", "lib"),
        ]
        folders = candidate_library_folders(
            "win32", find_spec=self.fake_spec(locations),
            list_folder=lambda folder: ["cudnn", "cublas"], is_folder=lambda folder: True,
        )
        self.assertEqual(folders, expected_windows)

    def test_linux_uses_the_lib_subfolder_and_drops_missing_folders(self):
        locations = {"nvidia": [os.path.join("site", "nvidia")]}
        wanted = os.path.join("site", "nvidia", "cublas", "lib")
        folders = candidate_library_folders(
            "linux", find_spec=self.fake_spec(locations),
            list_folder=lambda folder: ["cublas", "cuda_runtime"], is_folder=lambda folder: folder == wanted,
        )
        self.assertEqual(folders, [wanted])

    def test_nothing_installed_means_no_folders(self):
        self.assertEqual(candidate_library_folders("win32", find_spec=lambda name: None), [])


if __name__ == "__main__":
    unittest.main()
