"""Choosing where faster-whisper runs: an NVIDIA GPU when it truly works, else the CPU.

The choice also picks the model (through the runtime key): the Hebrew profile
uses full large-v3 on a GPU but the faster turbo model on the CPU.
"""

import importlib
import os
from dataclasses import dataclass

from engines.cuda_libraries import CudaLibraryProbe
from platform_hints import installer_command
from profiles import RUNTIME_CT2_CPU, RUNTIME_CT2_CUDA

# CTranslate2 uses 4 CPU threads unless told otherwise, which leaves most of a
# modern machine idle. Half the logical cores keeps it responsive meanwhile.
MINIMUM_CPU_THREADS = 4


@dataclass(frozen=True)
class DevicePlan:
    """Everything WhisperModel needs to know about where it runs."""

    device: str
    compute_type: str
    runtime_key: str
    cpu_threads: int

    def describe(self):
        if self.device == "cuda":
            return f"cuda ({self.compute_type})"
        return f"cpu ({self.compute_type}, {self.cpu_threads} threads)"


def cpu_thread_count(logical_cores):
    """Half the logical cores, never fewer than CTranslate2's own default."""
    return max(MINIMUM_CPU_THREADS, (logical_cores or 0) // 2)


def cuda_plan():
    return DevicePlan("cuda", "float16", RUNTIME_CT2_CUDA, 0)


def cpu_plan(logical_cores):
    return DevicePlan("cpu", "int8", RUNTIME_CT2_CPU, cpu_thread_count(logical_cores))


def gpu_fallback_message(missing_library):
    return (
        f"  An NVIDIA GPU was found but {missing_library} could not be loaded, "
        "so transcription runs on the CPU. To use the GPU, re-run "
        f"`{installer_command()} --gpu`."
    )


def choose_device(cuda_device_count, first_unloadable_library, logical_cores, report=print):
    """CUDA if a GPU exists and its libraries load, otherwise the CPU. Says which.

    The library check only runs when there is a GPU: on a machine without one
    the libraries are irrelevant and their absence is not worth a message.
    """
    plan = cpu_plan(logical_cores)
    if cuda_device_count() > 0:
        missing_library = first_unloadable_library()
        if missing_library is None:
            plan = cuda_plan()
        else:
            report(gpu_fallback_message(missing_library))
    report(f"  Transcription running on: {plan.describe()}")
    return plan


def detect_device_plan(probe=None, load_ctranslate2=None, logical_cores=None, report=print):
    """Inspect this machine and choose. The library folders are registered first,
    before CTranslate2 is imported, so nothing it loads misses them."""
    probe = probe or CudaLibraryProbe()
    probe.register_library_folders()
    ctranslate2 = (load_ctranslate2 or (lambda: importlib.import_module("ctranslate2")))()
    cores = os.cpu_count() if logical_cores is None else logical_cores
    return choose_device(
        ctranslate2.get_cuda_device_count, probe.first_unloadable_library, cores, report
    )
