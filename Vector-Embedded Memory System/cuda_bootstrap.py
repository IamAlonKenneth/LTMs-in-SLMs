"""
cuda_bootstrap.py
=================
One-shot CUDA driver pre-initialisation for WSL2 GPU passthrough.

Issue: `import torch` hangs in WSL2 when it tries to call `cuInit` for the
first time via its CUDA extension loader. Pre-calling `cuInit(0)` through
the ctypes interface before any torch import bypasses the deadlock.

Import this module BEFORE any import of torch, torchvision, bitsandbytes,
or any other CUDA-dependent library:

    import cuda_bootstrap   # must be first CUDA-related import
    import torch
    ...
"""

import ctypes
import os
import sys
import platform

# Platform detection
_IS_WINDOWS = platform.system() == "Windows"

# WSL2 / Linux: pre-initialise the CUDA driver via ctypes to prevent torch import hang
# Windows: CUDA loads natively through the NVIDIA driver — no bootstrap needed

if _IS_WINDOWS:
    # Windows NVIDIA driver loads automatically with torch import.
    # Nothing to do — mark as OK and return.
    os.environ["CUDA_BOOTSTRAP_STATUS"] = "ok:windows_native"
else:
    # Linux (WSL2 or bare-metal): ensure WSL GPU bridge is findable
    _wsl_lib_path = "/usr/lib/wsl/lib"
    if os.path.isdir(_wsl_lib_path) and _wsl_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{_wsl_lib_path}:{existing}" if existing else _wsl_lib_path

    # Pre-initialise the CUDA driver via ctypes
    try:
        libcuda = ctypes.CDLL("libcuda.so.1", use_errno=True)
        cuInit = libcuda.cuInit
        cuInit.restype = int
        result = cuInit(0)
        if result == 0:
            count = ctypes.c_int()
            cuDeviceGetCount = libcuda.cuDeviceGetCount
            cuDeviceGetCount.restype = int
            cuDeviceGetCount(ctypes.byref(count))
            if count.value > 0:
                name = ctypes.create_string_buffer(256)
                cuDeviceGetName = libcuda.cuDeviceGetName
                cuDeviceGetName.restype = int
                cuDeviceGetName(name, 256, ctypes.c_int(0))
                device_name = name.value.decode()
                os.environ["CUDA_BOOTSTRAP_STATUS"] = f"ok:{device_name}"
            else:
                os.environ["CUDA_BOOTSTRAP_STATUS"] = "ok:0_devices"
        else:
            os.environ["CUDA_BOOTSTRAP_STATUS"] = f"cuInit_failed:{result}"
    except OSError as exc:
        os.environ["CUDA_BOOTSTRAP_STATUS"] = f"libcuda_not_found:{exc}"
