"""Read-only, standard-library diagnostics. Never outputs credential values."""
import ctypes
import importlib.metadata as md
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import winreg

def attempt(fn):
    try:
        return fn()
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}

def registry_values(key, names):
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
        return {name: attempt(lambda: winreg.QueryValueEx(handle, name)[0]) for name in names}

class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical", "total_pagefile", "available_pagefile", "total_virtual", "available_virtual", "available_extended_virtual")]

def memory():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return {name: getattr(status, name) for name, _ in status._fields_}

def gpu_registry():
    result = []
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as handle:
        for i in range(winreg.QueryInfoKey(handle)[0]):
            child = winreg.EnumKey(handle, i)
            if child.isdigit():
                result.append(registry_values(base + "\\" + child, ["DriverDesc", "DriverVersion", "HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"]))
    return result

packages = ["torch", "torchvision", "opencv-python", "opencv-python-headless", "kornia", "pycolmap", "faiss-cpu", "faiss-gpu", "transformers", "kaggle", "numpy"]
result = {
    "command": "python scripts/audit_environment.py",
    "platform": platform.platform(), "architecture": platform.machine(),
    "python": sys.version, "executable": sys.executable,
    "logical_cores": os.cpu_count(),
    "cpu_registry": attempt(lambda: registry_values(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", ["ProcessorNameString", "~MHz"])),
    "os_registry": attempt(lambda: registry_values(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", ["ProductName", "DisplayVersion", "CurrentBuild", "UBR"])),
    "memory_bytes": attempt(memory),
    "disk_bytes": {drive: attempt(lambda d=drive: shutil.disk_usage(d)._asdict()) for drive in ["C:\\", "D:\\"]},
    "gpu_registry": attempt(gpu_registry),
    "packages": {name: attempt(lambda n=name: md.version(n)) for name in packages},
    "commands": {name: shutil.which(name) for name in ["python", "pip", "uv", "conda", "mamba", "git", "nvidia-smi", "nvcc", "colmap", "kaggle", "docker", "wsl"]},
    "environment_markers_present_only": {key: key in os.environ for key in ["KAGGLE_KERNEL_RUN_TYPE", "COLAB_RELEASE_TAG", "WSL_DISTRO_NAME", "CONTAINER_SANDBOX_MOUNT_POINT", "CUDA_PATH"]},
    "container_files": {p: pathlib.Path(p).exists() for p in ["/.dockerenv", "/run/.containerenv"]},
    "kaggle_auth_environment_present_only": {key: bool(os.environ.get(key)) for key in ["KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "KAGGLE_CONFIG_DIR"]},
}
config = pathlib.Path(os.environ.get("KAGGLE_CONFIG_DIR", str(pathlib.Path.home() / ".kaggle")))
result["kaggle_credential_files_present_only"] = {name: (config / name).is_file() for name in ["kaggle.json", "access_token"]}
result["imports"] = {}
for module in ["torch", "torchvision", "cv2", "kornia", "pycolmap", "faiss", "transformers"]:
    command = [sys.executable, "-c", f"import {module} as m; print(getattr(m, '__version__', 'version attribute unavailable'))" + ("; print('torch CUDA build:', m.version.cuda); print('CUDA available:', m.cuda.is_available()); print('CUDA device count:', m.cuda.device_count())" if module == "torch" else "")]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=8)
        result["imports"][module] = {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired:
        result["imports"][module] = {"error": "timeout after 8 seconds"}
print(json.dumps(result, indent=2, default=lambda value: {"registry_binary_hex": value.hex()} if isinstance(value, bytes) else str(value)))
