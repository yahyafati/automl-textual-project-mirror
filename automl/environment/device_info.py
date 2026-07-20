from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cpuinfo as py_cpuinfo
import psutil

from automl.logger import get_logger

logger = get_logger()


def _run_cmd(cmd: list[str], timeout: float = 5.0) -> Optional[str]:
    """Run a command and return stripped stdout, or None on any failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()
        return None
    except (OSError, subprocess.SubprocessError):
        return None


def _which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def _bytes_to_gb(n: Optional[int]) -> Optional[float]:
    if n is None:
        return None
    return round(n / (1024**3), 2)


# ----------------------------------------------------------------------
# Section builders
# ----------------------------------------------------------------------


def _get_platform_info() -> dict[str, Any]:
    uname = platform.uname()
    info: dict[str, Any] = {
        "system": uname.system,  # "Linux", "Darwin", "Windows"
        "node_name": uname.node,
        "hostname": socket.gethostname(),
        "os_release": uname.release,
        "os_version": uname.version,
        "machine": uname.machine,  # e.g. "x86_64", "arm64", "AMD64"
        "architecture": platform.architecture()[0],  # "64bit" / "32bit"
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
    }

    if uname.system == "Darwin":
        info["macos_version"] = _run_cmd(["sw_vers", "-productVersion"])
        info["mac_model"] = _run_cmd(["sysctl", "-n", "hw.model"])
        # True on Apple Silicon, False on Intel Macs
        info["apple_silicon"] = uname.machine == "arm64"

    if uname.system == "Linux":
        try:
            distro_lines = {}
            os_release_path = Path("/etc/os-release")
            if os_release_path.exists():
                for line in os_release_path.read_text().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        distro_lines[k] = v.strip('"')
                info["linux_distro"] = distro_lines.get("PRETTY_NAME")
        except Exception:
            info["linux_distro"] = None

    if uname.system == "Windows":
        info["windows_edition"] = _run_cmd(["wmic", "os", "get", "Caption", "/value"])

    return info


def _get_cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "processor_raw": platform.processor(),
        "physical_cores": None,
        "logical_cores": None,
        "max_frequency_mhz": None,
        "current_frequency_mhz": None,
        "cpu_usage_percent": None,
        "brand": None,
    }

    try:
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        if freq:
            info["max_frequency_mhz"] = freq.max or None
            info["current_frequency_mhz"] = freq.current or None
        # Short non-blocking-ish sample; interval=0.2 keeps it fast
        info["cpu_usage_percent"] = psutil.cpu_percent(interval=0.2)
    except Exception as e:
        info["psutil_error"] = str(e)

    # py-cpuinfo gives a clean human-readable brand string on all 3 OSes

    try:
        cpu = py_cpuinfo.get_cpu_info()
        info["brand"] = cpu.get("brand_raw")
        info["arch"] = cpu.get("arch")
        info["bits"] = cpu.get("bits")
        info["hz_advertised_ghz"] = cpu.get("hz_advertised_friendly") or None
    except Exception as e:
        info["py_cpuinfo_error"] = str(e)

    # Fallbacks for a clean brand name if py-cpuinfo isn't installed
    if not info["brand"]:
        system = platform.system()
        if system == "Darwin":
            info["brand"] = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        elif system == "Linux":
            try:
                cpuinfo_path = Path("/proc/cpuinfo")
                if cpuinfo_path.exists():
                    for line in cpuinfo_path.read_text().splitlines():
                        if line.lower().startswith("model name"):
                            info["brand"] = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass
        elif system == "Windows":
            info["brand"] = _run_cmd(["wmic", "cpu", "get", "name"])

    return info


def _get_ram_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            info["total_gb"] = _bytes_to_gb(vm.total)
            info["available_gb"] = _bytes_to_gb(vm.available)
            info["used_gb"] = _bytes_to_gb(vm.used)
            info["percent_used"] = vm.percent
            swap = psutil.swap_memory()
            info["swap_total_gb"] = _bytes_to_gb(swap.total)
            info["swap_used_gb"] = _bytes_to_gb(swap.used)
        except Exception as e:
            info["error"] = str(e)
    else:
        info["error"] = "psutil not installed — run `pip install psutil` for RAM info"
    return info


def _get_disk_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    if psutil is not None:
        try:
            usage = psutil.disk_usage(str(Path.home()))
            info["home_partition_total_gb"] = _bytes_to_gb(usage.total)
            info["home_partition_free_gb"] = _bytes_to_gb(usage.free)
            info["home_partition_used_percent"] = usage.percent
        except Exception as e:
            info["error"] = str(e)
    return info


def _get_nvidia_gpu_info() -> Optional[list[dict[str, Any]]]:
    """Try pynvml first (structured), fall back to parsing nvidia-smi CSV."""
    # --- Try pynvml / nvidia-ml-py ---
    try:
        import pynvml

        pynvml.nvmlInit()
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                util = None
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "memory_total_gb": _bytes_to_gb(mem.total),
                    "memory_used_gb": _bytes_to_gb(mem.used),
                    "utilization_percent": util,
                    "driver_version": (
                        pynvml.nvmlSystemGetDriverVersion()
                        if hasattr(pynvml, "nvmlSystemGetDriverVersion")
                        else None
                    ),
                }
            )
        pynvml.nvmlShutdown()
        if gpus:
            return gpus
    except Exception:
        pass  # pynvml not installed, or no NVIDIA driver — fall through

    # --- Fall back to nvidia-smi CLI ---
    if _which("nvidia-smi") is None:
        return None

    out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, mem_total, mem_used, util, driver = parts[:6]
        try:
            gpus.append(
                {
                    "index": int(idx),
                    "name": name,
                    "memory_total_gb": round(float(mem_total) / 1024, 2),
                    "memory_used_gb": round(float(mem_used) / 1024, 2),
                    "utilization_percent": float(util),
                    "driver_version": driver,
                }
            )
        except ValueError:
            continue
    return gpus or None


def _get_amd_gpu_info() -> Optional[str]:
    """Best-effort AMD GPU detection via rocm-smi, if present."""
    if _which("rocm-smi") is None:
        return None
    return _run_cmd(["rocm-smi", "--showproductname"])


def _get_apple_gpu_info() -> Optional[dict[str, Any]]:
    """macOS GPU / Apple Silicon info via system_profiler."""
    if platform.system() != "Darwin":
        return None
    out = _run_cmd(["system_profiler", "SPDisplaysDataType", "-json"])
    if not out:
        return None
    try:
        data = json.loads(out)
        displays = data.get("SPDisplaysDataType", [])
        return displays or None
    except Exception:
        return {"raw": out}


def _get_windows_gpu_info() -> Optional[str]:
    if platform.system() != "Windows":
        return None
    return _run_cmd(
        [
            "powershell",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | Format-List",
        ]
    ) or _run_cmd(["wmic", "path", "win32_VideoController", "get", "name"])


def _get_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "nvidia": _get_nvidia_gpu_info(),
        "amd": _get_amd_gpu_info(),
        "apple": _get_apple_gpu_info(),
        "windows_wmi": _get_windows_gpu_info(),
    }
    # Trim Nones for readability
    return {k: v for k, v in info.items() if v is not None}


def _get_torch_info() -> dict[str, Any]:
    """Deep-learning-framework view of accelerators (most useful part for a
    training benchmark: this is what your model actually sees)."""
    info: dict[str, Any] = {"available": False}
    try:
        import torch  # type: ignore

        info["available"] = True
        info["version"] = torch.__version__

        # CUDA (NVIDIA / some AMD via ROCm-as-CUDA-API)
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            )
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["cuda_devices"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": _bytes_to_gb(props.total_memory),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count,
                    }
                )

        # Apple Metal Performance Shaders
        mps_backend = getattr(torch.backends, "mps", None)
        info["mps_built"] = bool(mps_backend and mps_backend.is_built())
        info["mps_available"] = bool(mps_backend and mps_backend.is_available())

        info["default_device"] = (
            "cuda"
            if info["cuda_available"]
            else "mps" if info["mps_available"] else "cpu"
        )
    except ImportError:
        info["note"] = "torch not installed — install it to see CUDA/MPS/device details"
    except Exception as e:
        info["error"] = str(e)
    return info


def _get_relevant_package_versions() -> dict[str, Optional[str]]:
    """Versions of common ML packages, useful for reproducibility metadata
    in a benchmark. Only reports packages that are actually installed."""
    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "pandas",
        "transformers",
        "accelerate",
        "datasets",
        "xformers",
        "flash-attn",
        "triton",
        "deepspeed",
        "bitsandbytes",
    ]
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover (py<3.8)
        return {}

    versions: dict[str, Optional[str]] = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return versions


def _get_relevant_env_vars() -> dict[str, Optional[str]]:
    """Environment variables that commonly affect ML training/inference
    behavior and reproducibility."""
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "NCCL_DEBUG",
        "CUDA_HOME",
    ]
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}


def get_device_info() -> dict[str, Any]:
    """Collect a full snapshot of device/software info. Never raises —
    any section that fails records its own error instead of aborting
    the whole collection."""

    logger.debug("Starting device and software info collection...")
    sections = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": _get_platform_info,
        "cpu": _get_cpu_info,
        "ram": _get_ram_info,
        "disk": _get_disk_info,
        "gpu": _get_gpu_info,
        "torch": _get_torch_info,
        "package_versions": _get_relevant_package_versions,
        "relevant_env_vars": _get_relevant_env_vars,
    }

    info: dict[str, Any] = {}
    for key, value in sections.items():
        if callable(value):
            try:
                info[key] = value()
            except Exception as e:
                logger.warning(f"Failed to collect section '{key}': {e}", exc_info=True)
                info[key] = {"error": f"failed to collect: {e}"}
        else:
            info[key] = value

    return info


def save_device_info(
    info: Optional[dict[str, Any]] = None,
    path: str | Path = "device_info.json",
    pretty: bool = True,
) -> Path:
    """Collect (if not already collected) and save device info as JSON.

    Args:
        info: pre-collected info dict; if None, get_device_info() is called.
        path: output file path. Parent directories are created if needed.
        pretty: pretty-print the JSON (indent=2) if True.

    Returns:
        The resolved Path the file was written to.
    """
    if info is None:
        info = get_device_info()

    out_path = Path(path).expanduser().resolve()
    logger.debug(f"Preparing to save device info to target path: {out_path}")

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            if pretty:
                json.dump(info, f, indent=2, default=str)
            else:
                json.dump(info, f, default=str)

        logger.info(f"Successfully saved device info to {out_path}")
    except Exception as e:
        # If writing the file itself fails, we definitely want to log it and reraise
        logger.error(f"Failed to write device info to {out_path}: {e}", exc_info=True)
        raise

    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        default="device_info.json",
        help="Output JSON file path (default: ./device_info.json)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Don't print the collected info to stdout",
    )
    args = parser.parse_args()

    info = get_device_info()
    out_path = save_device_info(info, args.output)

    if not args.quiet:
        print(json.dumps(info, indent=2, default=str))
    print(f"\nSaved device info to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    _main()
