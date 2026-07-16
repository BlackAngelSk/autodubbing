"""GPU detection, CUDA/ROCm device selection, and Whisper compute type helpers."""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

from autodub.config import LARGE_WHISPER_MODELS

logger = logging.getLogger(__name__)


class _HFUnauthenticatedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "unauthenticated requests to the hf hub" not in record.getMessage().lower()


def configure_hf_hub_access(hf_token: str | None = None) -> bool:
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*unauthenticated requests to the HF Hub.*",
        category=UserWarning,
    )

    for logger_name in (
        "",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "huggingface_hub.utils._http",
        "huggingface_hub.utils._validators",
    ):
        logger = logging.getLogger(logger_name)
        if not any(isinstance(existing, _HFUnauthenticatedFilter) for existing in logger.filters):
            logger.addFilter(_HFUnauthenticatedFilter())

    token = (hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if not token:
        return False

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return True


def detect_cuda_available() -> bool:
    try:
        ctranslate2 = importlib.import_module("ctranslate2")
        get_cuda_device_count = getattr(ctranslate2, "get_cuda_device_count", None)
        if callable(get_cuda_device_count):
            device_count = get_cuda_device_count()
            if isinstance(device_count, int):
                return device_count > 0
    except Exception:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False

    try:
        completed = subprocess.run(
            [nvidia_smi, "-L"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False

    return bool(completed.stdout.strip())


def detect_rocm_available() -> bool:
    """Detect AMD ROCm GPU availability via rocm-smi, rocminfo, or the KFD kernel device node."""
    kfd_exists = Path("/dev/kfd").exists()
    rocm_smi = shutil.which("rocm-smi")
    rocminfo = shutil.which("rocminfo")
    if rocm_smi is None and rocminfo is None and not kfd_exists:
        return False

    if rocm_smi is not None:
        try:
            completed = subprocess.run(
                [rocm_smi, "--showid"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            if completed.stdout.strip():
                return True
        except (subprocess.SubprocessError, OSError):
            pass

    if rocminfo is not None:
        try:
            completed = subprocess.run(
                [rocminfo],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            if "agent" in completed.stdout.lower():
                return True
        except (subprocess.SubprocessError, OSError):
            pass

    return kfd_exists


def resolve_device_selection(device: str) -> str:
    if device == "rocm":
        return "cuda"
    if device != "auto":
        return device
    if detect_cuda_available():
        return "cuda"
    if detect_rocm_available():
        return "cuda"
    return "cpu"


def preferred_whisper_compute_type(model_name: str, device: str) -> str:
    if device == "cuda":
        return "float16" if model_name in LARGE_WHISPER_MODELS else "int8_float16"
    return "int8"


def whisper_compute_type_candidates(model_name: str, device: str) -> list[str]:
    preferred = preferred_whisper_compute_type(model_name, device)
    if device == "cuda":
        candidates = [preferred, "float16", "int8_float16", "int8"]
    else:
        candidates = [preferred, "int8", "float32"]
    return list(dict.fromkeys(candidates))


def is_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "cublas", "cublas64", "cudnn", "cudart", "nvcuda",
        "cuda driver", "cannot be loaded", "failed to load",
        "hipblas", "libamdhip64", "amdhip", "rocblas", "hip error",
    )
    return any(marker in message for marker in markers)


def cpu_fallback_whisper_model(model_name: str) -> str:
    """Pick a faster model when CUDA is unavailable and we must run on CPU."""
    normalized = model_name.strip().lower()
    if normalized in LARGE_WHISPER_MODELS:
        return "small"
    if normalized == "medium":
        return "small"
    return model_name