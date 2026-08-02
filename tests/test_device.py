"""Tests for autodub.device module."""

from autodub.device import (
    resolve_device_selection,
    preferred_whisper_compute_type,
    whisper_compute_type_candidates,
    is_cuda_runtime_error,
    cpu_fallback_whisper_model,
)


class TestResolveDeviceSelection:
    def test_rocm_returns_cuda(self):
        assert resolve_device_selection("rocm") == "cuda"

    def test_explicit_cpu(self):
        assert resolve_device_selection("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert resolve_device_selection("cuda") == "cuda"

    def test_auto_returns_cpu_or_cuda(self):
        result = resolve_device_selection("auto")
        assert result in ("cpu", "cuda")


class TestPreferredWhisperComputeType:
    def test_large_gpu(self):
        result = preferred_whisper_compute_type("large-v3", "cuda")
        assert result in ("float16", "int8_float16")

    def test_small_cpu(self):
        result = preferred_whisper_compute_type("small", "cpu")
        assert result == "int8"


class TestWhisperComputeTypeCandidates:
    def test_gpu_candidates(self):
        result = whisper_compute_type_candidates("small", "cuda")
        assert len(result) >= 2
        assert "float16" in result or "int8_float16" in result

    def test_cpu_candidates(self):
        result = whisper_compute_type_candidates("small", "cpu")
        assert len(result) >= 2


class TestIsCudaRuntimeError:
    def test_cuda_error(self):
        assert is_cuda_runtime_error(Exception("cublas64 not found"))

    def test_rocm_error(self):
        assert is_cuda_runtime_error(Exception("hipblas not found"))

    def test_normal_error(self):
        assert not is_cuda_runtime_error(Exception("some other error"))


class TestCpuFallbackWhisperModel:
    def test_large_fallback(self):
        assert cpu_fallback_whisper_model("large-v3") == "small"

    def test_medium_fallback(self):
        assert cpu_fallback_whisper_model("medium") == "small"

    def test_small_keeps(self):
        assert cpu_fallback_whisper_model("small") == "small"

    def test_base_keeps(self):
        assert cpu_fallback_whisper_model("base") == "base"