#!/usr/bin/env python3
"""Auto-dub a video by transcribing, translating, and re-synthesizing speech."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, cast

from deep_translator import GoogleTranslator, MyMemoryTranslator
from faster_whisper import WhisperModel
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from tqdm import tqdm


DEFAULT_EDGE_VOICES: dict[str, str] = {
    "en": "en-US-AriaNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "ja": "ja-JP-NanamiNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "sk": "sk-SK-ViktoriaNeural",
}

LARGE_WHISPER_MODELS = {
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v2",
    "distil-large-v3",
}

TRANSLATION_PROVIDERS = {"google", "mymemory"}
ASR_ENGINE_CHOICES = {"auto", "whisper", "stable-ts"}
HF_UNAUTH_WARNING_TEXT = "unauthenticated requests to the hf hub"


class _HFUnauthenticatedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return HF_UNAUTH_WARNING_TEXT not in record.getMessage().lower()


def configure_hf_hub_access(hf_token: str | None = None) -> bool:
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


def configure_windows_asyncio_policy() -> None:
    """Use selector policy on Windows to avoid noisy Proactor transport shutdown errors."""
    if os.name != "nt" or not hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        return

    current_policy = asyncio.get_event_loop_policy()
    if not isinstance(current_policy, asyncio.WindowsSelectorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def suppress_windows_proactor_connection_reset_noise() -> None:
    """Suppress known benign WinError 10054 noise from Proactor shutdown callbacks."""
    if os.name != "nt":
        return

    if getattr(asyncio, "_autodub_proactor_reset_patch", False):
        return

    original_handler = asyncio.BaseEventLoop.default_exception_handler

    def patched_default_exception_handler(self: asyncio.BaseEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        handle = context.get("handle")
        callback_text = repr(handle) if handle is not None else str(context.get("message", ""))
        if (
            isinstance(exc, ConnectionResetError)
            and "winerror 10054" in str(exc).lower()
            and "_proactorbasepipetransport._call_connection_lost" in callback_text.lower()
        ):
            return
        original_handler(self, context)

    asyncio.BaseEventLoop.default_exception_handler = patched_default_exception_handler
    setattr(asyncio, "_autodub_proactor_reset_patch", True)


configure_windows_asyncio_policy()
suppress_windows_proactor_connection_reset_noise()


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
        # AMD ROCm uses ctranslate2's CUDA backend via the HIP compatibility layer.
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
        # Different NVIDIA generations and driver/runtime combos can prefer
        # different compute modes, so try a short fallback chain on GPU first.
        candidates = [preferred, "float16", "int8_float16", "int8"]
    else:
        candidates = [preferred, "int8", "float32"]

    return list(dict.fromkeys(candidates))


def is_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        # NVIDIA CUDA
        "cublas",
        "cublas64",
        "cudnn",
        "cudart",
        "nvcuda",
        "cuda driver",
        "cannot be loaded",
        "failed to load",
        # AMD ROCm / HIP
        "hipblas",
        "libamdhip64",
        "amdhip",
        "rocblas",
        "hip error",
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


def load_whisper_model(
    model_name: str,
    device: str,
    hf_token: str | None = None,
    cpu_fallback_model: str | None = None,
) -> WhisperModel:
    has_hf_token = configure_hf_hub_access(hf_token)
    attempted_compute_types = whisper_compute_type_candidates(model_name, device)
    last_error: Exception | None = None

    for compute_type in attempted_compute_types:
        try:
            return WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as exc:
            last_error = exc
            continue

    if device == "cuda" and last_error is not None and is_cuda_runtime_error(last_error):
        fallback_model = cpu_fallback_model or cpu_fallback_whisper_model(model_name)
        cpu_compute_candidates = whisper_compute_type_candidates(fallback_model, "cpu")
        logging.info(
            "CUDA runtime libraries are unavailable (%s). Falling back to CPU for Whisper model '%s'.",
            last_error,
            fallback_model,
        )
        for cpu_compute_type in cpu_compute_candidates:
            try:
                return WhisperModel(fallback_model, device="cpu", compute_type=cpu_compute_type)
            except Exception as exc:
                last_error = exc
                continue

    hint = (
        "The first run may need to download the model, so check network access and free disk space. "
        "Set HF_TOKEN for higher rate limits if needed."
        if model_name in LARGE_WHISPER_MODELS
        else "Try a smaller model such as 'base' or 'small'."
    )
    auth_hint = "" if has_hf_token else " You can also set HF_TOKEN to reduce Hub rate-limit issues."
    attempted = ", ".join(attempted_compute_types)
    raise RuntimeError(
        f"Unable to load Whisper model '{model_name}' on '{device}'. "
        f"Tried compute types: {attempted}. {hint}{auth_hint} Original error: {last_error}"
    ) from last_error


@dataclass
class Segment:
    start_s: float
    end_s: float
    source_text: str
    translated_text: str = ""


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: Iterable[Segment], srt_path: Path) -> None:
    lines: list[str] = []
    for index, seg in enumerate(segments, start=1):
        subtitle_text = (safe_text(seg.translated_text) or safe_text(seg.source_text)).strip()
        if not subtitle_text:
            continue
        lines.extend(
            [
                str(index),
                f"{format_srt_timestamp(seg.start_s)} --> {format_srt_timestamp(seg.end_s)}",
                subtitle_text,
                "",
            ]
        )
    srt_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def normalize_subtitle_for_dedupe(text: str) -> str:
    normalized = re.sub(r"\s+", " ", safe_text(text)).strip().lower()
    return re.sub(r"[^\w\s]", "", normalized)


def collapse_consecutive_duplicate_segments(segments: List[Segment], max_gap_s: float = 0.35) -> List[Segment]:
    if not segments:
        return segments

    ordered = sorted(segments, key=lambda seg: seg.start_s)
    merged: List[Segment] = [ordered[0]]

    for seg in ordered[1:]:
        prev = merged[-1]
        prev_text = normalize_subtitle_for_dedupe(prev.translated_text or prev.source_text)
        current_text = normalize_subtitle_for_dedupe(seg.translated_text or seg.source_text)

        is_duplicate_text = bool(prev_text) and prev_text == current_text
        close_enough = seg.start_s <= prev.end_s + max_gap_s

        if is_duplicate_text and close_enough:
            prev.end_s = max(prev.end_s, seg.end_s)
            if len(safe_text(seg.translated_text).strip()) > len(safe_text(prev.translated_text).strip()):
                prev.translated_text = safe_text(seg.translated_text)
            if len(safe_text(seg.source_text).strip()) > len(safe_text(prev.source_text).strip()):
                prev.source_text = safe_text(seg.source_text)
            continue

        merged.append(seg)

    return merged


def cached_translation_looks_poor(segments: Iterable[Segment], target_lang: str) -> bool:
    if target_lang == "en":
        return False

    items = list(segments)
    if not items:
        return False

    unchanged = 0
    empty = 0
    for seg in items:
        source_clean = re.sub(r"\s+", " ", safe_text(seg.source_text)).strip().lower()
        translated_clean = re.sub(r"\s+", " ", safe_text(seg.translated_text)).strip().lower()
        if not translated_clean:
            empty += 1
            continue
        if source_clean and source_clean == translated_clean:
            unchanged += 1

    total = len(items)
    unchanged_ratio = unchanged / total
    empty_ratio = empty / total
    return unchanged_ratio > 0.35 or empty_ratio > 0.12


def translation_looks_wrong_language(segments: Iterable[Segment], target_lang: str) -> bool:
    if target_lang == "en":
        return False

    stopword_hints: dict[str, set[str]] = {
        "es": {"el", "la", "los", "las", "que", "de", "por", "para", "con", "una", "un", "como"},
        "fr": {"le", "la", "les", "des", "une", "que", "pour", "avec", "pas", "est", "dans"},
        "de": {"der", "die", "das", "und", "nicht", "mit", "ist", "für", "ein", "eine", "ich"},
        "pt": {"de", "do", "da", "que", "para", "com", "não", "uma", "um", "como", "você"},
        "sk": {"som", "si", "je", "sme", "ste", "sa", "že", "ako", "čo", "pre", "to", "nie"},
        "ru": {"и", "в", "не", "на", "что", "это", "как", "для", "с", "я", "ты"},
        "hi": {"है", "और", "नहीं", "के", "यह", "से", "मैं", "आप", "हम", "क्या"},
        "ja": {"です", "ます", "して", "ない", "する", "これ", "それ", "から", "まで", "よう"},
    }
    script_patterns: dict[str, str] = {
        "ru": r"[А-Яа-яЁё]",
        "hi": r"[\u0900-\u097F]",
        "ja": r"[\u3040-\u30FF\u4E00-\u9FFF]",
    }
    english_hints = {
        "the", "and", "you", "that", "this", "with", "for", "not", "are", "was", "have", "will", "what",
        "your", "from", "they", "can", "about", "just", "like", "there",
    }

    target_hints = stopword_hints.get(target_lang, set())
    script_pattern = script_patterns.get(target_lang)

    items = [seg for seg in segments if safe_text(seg.translated_text).strip()]
    if not items:
        return True

    checked = 0
    unchanged = 0
    english_like = 0
    target_like = 0

    for seg in items:
        source_clean = re.sub(r"\s+", " ", safe_text(seg.source_text)).strip().lower()
        text = safe_text(seg.translated_text).strip().lower()
        if len(text) < 8:
            continue

        checked += 1
        if source_clean and source_clean == text:
            unchanged += 1

        target_match = False
        if script_pattern is not None and re.search(script_pattern, text):
            target_match = True

        latin_tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ']+", text)
        en_hits = sum(1 for token in latin_tokens if token in english_hints)
        target_hits = sum(1 for token in latin_tokens if token in target_hints)

        if target_hits >= 1:
            target_match = True

        if target_match:
            target_like += 1
        if en_hits >= 2:
            english_like += 1

    if checked <= 0:
        return True

    unchanged_ratio = unchanged / checked
    english_ratio = english_like / checked
    target_ratio = target_like / checked

    if unchanged_ratio >= 0.55:
        return True

    if script_pattern is not None:
        return target_ratio < 0.30

    if target_hints:
        if target_ratio >= 0.28:
            return False
        return english_ratio >= 0.50 or unchanged_ratio >= 0.35

    return unchanged_ratio >= 0.45


def build_translator(provider: str, source: str, target: str) -> Any:
    normalized = provider.strip().lower() if provider else "google"
    if normalized not in TRANSLATION_PROVIDERS:
        raise ValueError(f"Unsupported translation provider: {provider}")
    if normalized == "mymemory":
        try:
            return MyMemoryTranslator(source=source, target=target)
        except Exception:
            return GoogleTranslator(source=source, target=target)
    return GoogleTranslator(source=source, target=target)


def parse_glossary_overrides(glossary_text: str | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not glossary_text:
        return overrides

    for raw_line in glossary_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        separator = "=>" if "=>" in line else "->" if "->" in line else "=" if "=" in line else None
        if separator is None:
            continue

        source, replacement = (part.strip() for part in line.split(separator, 1))
        if source and replacement:
            overrides[source.lower()] = replacement

    return overrides


def apply_glossary_overrides(text: str, overrides: dict[str, str]) -> str:
    adjusted = text
    for source, replacement in sorted(overrides.items(), key=lambda item: len(item[0]), reverse=True):
        escaped = re.escape(source)
        pattern = escaped if " " in source else rf"\b{escaped}\b"
        adjusted = re.sub(pattern, replacement, adjusted, flags=re.IGNORECASE)
    return adjusted


def save_segments_to_json(segments: Iterable[Segment], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(seg) for seg in segments], indent=2), encoding="utf-8")


def load_segments_from_json(input_path: Path) -> List[Segment]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    return [
        Segment(
            start_s=float(item["start_s"]),
            end_s=float(item["end_s"]),
            source_text=safe_text(item.get("source_text", "")),
            translated_text=safe_text(item.get("translated_text", "")),
        )
        for item in raw
    ]


def build_dub_cache_signature(
    segments: Iterable[Segment],
    target_lang: str,
    tts_engine: str,
    edge_voice: str | None,
    min_stretch_speed: float,
    max_stretch_speed: float,
    silence_trim_ms: int,
) -> str:
    payload = {
        "target_lang": target_lang,
        "tts_engine": tts_engine,
        "edge_voice": edge_voice or "",
        "min_stretch_speed": round(min_stretch_speed, 4),
        "max_stretch_speed": round(max_stretch_speed, 4),
        "silence_trim_ms": int(silence_trim_ms),
        "segments": [asdict(seg) for seg in segments],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def build_resume_dir(
    input_path: Path,
    output_dir: Path,
    target_lang: str,
    whisper_model: str,
    translation_provider: str,
    tts_engine: str,
    edge_voice: str | None,
    optimization_profile: str,
    start_time_s: float,
    end_time_s: float | None,
    glossary_text: str,
    asr_engine: str = "whisper",
) -> Path:
    job_signature = "|".join(
        [
            str(input_path.resolve()),
            target_lang,
            whisper_model,
            asr_engine,
            translation_provider,
            tts_engine,
            edge_voice or "",
            optimization_profile,
            f"{start_time_s:.3f}",
            "none" if end_time_s is None else f"{end_time_s:.3f}",
            hashlib.sha1((glossary_text or "").encode("utf-8")).hexdigest()[:10],
        ]
    )
    short_hash = hashlib.sha1(job_signature.encode("utf-8")).hexdigest()[:12]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", input_path.stem)[:40] or "video"
    return output_dir / ".autodub_resume" / f"{safe_stem}_{target_lang}_{short_hash}"


def run_cmd(cmd: list[str]) -> None:
    """Run a subprocess command and fail with readable output."""
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        missing = cmd[0]
        raise RuntimeError(f"Required executable not found: {missing}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        ) from exc


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH")


def probe_media_duration(media_path: Path) -> float | None:
    """Best-effort media duration probe used for auto optimization decisions."""
    if shutil.which("ffprobe") is None:
        return None

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    value = completed.stdout.strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def resolve_processing_profile(
    selected_profile: str,
    clip_duration_s: float | None,
    whisper_model: str,
    device: str,
    min_stretch_speed: float,
    max_stretch_speed: float,
    silence_trim_ms: int,
) -> dict[str, Any]:
    applied_profile = selected_profile
    if selected_profile == "auto":
        if clip_duration_s is not None and clip_duration_s <= 150:
            applied_profile = "short"
        elif clip_duration_s is not None and clip_duration_s >= 12 * 60:
            applied_profile = "long"
        else:
            applied_profile = "balanced"

    resolved_device = resolve_device_selection(device)
    resolved_model = whisper_model
    resolved_min = min_stretch_speed
    resolved_max = max_stretch_speed
    resolved_trim = silence_trim_ms
    transcribe_chunk_s: float | None = None
    tts_chunk_window_s: float | None = 75.0

    if applied_profile == "short":
        resolved_min = max(resolved_min, 0.90)
        resolved_max = min(max(resolved_max, 1.45), 1.65)
        resolved_trim = max(resolved_trim, 18)
        tts_chunk_window_s = 60.0
    elif applied_profile == "long":
        if resolved_device != "cuda" and whisper_model in ({"small", "medium"} | LARGE_WHISPER_MODELS):
            resolved_model = "base"
        resolved_min = min(resolved_min, 0.92)
        resolved_max = max(resolved_max, 1.95)
        resolved_trim = max(resolved_trim, 12)
        transcribe_chunk_s = 420.0
        tts_chunk_window_s = 120.0
    else:
        transcribe_chunk_s = 300.0 if clip_duration_s is not None and clip_duration_s >= 8 * 60 else None
        tts_chunk_window_s = 90.0

    return {
        "label": applied_profile,
        "device": resolved_device,
        "whisper_model": resolved_model,
        "min_stretch_speed": resolved_min,
        "max_stretch_speed": resolved_max,
        "silence_trim_ms": resolved_trim,
        "clip_duration_s": clip_duration_s,
        "transcribe_chunk_s": transcribe_chunk_s,
        "tts_chunk_window_s": tts_chunk_window_s,
    }


def extract_audio(video_path: Path, audio_out: Path) -> None:
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_out),
        ]
    )


def trim_video(input_video: Path, output_video: Path, start_time_s: float, end_time_s: float | None) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(input_video), "-ss", f"{start_time_s}"]
    if end_time_s is not None:
        cmd.extend(["-to", f"{end_time_s}"])
    cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(output_video)])
    run_cmd(cmd)


def stable_ts_available() -> bool:
    """Return True if stable-ts is installed and importable."""
    try:
        importlib.import_module("stable_whisper")
        return True
    except ImportError:
        return False


def resolve_asr_engine(asr_engine: str | None, status_callback: Callable[[str], None] | None = None) -> str:
    normalized = safe_text(asr_engine).strip().lower() or "auto"
    if normalized not in ASR_ENGINE_CHOICES:
        raise ValueError(f"Unsupported ASR engine: {asr_engine}")

    if normalized != "auto":
        return normalized

    if stable_ts_available():
        if status_callback is not None:
            status_callback("[asr] Auto-selected stable-ts for stronger speech detection and timestamp alignment.")
        return "stable-ts"

    if status_callback is not None:
        status_callback("[asr] stable-ts is unavailable; falling back to Whisper.")
    return "whisper"


def _transcribe_with_stable_ts(
    audio_path: Path,
    model_name: str,
    device: str,
    chunk_length_s: float | None,
    cache_dir: Path | None,
    chunk_progress_callback: Callable[[int, int], None] | None,
    status_callback: Callable[[str], None] | None,
    hf_token: str | None,
) -> List[Segment]:
    """Transcribe audio with stable-ts for accurate timestamps and reliable opening coverage."""
    stable_whisper = importlib.import_module("stable_whisper")
    resolved_device = resolve_device_selection(device)
    configure_hf_hub_access(hf_token)

    full_cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        full_cache_path = cache_dir / f"segments_stable-ts_{model_name}_{resolved_device}.json"
        if full_cache_path.exists():
            cached = load_segments_from_json(full_cache_path)
            if cached and cached[0].start_s <= 7.0:
                return cached
            if status_callback is not None:
                status_callback("[resume] stable-ts ASR cache starts too late; rebuilding...")

    if status_callback is not None:
        status_callback(
            f"[stable-ts] Loading model '{model_name}' on {resolved_device.upper()} "
            "(first use may download weights)..."
        )

    def _load_model(target_device: str, target_model: str) -> Any:
        return stable_whisper.load_faster_whisper(
            target_model,
            device=target_device,
            compute_type=preferred_whisper_compute_type(target_model, target_device),
        )

    active_device = resolved_device
    active_model = model_name
    try:
        model = _load_model(active_device, active_model)
    except Exception as exc:
        if active_device == "cuda" and is_cuda_runtime_error(exc):
            fallback_model = cpu_fallback_whisper_model(active_model)
            if status_callback is not None:
                status_callback(
                    f"[stable-ts] CUDA unavailable ({exc}). "
                    f"Retrying on CPU with '{fallback_model}'..."
                )
            logging.info("stable-ts CUDA load failed (%s). Retrying on CPU with '%s'.", exc, fallback_model)
            active_device = "cpu"
            active_model = fallback_model
            model = _load_model(active_device, active_model)
        else:
            raise

    def transcribe_chunk(path: Path, offset_s: float = 0.0) -> List[Segment]:
        nonlocal model, active_device, active_model
        if status_callback is not None:
            status_callback(f"[stable-ts] Transcribing '{path.name}'...")
        try:
            result = model.transcribe(str(path), word_timestamps=True, vad=False)
        except Exception as exc:
            if active_device == "cuda" and is_cuda_runtime_error(exc):
                fallback_model = cpu_fallback_whisper_model(active_model)
                if status_callback is not None:
                    status_callback(
                        f"[stable-ts] CUDA runtime error during transcription. "
                        f"Retrying on CPU with '{fallback_model}'..."
                    )
                logging.info(
                    "stable-ts CUDA transcription failed (%s). Retrying on CPU with '%s'.", exc, fallback_model
                )
                active_device = "cpu"
                active_model = fallback_model
                model = _load_model(active_device, active_model)
                result = model.transcribe(str(path), word_timestamps=True, vad=False)
            else:
                raise
        segs: List[Segment] = []
        for seg in result.segments:
            text = safe_text(getattr(seg, "text", "")).strip()
            if not text:
                continue
            start_s = float(getattr(seg, "start", 0.0)) + offset_s
            end_s = float(getattr(seg, "end", 0.0)) + offset_s
            if end_s <= start_s + 0.09:
                continue
            segs.append(Segment(start_s=start_s, end_s=end_s, source_text=text))
        return segs

    audio_duration_s = len(AudioSegment.from_wav(audio_path)) / 1000.0
    should_chunk = chunk_length_s is not None and audio_duration_s > max(chunk_length_s * 1.15, 90.0)

    if should_chunk and chunk_length_s is not None:
        all_segments: List[Segment] = []
        # Non-overlapping chunks: stable-ts timestamp accuracy avoids the need for overlap.
        step_s = chunk_length_s
        total_chunks = max(int((audio_duration_s - 0.001) // step_s) + 1, 1)

        for chunk_index in range(total_chunks):
            chunk_start_s = min(chunk_index * step_s, max(audio_duration_s - 1.0, 0.0))
            chunk_end_s = min(chunk_start_s + step_s, audio_duration_s)
            if chunk_end_s <= chunk_start_s + 0.1:
                continue

            chunk_json_path = (
                cache_dir / f"asr_stable_chunk_{chunk_index + 1:04d}.json"
                if cache_dir is not None
                else None
            )
            if chunk_json_path is not None and chunk_json_path.exists():
                chunk_segs = load_segments_from_json(chunk_json_path)
            else:
                chunk_wav = (
                    cache_dir / f"asr_stable_chunk_{chunk_index + 1:04d}.wav"
                    if cache_dir is not None
                    else audio_path.parent / f"asr_stable_chunk_{chunk_index + 1:04d}.wav"
                )
                run_cmd([
                    "ffmpeg", "-y", "-i", str(audio_path),
                    "-ss", f"{chunk_start_s:.3f}", "-to", f"{chunk_end_s:.3f}",
                    "-acodec", "pcm_s16le", str(chunk_wav),
                ])
                chunk_segs = transcribe_chunk(chunk_wav, offset_s=chunk_start_s)
                if chunk_json_path is not None:
                    save_segments_to_json(chunk_segs, chunk_json_path)

            all_segments.extend(chunk_segs)
            if chunk_progress_callback is not None:
                chunk_progress_callback(chunk_index + 1, total_chunks)

        all_segments.sort(key=lambda s: s.start_s)
        # Collapse any near-duplicate entries produced at chunk boundaries.
        all_segments = collapse_consecutive_duplicate_segments(all_segments, max_gap_s=0.05)
        if full_cache_path is not None:
            save_segments_to_json(all_segments, full_cache_path)
        return all_segments

    segments = transcribe_chunk(audio_path)
    if full_cache_path is not None:
        save_segments_to_json(segments, full_cache_path)
    return segments


def transcribe_segments(
    audio_path: Path,
    model_name: str,
    device: str,
    chunk_length_s: float | None = None,
    cache_dir: Path | None = None,
    chunk_progress_callback: Callable[[int, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    hf_token: str | None = None,
    asr_engine: str = "auto",
) -> List[Segment]:
    resolved_asr_engine = resolve_asr_engine(asr_engine, status_callback=status_callback)

    if resolved_asr_engine == "stable-ts":
        if not stable_ts_available():
            logging.warning(
                "stable-ts is not installed (pip install stable-ts). "
                "Falling back to standard Whisper for this job."
            )
            if status_callback is not None:
                status_callback(
                    "[asr] stable-ts not found, falling back to Whisper. "
                    "Install it with: pip install stable-ts"
                )
        else:
            return _transcribe_with_stable_ts(
                audio_path=audio_path,
                model_name=model_name,
                device=device,
                chunk_length_s=chunk_length_s,
                cache_dir=cache_dir,
                chunk_progress_callback=chunk_progress_callback,
                status_callback=status_callback,
                hf_token=hf_token,
            )
    resolved_device = resolve_device_selection(device)
    active_device = resolved_device
    active_model_name = model_name

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        full_cache_path = cache_dir / f"segments_{model_name}_{resolved_device}.json"
        if full_cache_path.exists():
            cached_segments = load_segments_from_json(full_cache_path)
            # Guard against stale/weak ASR caches that miss long opening speech.
            if cached_segments and cached_segments[0].start_s <= 7.0:
                return cached_segments
            if status_callback is not None:
                status_callback("[resume] ASR cache starts too late; rebuilding transcription for better opening coverage...")
    else:
        full_cache_path = None

    if status_callback is not None:
        status_callback(
            f"[whisper] Loading model '{model_name}' on {resolved_device.upper()} "
            "(first use may download weights)..."
        )

    model = load_whisper_model(
        model_name,
        resolved_device,
        hf_token=hf_token,
        cpu_fallback_model=cpu_fallback_whisper_model(model_name),
    )

    def normalize_text(text: str) -> str:
        lowered = safe_text(text).strip().lower()
        lowered = re.sub(r"\s+", " ", lowered)
        return re.sub(r"[^\w\s']+", "", lowered)

    def resolve_segment_bounds(seg: object) -> tuple[float, float]:
        start_s = float(getattr(seg, "start"))
        end_s = float(getattr(seg, "end"))
        words = getattr(seg, "words", None)
        if not words:
            return start_s, end_s

        timed_words = []
        for word in words:
            word_start = getattr(word, "start", None)
            word_end = getattr(word, "end", None)
            word_text = getattr(word, "word", "")
            if word_start is None or word_end is None or not str(word_text).strip():
                continue
            timed_words.append((float(word_start), float(word_end)))

        if not timed_words:
            return start_s, end_s

        return timed_words[0][0], timed_words[-1][1]

    def has_time_overlap(a: Segment, b: Segment, padding_s: float = 0.18) -> bool:
        return not (a.end_s < b.start_s - padding_s or b.end_s < a.start_s - padding_s)

    def is_same_text(a_text: str, b_text: str) -> bool:
        a_norm = normalize_text(a_text)
        b_norm = normalize_text(b_text)
        if not a_norm or not b_norm:
            return False
        if a_norm == b_norm:
            return True
        shorter, longer = sorted((a_norm, b_norm), key=len)
        return len(shorter) >= 8 and shorter in longer

    def merge_recall_segments(primary: List[Segment], secondary: List[Segment]) -> List[Segment]:
        merged = sorted(primary, key=lambda seg: seg.start_s)
        for cand in sorted(secondary, key=lambda seg: seg.start_s):
            # Drop tiny/noisy fragments from relaxed pass.
            if cand.end_s <= cand.start_s + 0.09:
                continue
            if len(safe_text(cand.source_text).strip()) < 2:
                continue

            overlaps = [existing for existing in merged if has_time_overlap(existing, cand)]
            if not overlaps:
                merged.append(cand)
                continue

            if any(is_same_text(existing.source_text, cand.source_text) for existing in overlaps):
                continue

            # Keep richer relaxed segment when it clearly carries more content.
            best = max(overlaps, key=lambda seg: seg.end_s - seg.start_s)  # type: ignore[union-attr]
            best_dur = best.end_s - best.start_s
            cand_dur = cand.end_s - cand.start_s
            if cand_dur >= best_dur * 1.35 and len(cand.source_text) >= len(best.source_text) + 8:
                merged.remove(best)
                merged.append(cand)

        return sorted(merged, key=lambda seg: seg.start_s)

    def merge_tts_friendly_segments(existing_segments: List[Segment]) -> List[Segment]:
        if not existing_segments:
            return existing_segments

        merged: List[Segment] = [existing_segments[0]]
        for seg in existing_segments[1:]:
            prev = merged[-1]
            gap_s = max(seg.start_s - prev.end_s, 0.0)
            prev_duration_s = prev.end_s - prev.start_s
            combined_duration_s = seg.end_s - prev.start_s
            prev_text = safe_text(prev.source_text).strip()
            next_text = safe_text(seg.source_text).strip()
            looks_like_continuation = (
                (prev_text and prev_text[-1] not in ".!?;:")
                or (next_text[:1].islower() if next_text else False)
                or gap_s <= 0.10
                or (len(prev_text.split()) <= 5 and len(next_text.split()) <= 6)
            )
            should_merge = (
                gap_s <= 0.30
                and combined_duration_s <= 11.0
                and (prev_duration_s <= 1.15 or looks_like_continuation)
            )
            if not should_merge:
                merged.append(seg)
                continue

            joiner = "" if prev_text.endswith("-") or next_text.startswith(("-", "'")) else " "
            merged[-1] = Segment(
                start_s=prev.start_s,
                end_s=max(prev.end_s, seg.end_s),
                source_text=(f"{prev_text}{joiner}{next_text}").strip(),
            )

        return merged

    def collect_segments(target_audio_path: Path, vad_filter: bool, relaxed: bool = False) -> List[Segment]:
        nonlocal model, active_device, active_model_name

        transcribe_kwargs: dict[str, Any] = {
            "vad_filter": vad_filter,
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "beam_size": 1,
            "best_of": 1,
        }
        if vad_filter:
            if relaxed:
                transcribe_kwargs["vad_parameters"] = {
                    "threshold": 0.26,
                    "neg_threshold": 0.18,
                    "min_speech_duration_ms": 70,
                    "min_silence_duration_ms": 140,
                    "speech_pad_ms": 360,
                }
                transcribe_kwargs["no_speech_threshold"] = 0.88
            else:
                transcribe_kwargs["vad_parameters"] = {
                    "threshold": 0.34,
                    "neg_threshold": 0.22,
                    "min_speech_duration_ms": 110,
                    "min_silence_duration_ms": 260,
                    "speech_pad_ms": 280,
                }
        if relaxed:
            # Favor recall for quiet/overlapped speech in the second pass.
            transcribe_kwargs["no_speech_threshold"] = 0.92

        try:
            whisper_segments, _info = model.transcribe(
                str(target_audio_path),
                **transcribe_kwargs,
            )
        except Exception as exc:
            if active_device == "cuda" and is_cuda_runtime_error(exc):
                fallback_model = cpu_fallback_whisper_model(active_model_name)
                if status_callback is not None:
                    status_callback(
                        f"[whisper] CUDA runtime became unavailable during transcription. Retrying on CPU with '{fallback_model}'..."
                    )
                logging.info(
                    "Whisper CUDA transcription failed (%s). Retrying with CPU model '%s'.",
                    exc,
                    fallback_model,
                )
                active_device = "cpu"
                active_model_name = fallback_model
                model = load_whisper_model(active_model_name, "cpu", hf_token=hf_token)
                whisper_segments, _info = model.transcribe(
                    str(target_audio_path),
                    **transcribe_kwargs,
                )
            else:
                raise

        collected: List[Segment] = []
        for seg in whisper_segments:
            text = safe_text(getattr(seg, "text", "")).strip()
            if not text:
                continue
            start_s, end_s = resolve_segment_bounds(seg)
            if end_s <= start_s + 0.09:
                continue
            collected.append(Segment(start_s=start_s, end_s=end_s, source_text=text))
        return collected

    def recover_tail_segments(existing_segments: List[Segment], source_audio_path: Path) -> List[Segment]:
        if not existing_segments:
            return existing_segments

        audio_duration_s = len(AudioSegment.from_wav(source_audio_path)) / 1000.0
        last_end_s = existing_segments[-1].end_s
        trailing_gap_s = audio_duration_s - last_end_s

        probe_starts: list[float] = []
        if trailing_gap_s >= 0.8:
            probe_starts.append(max(last_end_s - 2.5, 0.0))
        # Always inspect the absolute tail; this catches missed final lines even
        # when Whisper's earlier pass ended near the clip boundary.
        probe_starts.append(max(audio_duration_s - 12.0, 0.0))
        probe_starts = sorted(set(round(value, 3) for value in probe_starts))

        appended: List[Segment] = []
        # Use text-only dedup to guard against repeated lyrics at similar times.
        seen_texts = {normalize_text(seg.source_text) for seg in existing_segments[-6:]}
        last_text = normalize_text(existing_segments[-1].source_text)
        seen_positions = [
            (round(seg.start_s, 2), normalize_text(seg.source_text)) for seg in existing_segments
        ]

        for probe_index, tail_start_s in enumerate(probe_starts, start=1):
            tail_audio = AudioSegment.from_wav(source_audio_path)[int(tail_start_s * 1000) :]
            tail_path = source_audio_path.parent / f"tail_recheck_{probe_index}.wav"
            tail_audio.export(tail_path, format="wav")

            recovered_segments = collect_segments(tail_path, vad_filter=False)
            for seg in recovered_segments:
                adjusted = Segment(
                    start_s=seg.start_s + tail_start_s,
                    end_s=seg.end_s + tail_start_s,
                    source_text=seg.source_text,
                )
                if adjusted.end_s <= last_end_s + 0.20 and adjusted.start_s <= last_end_s + 0.20:
                    continue
                normalized = normalize_text(adjusted.source_text)
                if not normalized:
                    continue
                if (round(adjusted.start_s, 2), normalized) in seen_positions:
                    continue
                if normalized in seen_texts:
                    continue
                # If tail ASR overlaps the prior segment boundary, keep only the
                # genuinely new continuation portion instead of dropping it.
                if adjusted.start_s <= last_end_s:
                    if normalized == last_text or normalized in last_text or last_text in normalized:
                        continue
                    adjusted.start_s = last_end_s + 0.05
                    if adjusted.end_s <= adjusted.start_s + 0.08:
                        continue

                appended.append(adjusted)
                seen_texts.add(normalized)
                seen_positions.append((round(adjusted.start_s, 2), normalized))
                last_end_s = max(last_end_s, adjusted.end_s)
                last_text = normalized

        if not appended:
            return existing_segments
        return existing_segments + sorted(appended, key=lambda seg: seg.start_s)

    def recover_head_segments(existing_segments: List[Segment], source_audio_path: Path) -> List[Segment]:
        if not existing_segments:
            return existing_segments

        first_start_s = existing_segments[0].start_s

        # Always probe the opening window with relaxed VAD to catch intro narration
        # that can be partially missed even when first_start_s is near zero.
        head_probe_end_s = min(max(first_start_s + 2.0, 12.0), 20.0)
        head_audio = AudioSegment.from_wav(source_audio_path)[: int(head_probe_end_s * 1000)]
        head_path = source_audio_path.parent / "head_recheck.wav"
        head_audio.export(head_path, format="wav")

        recovered_segments = collect_segments(head_path, vad_filter=False, relaxed=True)
        prepended: List[Segment] = []
        seen_texts = {normalize_text(seg.source_text) for seg in existing_segments[:5]}

        for seg in recovered_segments:
            normalized = normalize_text(seg.source_text)
            if not normalized or normalized in seen_texts:
                continue
            # Keep only opening-window segments and avoid overlap duplicates.
            if seg.start_s > head_probe_end_s:
                continue
            if any(has_time_overlap(seg, existing, padding_s=0.10) for existing in existing_segments[:6]):
                continue

            prepended.append(seg)
            seen_texts.add(normalized)

        if not prepended:
            return existing_segments
        return merge_recall_segments(existing_segments, prepended)

    def transcribe_single_audio(target_audio_path: Path) -> List[Segment]:
        primary_segments = collect_segments(target_audio_path, vad_filter=True)
        recall_vad_segments = collect_segments(target_audio_path, vad_filter=True, relaxed=True)
        recall_segments = collect_segments(target_audio_path, vad_filter=False, relaxed=True)

        if primary_segments:
            merged_segments = merge_recall_segments(primary_segments, recall_vad_segments)
            merged_segments = merge_recall_segments(merged_segments, recall_segments)
            recovered = recover_tail_segments(merged_segments, target_audio_path)
            recovered = recover_head_segments(recovered, target_audio_path)
            return merge_tts_friendly_segments(recovered)

        # If the primary VAD pass found nothing, fall back to the most permissive passes.
        fallback_segments = recall_vad_segments or recall_segments
        if recall_vad_segments and recall_segments:
            fallback_segments = merge_recall_segments(recall_vad_segments, recall_segments)

        recovered = recover_tail_segments(fallback_segments, target_audio_path)
        recovered = recover_head_segments(recovered, target_audio_path)
        return merge_tts_friendly_segments(recovered)

    audio_duration_s = len(AudioSegment.from_wav(audio_path)) / 1000.0
    should_chunk = chunk_length_s is not None and audio_duration_s > max(chunk_length_s * 1.15, 90.0)

    if should_chunk and chunk_length_s is not None:
        merged_all: List[Segment] = []
        chunk_start_s = 0.0
        step_s = max(chunk_length_s - 1.25, 30.0)
        total_chunks = max(int((audio_duration_s - 0.001) // step_s) + 1, 1)

        for chunk_index in range(total_chunks):
            chunk_start_s = min(chunk_index * step_s, max(audio_duration_s - 1.0, 0.0))
            chunk_end_s = min(chunk_start_s + chunk_length_s, audio_duration_s)
            if chunk_end_s <= chunk_start_s + 0.1:
                continue

            chunk_json_path = cache_dir / f"asr_chunk_{chunk_index + 1:04d}.json" if cache_dir is not None else None
            if chunk_json_path is not None and chunk_json_path.exists():
                chunk_segments = load_segments_from_json(chunk_json_path)
            else:
                chunk_wav_path = (
                    cache_dir / f"asr_chunk_{chunk_index + 1:04d}.wav"
                    if cache_dir is not None
                    else audio_path.parent / f"asr_chunk_{chunk_index + 1:04d}.wav"
                )
                run_cmd(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(audio_path),
                        "-ss",
                        f"{chunk_start_s:.3f}",
                        "-to",
                        f"{chunk_end_s:.3f}",
                        "-acodec",
                        "pcm_s16le",
                        str(chunk_wav_path),
                    ]
                )
                local_segments = transcribe_single_audio(chunk_wav_path)
                chunk_segments = [
                    Segment(
                        start_s=seg.start_s + chunk_start_s,
                        end_s=seg.end_s + chunk_start_s,
                        source_text=seg.source_text,
                        translated_text=seg.translated_text,
                    )
                    for seg in local_segments
                ]
                if chunk_json_path is not None:
                    save_segments_to_json(chunk_segments, chunk_json_path)

            merged_all = merge_recall_segments(merged_all, chunk_segments) if merged_all else chunk_segments
            if chunk_progress_callback is not None:
                chunk_progress_callback(chunk_index + 1, total_chunks)

        if full_cache_path is not None:
            save_segments_to_json(merged_all, full_cache_path)
        return merged_all

    single_segments = transcribe_single_audio(audio_path)
    if full_cache_path is not None:
        save_segments_to_json(single_segments, full_cache_path)
    return single_segments


def english_word_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
    }


def has_untranslated_english_tokens(source_text: str, translated_text: str, target_lang: str) -> bool:
    if target_lang == "en":
        return False

    source_tokens = english_word_tokens(source_text)
    translated_tokens = english_word_tokens(translated_text)
    if not source_tokens or not translated_tokens:
        return False

    common_tokens = source_tokens & translated_tokens
    ignored_tokens = {"oh", "yeah", "hey", "la", "na"}
    return any(token not in ignored_tokens for token in common_tokens)


def replace_untranslated_tokens(
    source_text: str,
    translated_text: str,
    word_translator: GoogleTranslator,
) -> str:
    source_tokens = english_word_tokens(source_text)
    translated_tokens = english_word_tokens(translated_text)
    common_tokens = [token for token in source_tokens & translated_tokens if token not in {"oh", "yeah", "hey", "la", "na"}]

    repaired_text = translated_text
    for token in sorted(common_tokens, key=len, reverse=True):
        try:
            replacement = word_translator.translate(token)
        except Exception:
            # Some providers raise on uncommon/non-English tokens; keep pipeline running.
            continue
        if replacement is None:
            continue
        replacement = replacement.strip()
        if not replacement or replacement.lower() == token.lower():
            continue
        repaired_text = re.sub(
            rf"\b{re.escape(token)}\b",
            replacement,
            repaired_text,
            flags=re.IGNORECASE,
        )
    return repaired_text


def split_for_translation(text: str, max_chars: int = 420) -> List[str]:
    """Split long text into smaller chunks to avoid provider payload failures."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: List[str] = []
    cursor = 0
    while cursor < len(normalized):
        window = normalized[cursor : cursor + max_chars]
        if len(window) < max_chars:
            chunks.append(window.strip())
            break

        split_at = max(
            window.rfind(". "),
            window.rfind("? "),
            window.rfind("! "),
            window.rfind(", "),
            window.rfind("; "),
        )
        if split_at < 60:
            split_at = window.rfind(" ")
        if split_at < 30:
            split_at = len(window)

        piece = normalized[cursor : cursor + split_at].strip()
        if piece:
            chunks.append(piece)
        cursor += split_at
        while cursor < len(normalized) and normalized[cursor] == " ":
            cursor += 1

    return chunks or [normalized]


def safe_translate(
    text: str,
    translator: Any,
    fallback_translator: Any,
) -> str:
    """Translate text with retries and chunk fallback to avoid hard pipeline failures."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""

    errors: list[str] = []
    for attempt in range(3):
        try:
            result = translator.translate(normalized)
            if result is not None and result.strip():
                return result.strip()
        except Exception as exc:  # pragma: no cover - network/provider variability
            errors.append(str(exc))
        time.sleep(0.12 * (attempt + 1))

    for attempt in range(2):
        try:
            result = fallback_translator.translate(normalized)
            if result is not None and result.strip():
                return result.strip()
        except Exception as exc:  # pragma: no cover - network/provider variability
            errors.append(str(exc))
        time.sleep(0.12 * (attempt + 1))

    chunks = split_for_translation(normalized)
    if len(chunks) > 1:
        translated_chunks: List[str] = []
        for chunk in chunks:
            chunk_result = ""
            for candidate in (translator, fallback_translator):
                try:
                    translated = candidate.translate(chunk)
                    if translated is not None and translated.strip():
                        chunk_result = translated.strip()
                        break
                except Exception as exc:  # pragma: no cover - network/provider variability
                    errors.append(str(exc))
            if not chunk_result:
                chunk_result = chunk
            translated_chunks.append(chunk_result)
        combined = " ".join(part for part in translated_chunks if part).strip()
        if combined:
            return combined

    # Final fallback keeps long-video jobs running even if provider fails.
    return normalized


def translate_segments_with_progress(
    segments: List[Segment],
    target_lang: str,
    segment_progress_callback: Callable[[int, int], None] | None = None,
    glossary_overrides: dict[str, str] | None = None,
    translation_provider: str = "google",
    force_english_source: bool = False,
) -> None:
    normalized_provider = translation_provider.strip().lower() if translation_provider else "google"
    translator = build_translator(normalized_provider, source="auto", target=target_lang)
    fallback_provider = "mymemory" if normalized_provider == "google" else "google"
    fallback_translator = build_translator(fallback_provider, source="auto", target=target_lang)
    explicit_en_translator = build_translator(normalized_provider, source="en", target=target_lang)
    explicit_en_fallback_translator = build_translator(fallback_provider, source="en", target=target_lang)
    word_translator = build_translator("google", source="en", target=target_lang)
    translation_cache: dict[str, str] = {}

    def recommended_translation_workers(item_count: int, provider: str) -> int:
        # Keep concurrency conservative to avoid provider throttling and translation quality drops.
        if item_count < 20:
            return 1
        if provider == "google":
            return 2
        if provider == "mymemory":
            return 3
        return 1

    def should_retry_with_english_source(source_text: str, translated_text: str) -> bool:
        if target_lang == "en":
            return False
        source_clean = re.sub(r"\s+", " ", source_text).strip().lower()
        translated_clean = re.sub(r"\s+", " ", translated_text).strip().lower()
        if not source_clean or not translated_clean:
            return False
        if source_clean != translated_clean:
            return False
        # Retry when the text looks like normal sentence content, not just symbols.
        alpha_chars = re.findall(r"[A-Za-z]", source_clean)
        word_count = len(source_clean.split())
        return len(alpha_chars) >= 4 and word_count >= 2

    def looks_untranslated(source_text: str, translated_text: str) -> bool:
        if target_lang == "en":
            return False

        translated_clean = re.sub(r"\s+", " ", translated_text).strip()
        if not translated_clean:
            return True

        if should_retry_with_english_source(source_text, translated_text):
            return True

        if has_untranslated_english_tokens(source_text, translated_text, target_lang):
            return True

        return False

    def translate_source_text(source_text: str) -> str:
        if force_english_source:
            translated = safe_translate(source_text, explicit_en_translator, explicit_en_fallback_translator)
        else:
            translated = safe_translate(source_text, translator, fallback_translator)

        if not force_english_source and looks_untranslated(source_text, translated):
            retry = safe_translate(source_text, explicit_en_translator, explicit_en_fallback_translator)
            if retry is not None and retry.strip():
                translated = retry

        if not force_english_source and looks_untranslated(source_text, translated):
            retry = safe_translate(source_text, fallback_translator, translator)
            if retry is not None and retry.strip():
                translated = retry

        if looks_untranslated(source_text, translated):
            retry = safe_translate(source_text, explicit_en_fallback_translator, explicit_en_translator)
            if retry is not None and retry.strip():
                translated = retry

        if has_untranslated_english_tokens(source_text, translated, target_lang):
            translated = replace_untranslated_tokens(source_text, translated, word_translator)

        return translated

    total = len(segments)
    unchanged_count = 0
    if total <= 0:
        return

    segment_keys: list[str] = []
    source_by_key: dict[str, str] = {}
    key_counts: dict[str, int] = {}
    for seg in segments:
        source_key = re.sub(r"\s+", " ", seg.source_text).strip().lower()
        segment_keys.append(source_key)
        if source_key not in source_by_key:
            source_by_key[source_key] = seg.source_text
            key_counts[source_key] = 0
        key_counts[source_key] += 1

    unique_items = list(source_by_key.items())
    worker_count = min(recommended_translation_workers(total, normalized_provider), len(unique_items))

    if worker_count > 1 and len(unique_items) > 1:
        completed_segments = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(translate_source_text, source_text): source_key
                for source_key, source_text in unique_items
            }
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="Translating"):
                source_key = future_map[future]
                source_text = source_by_key[source_key]
                try:
                    translation_cache[source_key] = future.result()
                except Exception:
                    # Keep the job moving if one provider call fails unexpectedly.
                    translation_cache[source_key] = source_text
                completed_segments += key_counts.get(source_key, 1)
                if segment_progress_callback is not None:
                    segment_progress_callback(min(completed_segments, total), total)
    else:
        completed_segments = 0
        for idx, (source_key, source_text) in enumerate(tqdm(unique_items, desc="Translating"), start=1):
            try:
                translation_cache[source_key] = translate_source_text(source_text)
            except Exception:
                translation_cache[source_key] = source_text
            completed_segments += key_counts.get(source_key, 1)
            if segment_progress_callback is not None:
                segment_progress_callback(min(completed_segments, total), total)

    for seg, source_key in zip(segments, segment_keys):
        translated = translation_cache.get(source_key, seg.source_text)

        if glossary_overrides:
            translated = apply_glossary_overrides(translated, glossary_overrides)

        if target_lang != "en":
            source_clean = re.sub(r"\s+", " ", seg.source_text).strip().lower()
            translated_clean = re.sub(r"\s+", " ", translated).strip().lower()
            if source_clean and source_clean == translated_clean:
                unchanged_count += 1

        seg.translated_text = translated

    if target_lang != "en" and total > 0 and unchanged_count / total > 0.45:
        raise RuntimeError(
            "Translation provider returned too many unchanged lines. "
            "Try again in a minute or use a different target language code."
        )


def edge_tts_segment(
    text: str,
    voice: str,
    output_mp3: Path,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
) -> None:
    edge_tts = importlib.import_module("edge_tts")
    configure_windows_asyncio_policy()

    async def synthesize() -> None:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )
        await communicate.save(str(output_mp3))

    if os.name == "nt" and hasattr(asyncio, "SelectorEventLoop"):
        loop = asyncio.SelectorEventLoop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(synthesize())
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
        return

    asyncio.run(synthesize())


def tts_segment(
    text: str,
    lang: str,
    output_mp3: Path,
    tts_engine: str = "edge",
    edge_voice: str | None = None,
    edge_rate: str = "+0%",
    edge_pitch: str = "+0Hz",
    edge_volume: str = "+0%",
) -> None:
    last_error: Exception | None = None

    if tts_engine == "edge":
        voice = edge_voice or DEFAULT_EDGE_VOICES.get(lang, DEFAULT_EDGE_VOICES["en"])
        for attempt in range(3):
            try:
                edge_tts_segment(
                    text,
                    voice,
                    output_mp3,
                    rate=edge_rate,
                    pitch=edge_pitch,
                    volume=edge_volume,
                )
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.18 * (attempt + 1))

    for attempt in range(2):
        try:
            tts = gTTS(text=text, lang=lang)
            tts.save(str(output_mp3))
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.18 * (attempt + 1))

    raise RuntimeError(f"TTS synthesis failed after retries: {last_error}")


def sanitize_tts_text(text: str) -> str:
    """Clean text so TTS gets stable, natural input."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"([!?.,])\1{1,}", r"\1", cleaned)
    cleaned = re.sub(r"\s*[-–—]\s*", ", ", cleaned)
    cleaned = re.sub(r"\s*([,;:.!?])\s*", r"\1 ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def format_edge_rate(percent: int) -> str:
    bounded = max(min(percent, 40), -50)
    if bounded >= 0:
        return f"+{bounded}%"
    return f"{bounded}%"


def format_edge_pitch(hz: int) -> str:
    bounded = max(min(hz, 18), -12)
    if bounded >= 0:
        return f"+{bounded}Hz"
    return f"{bounded}Hz"


def inject_mid_sentence_pause(text: str) -> str:
    if len(text) < 70 or re.search(r"[,;:!?]", text):
        return text

    midpoint = len(text) // 2
    connector_matches = list(
        re.finditer(r"\b(and|but|because|so|while|when|which|that|although|though)\b", text, flags=re.IGNORECASE)
    )
    if not connector_matches:
        return text

    split_at = min(connector_matches, key=lambda match: abs(match.start() - midpoint)).start()
    if split_at < 24 or len(text) - split_at < 24:
        return text
    return f"{text[:split_at].rstrip()}, {text[split_at:].lstrip()}"


def build_edge_tts_profile(text: str) -> tuple[str, int, int, str]:
    spoken_text = inject_mid_sentence_pause(sanitize_tts_text(text))
    word_count = len(spoken_text.split())
    punctuation_count = len(re.findall(r"[,;:]", spoken_text))

    rate_percent = 0
    if word_count >= 16:
        rate_percent = -8
    elif word_count >= 10:
        rate_percent = -5
    elif word_count >= 6:
        rate_percent = -3

    if punctuation_count >= 2:
        rate_percent -= 2

    pitch_hz = 2
    if spoken_text.endswith("?"):
        pitch_hz = 10
    elif spoken_text.endswith("!"):
        pitch_hz = 7
    elif word_count <= 4:
        pitch_hz = 4

    return spoken_text, rate_percent, pitch_hz, "+0%"


def build_atempo_filter(speed: float) -> str:
    """Build ffmpeg atempo chain within per-filter 0.5..2.0 limits."""
    safe_speed = max(speed, 0.01)
    parts: list[str] = []

    while safe_speed < 0.5:
        parts.append("atempo=0.5")
        safe_speed /= 0.5

    while safe_speed > 2.0:
        parts.append("atempo=2.0")
        safe_speed /= 2.0

    parts.append(f"atempo={safe_speed:.5f}")
    return ",".join(parts)


def stretch_audio_preserve_pitch(audio: AudioSegment, speed: float, temp_dir: Path, stem: str) -> AudioSegment:
    """Time-stretch audio with ffmpeg while preserving pitch."""
    in_wav = temp_dir / f"{stem}_in.wav"
    out_wav = temp_dir / f"{stem}_out.wav"
    audio.export(in_wav, format="wav")

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(in_wav),
            "-filter:a",
            build_atempo_filter(speed),
            str(out_wav),
        ]
    )

    return AudioSegment.from_wav(out_wav)


def trim_segment_silence(audio: AudioSegment, trim_ms: int) -> AudioSegment:
    """Trim quiet leading/trailing regions while keeping a small natural pad."""
    if trim_ms <= 0 or len(audio) <= 0:
        return audio

    ranges = detect_nonsilent(audio, min_silence_len=80, silence_thresh=-42)
    if not ranges:
        return audio

    start = max(ranges[0][0] - trim_ms, 0)
    end = min(ranges[-1][1] + trim_ms, len(audio))
    if end <= start:
        return audio
    return cast(AudioSegment, audio[start:end])


def trim_initial_tts_latency(audio: AudioSegment, max_leading_trim_ms: int = 180) -> AudioSegment:
    """Remove synthetic leading silence so lines start closer to original timing."""
    if len(audio) <= 0 or max_leading_trim_ms <= 0:
        return audio

    ranges = detect_nonsilent(audio, min_silence_len=45, silence_thresh=-44)
    if not ranges:
        return audio

    lead_ms = max(ranges[0][0], 0)
    if lead_ms <= 18:
        return audio

    trim_ms = min(lead_ms, max_leading_trim_ms)
    if trim_ms >= len(audio):
        return audio
    return cast(AudioSegment, audio[trim_ms:])


def fit_audio_to_duration_with_controls(
    audio: AudioSegment,
    target_ms: int,
    temp_dir: Path,
    segment_index: int,
    min_stretch_speed: float,
    max_stretch_speed: float,
    silence_trim_ms: int,
) -> AudioSegment:
    audio = trim_initial_tts_latency(audio)

    if silence_trim_ms > 0:
        audio = trim_segment_silence(audio, silence_trim_ms)

    if target_ms <= 0:
        return AudioSegment.silent(duration=0)

    current_ms = len(audio)
    if current_ms <= 0:
        return AudioSegment.silent(duration=target_ms)

    required_speed = current_ms / max(target_ms, 1)
    clamped_speed = min(max(required_speed, min_stretch_speed), max_stretch_speed)

    if abs(clamped_speed - 1.0) > 0.08:
        audio = stretch_audio_preserve_pitch(audio, clamped_speed, temp_dir, f"seg_{segment_index:05d}")

    # Safety pass: if audio is still too long, allow one extra stretch to avoid dropping words.
    if len(audio) > target_ms:
        overflow_speed = len(audio) / max(target_ms, 1)
        if overflow_speed > 1.08:
            # Keep emergency compression bounded so a single difficult line does
            # not become unnaturally fast.
            safety_speed = min(max(overflow_speed, 1.0), max(max_stretch_speed + 0.18, 1.95))
            if abs(safety_speed - 1.0) > 0.08:
                audio = stretch_audio_preserve_pitch(audio, safety_speed, temp_dir, f"seg_{segment_index:05d}_safe")

    if len(audio) > target_ms:
        fade_ms = min(80, max(target_ms // 6, 20))
        clipped = cast(AudioSegment, audio[:target_ms])
        return clipped.fade_out(fade_ms)
    smoothed = audio.fade_in(min(22, max(len(audio) // 12, 10))).fade_out(min(48, max(len(audio) // 10, 18)))
    if len(smoothed) < target_ms:
        return smoothed + AudioSegment.silent(duration=target_ms - len(smoothed))
    return smoothed


def has_meaningful_audio(audio_path: Path, min_nonsilent_ms: int = 450) -> bool:
    """Return True when audio contains enough non-silent content to be considered usable speech."""
    if not audio_path.exists():
        return False

    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception:
        return False

    if len(audio) <= 0:
        return False

    silence_floor = audio.dBFS - 18 if audio.dBFS != float("-inf") else -45
    ranges = detect_nonsilent(audio, min_silence_len=120, silence_thresh=max(silence_floor, -45))
    nonsilent_ms = sum(max(end - start, 0) for start, end in ranges)
    return nonsilent_ms >= min_nonsilent_ms


def build_dubbed_track(
    segments: List[Segment],
    total_duration_ms: int,
    temp_dir: Path,
    target_lang: str,
    tts_engine: str = "edge",
    edge_voice: str | None = None,
    min_stretch_speed: float = 0.85,
    max_stretch_speed: float = 1.20,
    silence_trim_ms: int = 0,
    segment_progress_callback: Callable[[int, int], None] | None = None,
    chunk_window_s: float | None = None,
    cache_dir: Path | None = None,
) -> AudioSegment:
    dubbed = AudioSegment.silent(duration=total_duration_ms)
    total = len(segments)
    voice_cache: dict[tuple[str, str, str, str, str, str, str], AudioSegment] = {}

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def get_tts_audio(
        spoken_text: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> AudioSegment:
        cache_key = (spoken_text, target_lang, tts_engine, edge_voice or "", rate, pitch, volume)
        cached_audio = voice_cache.get(cache_key)
        if cached_audio is not None:
            return cached_audio

        mp3_path = temp_dir / f"tts_cache_{len(voice_cache) + 1:05d}.mp3"
        tts_segment(
            spoken_text,
            target_lang,
            mp3_path,
            tts_engine=tts_engine,
            edge_voice=edge_voice,
            edge_rate=rate,
            edge_pitch=pitch,
            edge_volume=volume,
        )
        cached_audio = AudioSegment.from_file(mp3_path)
        voice_cache[cache_key] = cached_audio
        return cached_audio

    def synthesize_chunk(chunk_items: list[tuple[int, Segment]], chunk_start_ms: int, chunk_end_ms: int) -> AudioSegment:
        chunk_audio = AudioSegment.silent(duration=max(chunk_end_ms - chunk_start_ms, 120))
        for i, seg in chunk_items:
            line_text = safe_text(seg.translated_text).strip() or safe_text(seg.source_text).strip()
            if not line_text:
                if segment_progress_callback is not None:
                    segment_progress_callback(i, total)
                continue

            spoken_text, base_rate_percent, pitch_hz, volume = build_edge_tts_profile(line_text)
            if not spoken_text:
                if segment_progress_callback is not None:
                    segment_progress_callback(i, total)
                continue

            global_start_ms = max(int(seg.start_s * 1000), 0)
            start_ms = max(global_start_ms - chunk_start_ms, 0)
            original_end_ms = max(int(seg.end_s * 1000), global_start_ms + 120)
            if i < total:
                next_start_ms = max(int(segments[i].start_s * 1000) - 60, original_end_ms)
                allowed_end_ms = min(next_start_ms, total_duration_ms)
            else:
                allowed_end_ms = total_duration_ms
            slot_ms = max(allowed_end_ms - global_start_ms, 120)

            edge_rate_percent = base_rate_percent
            if tts_engine == "edge" and slot_ms > 0:
                desired_ratio = len(get_tts_audio(spoken_text, rate=format_edge_rate(base_rate_percent), pitch=format_edge_pitch(pitch_hz), volume=volume)) / slot_ms
                if desired_ratio > 1.05:
                    edge_rate_percent += int(min((desired_ratio - 1.0) * 52, 34))
                elif desired_ratio < 0.78:
                    edge_rate_percent -= int(min((1.0 - desired_ratio) * 16, 6))

            edge_rate = format_edge_rate(edge_rate_percent)
            edge_pitch = format_edge_pitch(pitch_hz)
            voice = get_tts_audio(spoken_text, rate=edge_rate, pitch=edge_pitch, volume=volume)

            if tts_engine == "edge" and slot_ms > 0:
                natural_ratio = len(voice) / slot_ms
                if natural_ratio > max_stretch_speed + 0.12:
                    adaptive_rate = format_edge_rate(edge_rate_percent + int(min((natural_ratio - 1.0) * 28, 24)))
                    voice = get_tts_audio(spoken_text, rate=adaptive_rate, pitch=edge_pitch, volume=volume)
                    # Some lines still run long after one rate bump. Try one stronger pass
                    # rather than expanding the segment slot and drifting into the next line.
                    if len(voice) > slot_ms:
                        second_ratio = len(voice) / slot_ms
                        if second_ratio > max_stretch_speed + 0.08:
                            stronger_rate = format_edge_rate(edge_rate_percent + int(min((second_ratio - 1.0) * 34, 28)))
                            voice = get_tts_audio(spoken_text, rate=stronger_rate, pitch=edge_pitch, volume=volume)

            voice = fit_audio_to_duration_with_controls(
                voice,
                slot_ms,
                temp_dir,
                i,
                min_stretch_speed=min_stretch_speed,
                max_stretch_speed=max_stretch_speed,
                silence_trim_ms=silence_trim_ms,
            )
            chunk_audio = chunk_audio.overlay(voice, position=start_ms)
            if segment_progress_callback is not None:
                segment_progress_callback(i, total)

        return chunk_audio

    if not segments:
        return dubbed

    chunk_window_ms = int((chunk_window_s or (total_duration_ms / 1000.0)) * 1000)
    chunk_window_ms = max(chunk_window_ms, 15_000)
    chunk_pad_ms = 1_200
    grouped: dict[int, list[tuple[int, Segment]]] = {}
    for i, seg in enumerate(segments, start=1):
        chunk_index = int(max(seg.start_s * 1000, 0) // chunk_window_ms)
        grouped.setdefault(chunk_index, []).append((i, seg))

    for chunk_index in sorted(grouped):
        chunk_start_ms = chunk_index * chunk_window_ms
        chunk_end_ms = min(chunk_start_ms + chunk_window_ms + chunk_pad_ms, total_duration_ms)
        chunk_path = cache_dir / f"dub_chunk_{chunk_index:04d}.wav" if cache_dir is not None else None

        if chunk_path is not None and chunk_path.exists():
            chunk_audio = AudioSegment.from_wav(chunk_path)
            for i, _seg in grouped[chunk_index]:
                if segment_progress_callback is not None:
                    segment_progress_callback(i, total)
        else:
            chunk_audio = synthesize_chunk(grouped[chunk_index], chunk_start_ms, chunk_end_ms)
            if chunk_path is not None:
                chunk_audio.export(chunk_path, format="wav")

        dubbed = dubbed.overlay(chunk_audio, position=chunk_start_ms)

    return dubbed


def mux_video_with_dub(
    input_video: Path,
    dubbed_wav: Path,
    output_video: Path,
    background_mix_level: float = 0.08,
    include_original_audio: bool = True,
) -> None:
    def build_dub_only_cmd() -> list[str]:
        return [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(dubbed_wav),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_video),
        ]

    if include_original_audio:
        # Keep original ambience under the dub, but make dubbed speech dominant.
        mix = min(max(background_mix_level, 0.0), 1.0)
        advanced_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(dubbed_wav),
            "-filter_complex",
            (
                f"[1:a]acompressor=threshold=-20dB:ratio=2.2:attack=15:release=140,"
                f"alimiter=limit=0.97,volume=1.9[dub];"
                f"[0:a]volume={mix:.3f}[bg];"
                f"[bg][dub]sidechaincompress=threshold=0.025:ratio=10:attack=20:release=260:makeup=1[ducked];"
                f"[ducked][dub]amix=inputs=2:weights='0.22 1.78':duration=first:normalize=0[aout]"
            ),
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_video),
        ]
        simple_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(dubbed_wav),
            "-filter_complex",
            f"[0:a]volume={mix:.3f}[bg];[1:a]volume=1.9[dub];[bg][dub]amix=inputs=2:weights='0.24 1.76':duration=first:normalize=0[aout]",
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_video),
        ]

        try:
            run_cmd(advanced_cmd)
            return
        except RuntimeError as exc:
            logging.warning("Advanced audio ducking mix failed; retrying with simple amix. Error: %s", exc)
            try:
                run_cmd(simple_cmd)
                return
            except RuntimeError as fallback_exc:
                logging.warning("Simple amix fallback failed; exporting dubbed-only audio. Error: %s", fallback_exc)
                run_cmd(build_dub_only_cmd())
                return
    else:
        # Output dubbed speech only, preserving the original video stream.
        run_cmd(build_dub_only_cmd())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-dub a video to a target language")
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument("--output", required=True, type=Path, help="Output dubbed video path")
    parser.add_argument("--target-lang", required=True, help="Target language code, e.g. es, hi, fr")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument(
        "--translation-provider",
        default="google",
        choices=["google", "mymemory"],
        help="Translation backend to use",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for faster Whisper model downloads and higher rate limits",
    )
    parser.add_argument("--tts-engine", default="edge", choices=["edge", "gtts"], help="Speech synthesis engine")
    parser.add_argument("--edge-voice", default=None, help="Edge voice name, e.g. en-US-AriaNeural")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds for dubbing window")
    parser.add_argument("--end-time", type=float, default=None, help="Optional end time in seconds for dubbing window")
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate files")
    parser.add_argument(
        "--disable-original-audio",
        action="store_true",
        help="Disable original source audio in the final mix (dubbed speech only)",
    )
    parser.add_argument(
        "--optimization-profile",
        default="auto",
        choices=["auto", "balanced", "short", "long"],
        help="Auto-tune settings for short or long videos",
    )
    parser.add_argument("--no-export-srt", action="store_true", help="Skip translated subtitle export")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume cache reuse across reruns")
    parser.add_argument(
        "--glossary-file",
        type=Path,
        default=None,
        help="Optional glossary override file with 'source => replacement' rules",
    )
    parser.add_argument(
        "--asr-engine",
        default="auto",
        choices=["auto", "whisper", "stable-ts"],
        help="Speech recognition engine: 'auto' (prefer stable-ts when available), 'whisper', or 'stable-ts'",
    )
    return parser.parse_args()


def autodub_video(
    input_path: Path,
    output_path: Path,
    target_lang: str,
    whisper_model: str = "small",
    device: str = "auto",
    translation_provider: str = "google",
    hf_token: str | None = None,
    tts_engine: str = "edge",
    edge_voice: str | None = None,
    background_mix_level: float = 0.08,
    include_original_audio: bool = True,
    min_stretch_speed: float = 0.85,
    max_stretch_speed: float = 1.80,
    silence_trim_ms: int = 0,
    optimization_profile: str = "auto",
    export_srt: bool = True,
    resume_enabled: bool = True,
    glossary_text: str = "",
    asr_engine: str = "auto",
    start_time_s: float = 0.0,
    end_time_s: float | None = None,
    keep_temp: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    progress_percent_callback: Callable[[float, str], None] | None = None,
) -> int:
    started_at = time.perf_counter()

    def has_weak_opening_coverage(segments: List[Segment]) -> bool:
        if not segments:
            return True
        first_start = segments[0].start_s
        early_segments = [seg for seg in segments if seg.start_s < 12.0]
        return first_start > 1.4 or len(early_segments) < 2

    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    def report_progress(value: float, label: str) -> None:
        if progress_percent_callback is not None:
            bounded = min(max(value, 0.0), 1.0)
            progress_percent_callback(bounded, label)

    has_hf_token = configure_hf_hub_access(hf_token)
    ensure_ffmpeg()

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if start_time_s < 0:
        raise ValueError("start_time_s must be >= 0")
    if end_time_s is not None and end_time_s <= start_time_s:
        raise ValueError("end_time_s must be greater than start_time_s")
    if min_stretch_speed <= 0 or max_stretch_speed <= 0:
        raise ValueError("Stretch speeds must be > 0")
    if min_stretch_speed > max_stretch_speed:
        raise ValueError("min_stretch_speed must be <= max_stretch_speed")
    if silence_trim_ms < 0:
        raise ValueError("silence_trim_ms must be >= 0")

    resolved_asr_engine = resolve_asr_engine(asr_engine, status_callback=report)

    temp_base = Path(tempfile.mkdtemp(prefix="autodub_"))
    try:
        resume_dir = None
        if resume_enabled:
            resume_dir = build_resume_dir(
                input_path=input_path,
                output_dir=output_path.parent,
                target_lang=target_lang,
                whisper_model=whisper_model,
                translation_provider=translation_provider,
                tts_engine=tts_engine,
                edge_voice=edge_voice,
                optimization_profile=optimization_profile,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                glossary_text=glossary_text,
                asr_engine=resolved_asr_engine,
            )
            resume_dir.mkdir(parents=True, exist_ok=True)
            report(f"[resume] Cache directory: {resume_dir}")

        working_video = input_path
        extracted_wav = (resume_dir / "extracted.wav") if resume_dir is not None else (temp_base / "extracted.wav")
        dubbed_wav = (resume_dir / "dubbed.wav") if resume_dir is not None else (temp_base / "dubbed.wav")
        segments_json = (resume_dir / "segments.json") if resume_dir is not None else (temp_base / "segments.json")
        asr_cache_dir = (resume_dir / "asr") if resume_dir is not None else None
        tts_cache_dir = (resume_dir / "tts_chunks") if resume_dir is not None else None
        dub_meta_path = (resume_dir / "dubbed_meta.json") if resume_dir is not None else (temp_base / "dubbed_meta.json")
        subtitle_path = output_path.with_suffix(".srt")
        glossary_overrides = parse_glossary_overrides(glossary_text)

        if start_time_s > 0 or end_time_s is not None:
            clipped_video = (resume_dir / "trimmed_input.mp4") if resume_dir is not None else (temp_base / "trimmed_input.mp4")
            if clipped_video.exists():
                report("[resume] Reusing trimmed video range...")
            else:
                report("[0/5] Trimming selected video range...")
                report_progress(0.01, "Preparing selected time range")
                trim_video(input_path, clipped_video, start_time_s, end_time_s)
            working_video = clipped_video

        resolved_settings = resolve_processing_profile(
            optimization_profile,
            probe_media_duration(working_video),
            whisper_model,
            device,
            min_stretch_speed,
            max_stretch_speed,
            silence_trim_ms,
        )
        clip_duration_s = resolved_settings["clip_duration_s"]
        clip_label = "unknown length"
        if isinstance(clip_duration_s, float):
            clip_label = f"{clip_duration_s / 60:.1f} min" if clip_duration_s >= 60 else f"{clip_duration_s:.0f} sec"
        report(
            "[opt] "
            f"Profile={resolved_settings['label']} | clip={clip_label} | "
            f"Device={resolved_settings['device']} | Whisper={resolved_settings['whisper_model']} | "
            f"Translate={translation_provider}"
        )
        if cast(str, resolved_settings["whisper_model"]) in LARGE_WHISPER_MODELS:
            report("[whisper] Large models may download several GB on first use.")
            if not has_hf_token:
                report("[hf] Optional: set `HF_TOKEN` or use the UI token field for higher rate limits and faster downloads.")
        if cast(str, resolved_settings["device"]) != "cuda" and cast(str, resolved_settings["whisper_model"]) in LARGE_WHISPER_MODELS:
            report("[hw] Large Whisper on CPU can be slow; use CUDA or switch to 'medium' for faster runs.")

        report("[1/5] Extracting audio...")
        if extracted_wav.exists():
            report("[resume] Reusing extracted audio cache...")
            report_progress(0.10, "Audio extracted")
        else:
            report_progress(0.03, "Extracting audio")
            extract_audio(working_video, extracted_wav)
            report_progress(0.10, "Audio extracted")

        _asr_label = "stable-ts" if resolved_asr_engine == "stable-ts" else "Whisper"
        report(f"[2/5] Transcribing with {_asr_label}...")

        def asr_chunk_progress(done: int, total: int) -> None:
            start = 0.12
            end = 0.35
            fraction = done / max(total, 1)
            report_progress(start + (end - start) * fraction, f"Transcribing chunks ({done}/{total})")

        if segments_json.exists():
            segments = load_segments_from_json(segments_json)
            report(f"[resume] Reusing cached segments ({len(segments)} segments)...")
            if has_weak_opening_coverage(segments):
                report("[resume] Cached ASR seems to miss opening speech; rebuilding transcription...")
                report_progress(0.12, "Re-transcribing opening coverage")
                segments = transcribe_segments(
                    extracted_wav,
                    cast(str, resolved_settings["whisper_model"]),
                    cast(str, resolved_settings["device"]),
                    chunk_length_s=cast(float | None, resolved_settings["transcribe_chunk_s"]),
                    cache_dir=asr_cache_dir,
                    chunk_progress_callback=asr_chunk_progress,
                    status_callback=report,
                    hf_token=hf_token,
                    asr_engine=resolved_asr_engine,
                )
                save_segments_to_json(segments, segments_json)

            report_progress(0.35, f"Transcription complete ({len(segments)} segments)")
        else:
            report_progress(0.12, "Transcribing speech")
            segments = transcribe_segments(
                extracted_wav,
                cast(str, resolved_settings["whisper_model"]),
                cast(str, resolved_settings["device"]),
                chunk_length_s=cast(float | None, resolved_settings["transcribe_chunk_s"]),
                cache_dir=asr_cache_dir,
                chunk_progress_callback=asr_chunk_progress,
                status_callback=report,
                hf_token=hf_token,
                asr_engine=resolved_asr_engine,
            )
            save_segments_to_json(segments, segments_json)
            report_progress(0.35, f"Transcription complete ({len(segments)} segments)")

        if not segments:
            raise RuntimeError(
                "No speech segments found in the selected range. "
                "Try a different time window, use a larger Whisper model, or increase spoken content."
            )

        report("[3/5] Translating segments...")

        def translation_progress(done: int, total: int) -> None:
            start = 0.35
            end = 0.60
            fraction = done / max(total, 1)
            report_progress(start + (end - start) * fraction, f"Translating ({done}/{total})")

        needs_translation = any(not safe_text(seg.translated_text).strip() for seg in segments)
        if not needs_translation and cached_translation_looks_poor(segments, target_lang):
            report("[resume] Cached translations look low quality; re-translating segments...")
            needs_translation = True
        if not needs_translation and translation_looks_wrong_language(segments, target_lang):
            report("[resume] Cached translations appear to be in the wrong language; re-translating segments...")
            needs_translation = True
        if needs_translation:
            translate_segments_with_progress(
                segments,
                target_lang,
                segment_progress_callback=translation_progress,
                glossary_overrides=glossary_overrides,
                translation_provider=translation_provider,
            )
            if translation_looks_wrong_language(segments, target_lang):
                report("[translate] Output still looks wrong-language; forcing explicit English->target translation...")
                translate_segments_with_progress(
                    segments,
                    target_lang,
                    segment_progress_callback=translation_progress,
                    glossary_overrides=glossary_overrides,
                    translation_provider=translation_provider,
                    force_english_source=True,
                )
            if translation_looks_wrong_language(segments, target_lang):
                raise RuntimeError(
                    "Translation appears to remain in the wrong language for the selected target. "
                    "Please retry with --no-resume or switch translation provider."
                )
            save_segments_to_json(segments, segments_json)
        else:
            report("[resume] Reusing cached translations...")
            report_progress(0.60, f"Translating ({len(segments)}/{len(segments)})")

        deduped_segments = collapse_consecutive_duplicate_segments(segments)
        removed_count = len(segments) - len(deduped_segments)
        if removed_count > 0:
            report(f"[clean] Collapsed {removed_count} consecutive duplicate subtitle segment(s).")
            segments = deduped_segments
            save_segments_to_json(segments, segments_json)

        if export_srt:
            write_srt(segments, subtitle_path)
            report(f"[srt] Subtitle file written to: {subtitle_path}")

        report("[4/5] Generating dubbed track...")
        active_min_stretch = cast(float, resolved_settings["min_stretch_speed"])
        active_max_stretch = cast(float, resolved_settings["max_stretch_speed"])
        active_silence_trim = cast(int, resolved_settings["silence_trim_ms"])
        dub_signature = build_dub_cache_signature(
            segments,
            target_lang=target_lang,
            tts_engine=tts_engine,
            edge_voice=edge_voice,
            min_stretch_speed=active_min_stretch,
            max_stretch_speed=active_max_stretch,
            silence_trim_ms=active_silence_trim,
        )

        can_reuse_dubbed_audio = False
        if dubbed_wav.exists() and dub_meta_path.exists():
            try:
                dub_meta = json.loads(dub_meta_path.read_text(encoding="utf-8"))
                can_reuse_dubbed_audio = (
                    dub_meta.get("signature") == dub_signature
                    and has_meaningful_audio(dubbed_wav)
                )
            except Exception:
                can_reuse_dubbed_audio = False

        if can_reuse_dubbed_audio:
            report("[resume] Reusing cached dubbed audio...")
            report_progress(0.90, "Synthesizing voice (cached)")
        else:
            if dubbed_wav.exists():
                report("[resume] Dubbed audio cache is stale or silent; regenerating from current segments...")
            report_progress(0.62, "Generating neural voice")

            def tts_progress(done: int, total: int) -> None:
                start = 0.62
                end = 0.90
                fraction = done / max(total, 1)
                report_progress(start + (end - start) * fraction, f"Synthesizing voice ({done}/{total})")

            base_audio = AudioSegment.from_wav(extracted_wav)
            dubbed_track = build_dubbed_track(
                segments,
                len(base_audio),
                temp_base,
                target_lang,
                tts_engine=tts_engine,
                edge_voice=edge_voice,
                min_stretch_speed=active_min_stretch,
                max_stretch_speed=active_max_stretch,
                silence_trim_ms=active_silence_trim,
                segment_progress_callback=tts_progress,
                chunk_window_s=cast(float | None, resolved_settings["tts_chunk_window_s"]),
                cache_dir=tts_cache_dir,
            )
            dubbed_track.export(dubbed_wav, format="wav")
            if not has_meaningful_audio(dubbed_wav):
                raise RuntimeError(
                    "Generated dubbed audio is silent. "
                    "Try switching TTS engine/voice and rerun with --no-resume."
                )
            dub_meta_path.write_text(
                json.dumps({"signature": dub_signature}, indent=2),
                encoding="utf-8",
            )

        save_segments_to_json(segments, segments_json)

        report("[5/5] Muxing dubbed audio into video...")
        report_progress(0.92, "Muxing audio and video")
        if not include_original_audio:
            report("[mix] Original source audio disabled; exporting dubbed speech only.")
        mux_video_with_dub(
            working_video,
            dubbed_wav,
            output_path,
            background_mix_level=background_mix_level,
            include_original_audio=include_original_audio,
        )
        report_progress(1.0, "Completed")

        report(f"Done. Output written to: {output_path}")
        elapsed_s = time.perf_counter() - started_at
        elapsed_m, elapsed_rem_s = divmod(int(round(elapsed_s)), 60)
        report(f"Total processing time: {elapsed_m}m {elapsed_rem_s:02d}s")
        if keep_temp:
            report(f"Temp files kept at: {temp_base}")
        return 0
    finally:
        if not keep_temp:
            shutil.rmtree(temp_base, ignore_errors=True)


def main() -> int:
    args = parse_args()
    glossary_text = ""
    if args.glossary_file is not None:
        glossary_text = args.glossary_file.read_text(encoding="utf-8")

    return autodub_video(
        input_path=args.input,
        output_path=args.output,
        target_lang=args.target_lang,
        whisper_model=args.whisper_model,
        device=args.device,
        translation_provider=args.translation_provider,
        hf_token=args.hf_token,
        tts_engine=args.tts_engine,
        edge_voice=args.edge_voice,
        include_original_audio=not args.disable_original_audio,
        start_time_s=args.start_time,
        end_time_s=args.end_time,
        keep_temp=args.keep_temp,
        optimization_profile=args.optimization_profile,
        export_srt=not args.no_export_srt,
        resume_enabled=not args.no_resume,
        glossary_text=glossary_text,
        asr_engine=args.asr_engine,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        if sys.stderr is not None:
            print(f"Error: {exc}", file=sys.stderr)
        else:
            print(f"Error: {exc}")
        raise SystemExit(1)
