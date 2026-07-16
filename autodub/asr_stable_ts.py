"""stable-ts based ASR: transcription with accurate timestamps and improved coverage."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, List

from pydub import AudioSegment

from autodub.device import (
    configure_hf_hub_access,
    cpu_fallback_whisper_model,
    is_cuda_runtime_error,
    preferred_whisper_compute_type,
    resolve_device_selection,
)
from autodub.segments import Segment, safe_text
from autodub.ffmpeg_utils import run_cmd
from autodub.segments import save_segments_to_json, load_segments_from_json

logger = logging.getLogger(__name__)


def stable_ts_available() -> bool:
    """Return True if stable-ts is installed and importable."""
    try:
        importlib.import_module("stable_whisper")
        return True
    except ImportError:
        return False


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
                status_callback(f"[stable-ts] CUDA unavailable ({exc}). Retrying on CPU with '{fallback_model}'...")
            logger.info("stable-ts CUDA load failed (%s). Retrying on CPU with '%s'.", exc, fallback_model)
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
                    status_callback(f"[stable-ts] CUDA runtime error during transcription. Retrying on CPU with '{fallback_model}'...")
                logger.info("stable-ts CUDA transcription failed (%s). Retrying on CPU with '%s'.", exc, fallback_model)
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
        from autodub.asr_whisper import merge_recall_segments
        all_segments = merge_recall_segments(all_segments, [])  # dedup at boundaries
        if full_cache_path is not None:
            save_segments_to_json(all_segments, full_cache_path)
        return all_segments

    segments = transcribe_chunk(audio_path)
    if full_cache_path is not None:
        save_segments_to_json(segments, full_cache_path)
    return segments