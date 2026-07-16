"""Whisper-based ASR: model loading, transcription, segment merging, and recovery."""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path
from typing import Callable, List, cast

from faster_whisper import WhisperModel
from pydub import AudioSegment

from autodub.config import LARGE_WHISPER_MODELS
from autodub.device import (
    configure_hf_hub_access,
    cpu_fallback_whisper_model,
    is_cuda_runtime_error,
    preferred_whisper_compute_type,
    resolve_device_selection,
    whisper_compute_type_candidates,
)
from autodub.segments import Segment, safe_text
from autodub.ffmpeg_utils import run_cmd
from autodub.segments import save_segments_to_json, load_segments_from_json

logger = logging.getLogger(__name__)


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
            last_error, fallback_model,
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


def _normalize_text(text: str) -> str:
    lowered = safe_text(text).strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return re.sub(r"[^\w\s']+", "", lowered)


def _resolve_segment_bounds(seg: object) -> tuple[float, float]:
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


def _has_time_overlap(a: Segment, b: Segment, padding_s: float = 0.18) -> bool:
    return not (a.end_s < b.start_s - padding_s or b.end_s < a.start_s - padding_s)


def _is_same_text(a_text: str, b_text: str) -> bool:
    a_norm = _normalize_text(a_text)
    b_norm = _normalize_text(b_text)
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    shorter, longer = sorted((a_norm, b_norm), key=len)
    return len(shorter) >= 8 and shorter in longer


def merge_recall_segments(primary: List[Segment], secondary: List[Segment]) -> List[Segment]:
    merged = sorted(primary, key=lambda seg: seg.start_s)
    for cand in sorted(secondary, key=lambda seg: seg.start_s):
        if cand.end_s <= cand.start_s + 0.09:
            continue
        if len(safe_text(cand.source_text).strip()) < 2:
            continue

        overlaps = [existing for existing in merged if _has_time_overlap(existing, cand)]
        if not overlaps:
            merged.append(cand)
            continue

        if any(_is_same_text(existing.source_text, cand.source_text) for existing in overlaps):
            continue

        best = max(overlaps, key=lambda seg: seg.end_s - seg.start_s)
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


def collect_segments(
    target_audio_path: Path,
    model: WhisperModel,
    vad_filter: bool,
    relaxed: bool = False,
) -> List[Segment]:
    """Transcribe audio and return segments."""
    transcribe_kwargs: dict = {
        "vad_filter": vad_filter,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "beam_size": 1,
        "best_of": 1,
    }
    if relaxed:
        transcribe_kwargs["beam_size"] = 2
        transcribe_kwargs["best_of"] = 2
    if vad_filter:
        if relaxed:
            transcribe_kwargs["vad_parameters"] = {
                "threshold": 0.26, "neg_threshold": 0.18,
                "min_speech_duration_ms": 70, "min_silence_duration_ms": 140,
                "speech_pad_ms": 360,
            }
            transcribe_kwargs["no_speech_threshold"] = 0.92
        else:
            transcribe_kwargs["vad_parameters"] = {
                "threshold": 0.34, "neg_threshold": 0.22,
                "min_speech_duration_ms": 110, "min_silence_duration_ms": 260,
                "speech_pad_ms": 280,
            }
    if relaxed:
        transcribe_kwargs["no_speech_threshold"] = 0.95

    whisper_segments, _info = model.transcribe(str(target_audio_path), **transcribe_kwargs)

    collected: List[Segment] = []
    for seg in whisper_segments:
        text = safe_text(getattr(seg, "text", "")).strip()
        if not text:
            continue
        start_s, end_s = _resolve_segment_bounds(seg)
        if end_s <= start_s + 0.09:
            continue
        collected.append(Segment(start_s=start_s, end_s=end_s, source_text=text))
    return collected


def recover_tail_segments(existing_segments: List[Segment], source_audio_path: Path, collect_fn) -> List[Segment]:
    if not existing_segments:
        return existing_segments
    audio_duration_s = len(AudioSegment.from_wav(source_audio_path)) / 1000.0
    last_end_s = existing_segments[-1].end_s
    trailing_gap_s = audio_duration_s - last_end_s

    probe_starts: list[float] = []
    if trailing_gap_s >= 0.8:
        probe_starts.append(max(last_end_s - 2.5, 0.0))
    probe_starts.append(max(audio_duration_s - 12.0, 0.0))
    probe_starts = sorted(set(round(value, 3) for value in probe_starts))

    appended: List[Segment] = []
    seen_texts = {_normalize_text(seg.source_text) for seg in existing_segments[-6:]}
    last_text = _normalize_text(existing_segments[-1].source_text)
    seen_positions = [(round(seg.start_s, 2), _normalize_text(seg.source_text)) for seg in existing_segments]

    for probe_index, tail_start_s in enumerate(probe_starts, start=1):
        tail_audio = AudioSegment.from_wav(source_audio_path)[int(tail_start_s * 1000):]
        tail_path = source_audio_path.parent / f"tail_recheck_{probe_index}.wav"
        tail_audio.export(tail_path, format="wav")

        recovered_segments = collect_fn(tail_path)
        for seg in recovered_segments:
            adjusted = Segment(start_s=seg.start_s + tail_start_s, end_s=seg.end_s + tail_start_s, source_text=seg.source_text)
            if adjusted.end_s <= last_end_s + 0.20 and adjusted.start_s <= last_end_s + 0.20:
                continue
            normalized = _normalize_text(adjusted.source_text)
            if not normalized:
                continue
            if (round(adjusted.start_s, 2), normalized) in seen_positions:
                continue
            if normalized in seen_texts:
                continue
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


def recover_head_segments(existing_segments: List[Segment], source_audio_path: Path, collect_fn) -> List[Segment]:
    if not existing_segments:
        return existing_segments
    first_start_s = existing_segments[0].start_s
    head_probe_end_s = min(max(first_start_s + 2.0, 12.0), 20.0)
    head_audio = AudioSegment.from_wav(source_audio_path)[:int(head_probe_end_s * 1000)]
    head_path = source_audio_path.parent / "head_recheck.wav"
    head_audio.export(head_path, format="wav")

    recovered_segments = collect_fn(head_path)
    prepended: List[Segment] = []
    seen_texts = {_normalize_text(seg.source_text) for seg in existing_segments[:5]}

    for seg in recovered_segments:
        normalized = _normalize_text(seg.source_text)
        if not normalized or normalized in seen_texts:
            continue
        if seg.start_s > head_probe_end_s:
            continue
        if any(_has_time_overlap(seg, existing, padding_s=0.10) for existing in existing_segments[:6]):
            continue
        prepended.append(seg)
        seen_texts.add(normalized)

    if not prepended:
        return existing_segments
    return merge_recall_segments(existing_segments, prepended)


def recover_internal_gap_segments(existing_segments: List[Segment], source_audio_path: Path, collect_fn) -> List[Segment]:
    if len(existing_segments) < 2:
        return existing_segments
    audio_duration_s = len(AudioSegment.from_wav(source_audio_path)) / 1000.0

    candidate_windows: list[tuple[float, float, float]] = []
    for prev, nxt in zip(existing_segments, existing_segments[1:]):
        gap_s = nxt.start_s - prev.end_s
        if gap_s < 2.2:
            continue
        probe_start_s = max(prev.end_s - 0.35, 0.0)
        probe_end_s = min(nxt.start_s + 0.35, audio_duration_s)
        if probe_end_s <= probe_start_s + 0.9:
            continue
        candidate_windows.append((probe_start_s, probe_end_s, gap_s))

    if not candidate_windows:
        return existing_segments

    candidate_windows = sorted(candidate_windows, key=lambda item: item[2], reverse=True)[:10]
    candidate_windows = sorted(candidate_windows, key=lambda item: item[0])

    recovered: List[Segment] = []
    seen_positions = {(round(seg.start_s, 2), _normalize_text(seg.source_text)) for seg in existing_segments}

    for idx, (probe_start_s, probe_end_s, _gap_s) in enumerate(candidate_windows, start=1):
        probe_audio = AudioSegment.from_wav(source_audio_path)[int(probe_start_s * 1000):int(probe_end_s * 1000)]
        probe_path = source_audio_path.parent / f"gap_recheck_{idx}.wav"
        probe_audio.export(probe_path, format="wav")

        probe_segments = collect_fn(probe_path)
        for seg in probe_segments:
            adjusted = Segment(start_s=seg.start_s + probe_start_s, end_s=seg.end_s + probe_start_s, source_text=seg.source_text)
            if adjusted.end_s <= adjusted.start_s + 0.09:
                continue
            normalized = _normalize_text(adjusted.source_text)
            if not normalized:
                continue
            key = (round(adjusted.start_s, 2), normalized)
            if key in seen_positions:
                continue
            overlaps = [existing for existing in existing_segments if _has_time_overlap(existing, adjusted, padding_s=0.12)]
            if any(_is_same_text(existing.source_text, adjusted.source_text) for existing in overlaps):
                continue
            recovered.append(adjusted)
            seen_positions.add(key)

    if not recovered:
        return existing_segments
    return merge_recall_segments(existing_segments, recovered)


def transcribe_single_audio(
    target_audio_path: Path,
    model: WhisperModel,
) -> List[Segment]:
    """Transcribe a single audio file using Whisper with recall passes."""
    def collect_fn(path: Path) -> List[Segment]:
        return collect_segments(path, model, vad_filter=False)

    primary_segments = collect_segments(target_audio_path, model, vad_filter=True)
    recall_vad_segments = collect_segments(target_audio_path, model, vad_filter=True, relaxed=True)
    recall_segments = collect_segments(target_audio_path, model, vad_filter=False, relaxed=True)

    if primary_segments:
        merged_segments = merge_recall_segments(primary_segments, recall_vad_segments)
        merged_segments = merge_recall_segments(merged_segments, recall_segments)
        recovered = recover_tail_segments(merged_segments, target_audio_path, collect_fn)
        recovered = recover_head_segments(recovered, target_audio_path, collect_fn)
        recovered = recover_internal_gap_segments(recovered, target_audio_path, collect_fn)
        return merge_tts_friendly_segments(recovered)

    fallback_segments = recall_vad_segments or recall_segments
    if recall_vad_segments and recall_segments:
        fallback_segments = merge_recall_segments(recall_vad_segments, recall_segments)

    recovered = recover_tail_segments(fallback_segments, target_audio_path, collect_fn)
    recovered = recover_head_segments(recovered, target_audio_path, collect_fn)
    recovered = recover_internal_gap_segments(recovered, target_audio_path, collect_fn)
    return merge_tts_friendly_segments(recovered)


def transcribe_segments_whisper(
    audio_path: Path,
    model_name: str,
    device: str,
    chunk_length_s: float | None = None,
    cache_dir: Path | None = None,
    chunk_progress_callback: Callable[[int, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    hf_token: str | None = None,
) -> List[Segment]:
    """Transcribe audio using Whisper with chunked processing and recall passes."""
    resolved_device = resolve_device_selection(device)
    active_device = resolved_device
    active_model_name = model_name

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        full_cache_path = cache_dir / f"segments_{model_name}_{resolved_device}.json"
        if full_cache_path.exists():
            cached_segments = load_segments_from_json(full_cache_path)
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

    model = load_whisper_model(model_name, resolved_device, hf_token=hf_token, cpu_fallback_model=cpu_fallback_whisper_model(model_name))

    audio_duration_s = len(AudioSegment.from_wav(audio_path)) / 1000.0
    should_chunk = chunk_length_s is not None and audio_duration_s > max(chunk_length_s * 1.15, 90.0)

    if should_chunk and chunk_length_s is not None:
        merged_all: List[Segment] = []
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
                chunk_wav_path = cache_dir / f"asr_chunk_{chunk_index + 1:04d}.wav" if cache_dir is not None else audio_path.parent / f"asr_chunk_{chunk_index + 1:04d}.wav"
                run_cmd(["ffmpeg", "-y", "-i", str(audio_path), "-ss", f"{chunk_start_s:.3f}", "-to", f"{chunk_end_s:.3f}", "-acodec", "pcm_s16le", str(chunk_wav_path)])
                local_segments = transcribe_single_audio(chunk_wav_path, model)
                chunk_segments = [Segment(start_s=seg.start_s + chunk_start_s, end_s=seg.end_s + chunk_start_s, source_text=seg.source_text, translated_text=seg.translated_text) for seg in local_segments]
                if chunk_json_path is not None:
                    save_segments_to_json(chunk_segments, chunk_json_path)

            merged_all = merge_recall_segments(merged_all, chunk_segments) if merged_all else chunk_segments
            if chunk_progress_callback is not None:
                chunk_progress_callback(chunk_index + 1, total_chunks)

        if full_cache_path is not None:
            save_segments_to_json(merged_all, full_cache_path)
        return merged_all

    single_segments = transcribe_single_audio(audio_path, model)
    if full_cache_path is not None:
        save_segments_to_json(single_segments, full_cache_path)
    return single_segments