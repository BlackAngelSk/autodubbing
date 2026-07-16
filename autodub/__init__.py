"""Auto-dubbing pipeline: transcribe, translate, and re-synthesize speech in videos."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Iterable

from autodub.config import (
    DEFAULT_EDGE_VOICES,
    LARGE_WHISPER_MODELS,
    TRANSLATION_PROVIDERS,
    ASR_ENGINE_CHOICES,
    COQUI_DEFAULT_MODEL,
    COQUI_XTTS_SUPPORTED_LANGS,
    COQUI_LANGUAGE_ALIASES,
    TTS_POSTPROCESS_VERSION,
    HF_UNAUTH_WARNING_TEXT,
)
from autodub.segments import (
    Segment,
    safe_text,
    format_srt_timestamp,
    write_srt,
    write_transcript_txt,
    normalize_subtitle_for_dedupe,
    collapse_consecutive_duplicate_segments,
    save_segments_to_json,
    load_segments_from_json,
)
from autodub.device import (
    detect_cuda_available,
    detect_rocm_available,
    resolve_device_selection,
    is_cuda_runtime_error,
)
from autodub.ffmpeg_utils import (
    ensure_ffmpeg,
    probe_media_duration,
    extract_audio,
    trim_video,
    run_cmd,
)
from autodub.translate import (
    build_translator,
    parse_glossary_overrides,
    apply_glossary_overrides,
    safe_translate,
    split_for_translation,
    translate_segments_with_progress,
    cached_translation_looks_poor,
    translation_looks_wrong_language,
    english_word_tokens,
    has_untranslated_english_tokens,
    replace_untranslated_tokens,
)
from autodub.tts_edge import (
    edge_tts_segment,
    edge_human_tts_segment,
    _parse_edge_rate_percent,
    _parse_edge_pitch_hz,
)
from autodub.tts_postprocess import (
    sanitize_tts_text,
    format_edge_rate,
    format_edge_pitch,
    inject_mid_sentence_pause,
    build_edge_tts_profile,
    build_page_tts_profile,
    build_atempo_filter,
    stretch_audio_preserve_pitch,
    trim_segment_silence,
    trim_initial_tts_latency,
    fit_audio_to_duration_with_controls,
    has_meaningful_audio,
    enhance_coqui_audio,
    soften_sibilance,
    enhance_tts_audio,
    post_process_dubbed_track,
)
from autodub.tts import tts_segment
from autodub.cache import (
    build_dub_cache_signature,
    build_resume_dir,
)
from autodub.dub import build_dubbed_track
from autodub.mux import mux_video_with_dub
from autodub.pipeline import autodub_video
from autodub.energy import (
    normalize_video_energy,
    adjust_segment_energy,
    detect_meaningful_energy,
)

__all__ = [
    "DEFAULT_EDGE_VOICES",
    "LARGE_WHISPER_MODELS",
    "TRANSLATION_PROVIDERS",
    "ASR_ENGINE_CHOICES",
    "COQUI_DEFAULT_MODEL",
    "COQUI_XTTS_SUPPORTED_LANGS",
    "COQUI_LANGUAGE_ALIASES",
    "TTS_POSTPROCESS_VERSION",
    "HF_UNAUTH_WARNING_TEXT",
    "Segment",
    "safe_text",
    "format_srt_timestamp",
    "write_srt",
    "write_transcript_txt",
    "normalize_subtitle_for_dedupe",
    "collapse_consecutive_duplicate_segments",
    "save_segments_to_json",
    "load_segments_from_json",
    "detect_cuda_available",
    "detect_rocm_available",
    "resolve_device_selection",
    "is_cuda_runtime_error",
    "ensure_ffmpeg",
    "probe_media_duration",
    "extract_audio",
    "trim_video",
    "run_cmd",
    "build_translator",
    "parse_glossary_overrides",
    "apply_glossary_overrides",
    "safe_translate",
    "split_for_translation",
    "translate_segments_with_progress",
    "cached_translation_looks_poor",
    "translation_looks_wrong_language",
    "english_word_tokens",
    "has_untranslated_english_tokens",
    "replace_untranslated_tokens",
    "edge_tts_segment",
    "edge_human_tts_segment",
    "_parse_edge_rate_percent",
    "_parse_edge_pitch_hz",
    "sanitize_tts_text",
    "format_edge_rate",
    "format_edge_pitch",
    "inject_mid_sentence_pause",
    "build_edge_tts_profile",
    "build_page_tts_profile",
    "build_atempo_filter",
    "stretch_audio_preserve_pitch",
    "trim_segment_silence",
    "trim_initial_tts_latency",
    "fit_audio_to_duration_with_controls",
    "has_meaningful_audio",
    "enhance_coqui_audio",
    "soften_sibilance",
    "enhance_tts_audio",
    "post_process_dubbed_track",
    "tts_segment",
    "build_dub_cache_signature",
    "build_resume_dir",
    "build_dubbed_track",
    "mux_video_with_dub",
    "autodub_video",
    "normalize_video_energy",
    "adjust_segment_energy",
    "detect_meaningful_energy",
]
