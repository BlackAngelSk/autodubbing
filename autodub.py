#!/usr/bin/env python3
"""Auto-dub a video by transcribing, translating, and re-synthesizing speech.

This file is a thin wrapper for backwards compatibility.
Please import from the autodub package directly.
"""

from __future__ import annotations

import sys

from autodub.pipeline import autodub_video  # noqa: F401
from autodub import (  # noqa: F401  - re-export all public symbols
    Segment,
    safe_text,
    format_srt_timestamp,
    write_srt,
    write_transcript_txt,
    normalize_subtitle_for_dedupe,
    collapse_consecutive_duplicate_segments,
    save_segments_to_json,
    load_segments_from_json,
    detect_cuda_available,
    detect_rocm_available,
    resolve_device_selection,
    is_cuda_runtime_error,
    ensure_ffmpeg,
    probe_media_duration,
    extract_audio,
    trim_video,
    run_cmd,
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
    edge_tts_segment,
    edge_human_tts_segment,
    _parse_edge_rate_percent,
    _parse_edge_pitch_hz,
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
    tts_segment,
    build_dub_cache_signature,
    build_resume_dir,
    build_dubbed_track,
    mux_video_with_dub,
)
# Re-import config constants that were previously at module level
from autodub.config import (  # noqa: F401
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
# Re-export the energy module (referenced by ui.py). We create a thin shim so
# that old-style `autodub.energy` imports keep working.
from autodub.energy import (  # noqa: F401
    normalize_video_energy,
    adjust_segment_energy,
    detect_meaningful_energy,
)

from autodub.cli import main as _cli_main  # noqa: F401


def main() -> None:
    """CLI entry point kept for backwards compatibility."""
    sys.exit(_cli_main())


if __name__ == "__main__":
    main()