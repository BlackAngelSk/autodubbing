"""Coqui XTTS synthesis and language resolution."""

from __future__ import annotations

import importlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from autodub.config import (
    COQUI_DEFAULT_MODEL,
    COQUI_XTTS_SUPPORTED_LANGS,
    COQUI_LANGUAGE_ALIASES,
)
from autodub.segments import safe_text

logger = logging.getLogger(__name__)

_COQUI_MODEL_CACHE: dict[str, Any] = {}


def resolve_coqui_xtts_language(lang: str) -> str | None:
    normalized = safe_text(lang).strip().lower()
    normalized = COQUI_LANGUAGE_ALIASES.get(normalized, normalized)
    if normalized in COQUI_XTTS_SUPPORTED_LANGS:
        return normalized
    return None


def coqui_tts_segment(
    text: str,
    lang: str,
    output_mp3: Path,
    speaker_wav: Path | None = None,
) -> None:
    """Synthesize speech using Coqui XTTS model."""
    try:
        tts_api_module = importlib.import_module("TTS.api")
    except ImportError as exc:
        raise RuntimeError(
            "Coqui TTS engine is not installed. Install with: pip install TTS"
        ) from exc

    tts_class = getattr(tts_api_module, "TTS", None)
    if tts_class is None:
        raise RuntimeError("Coqui TTS import succeeded, but TTS.api.TTS is unavailable.")

    coqui_model = os.environ.get("AUTODUB_COQUI_MODEL", "").strip()
    if not coqui_model:
        coqui_model = COQUI_DEFAULT_MODEL

    tts_model = _COQUI_MODEL_CACHE.get(coqui_model)
    if tts_model is None:
        tts_model = tts_class(model_name=coqui_model, progress_bar=False)
        _COQUI_MODEL_CACHE[coqui_model] = tts_model

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        temp_wav_path = Path(tmp_wav.name)

    try:
        synth_kwargs: dict[str, Any] = {
            "text": text,
            "file_path": str(temp_wav_path),
        }

        lower_model_name = coqui_model.lower()
        if "xtts" in lower_model_name or "multilingual" in lower_model_name:
            resolved_lang = resolve_coqui_xtts_language(lang)
            if resolved_lang is None:
                raise RuntimeError(
                    f"Coqui XTTS does not support target language '{lang}'. "
                    "Use Edge TTS for this language or override AUTODUB_COQUI_MODEL."
                )
            synth_kwargs["language"] = resolved_lang
            env_speaker_wav = os.environ.get("AUTODUB_COQUI_SPEAKER_WAV", "").strip()
            speaker_name = os.environ.get("AUTODUB_COQUI_SPEAKER", "").strip()
            if speaker_wav is not None and speaker_wav.exists():
                synth_kwargs["speaker_wav"] = str(speaker_wav)
            elif env_speaker_wav:
                synth_kwargs["speaker_wav"] = env_speaker_wav
            elif speaker_name:
                synth_kwargs["speaker"] = speaker_name

        tts_model.tts_to_file(**synth_kwargs)
        AudioSegment.from_wav(temp_wav_path).export(output_mp3, format="mp3")
    finally:
        if temp_wav_path.exists():
            temp_wav_path.unlink(missing_ok=True)