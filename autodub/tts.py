"""Unified TTS dispatch: routes text to edge, edge_human, gtts, or coqui engines."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from gtts import gTTS

from autodub.config import DEFAULT_EDGE_VOICES
from autodub.tts_coqui import coqui_tts_segment
from autodub.tts_edge import edge_tts_segment, edge_human_tts_segment
from autodub.segments import safe_text

logger = logging.getLogger(__name__)


def tts_segment(
    text: str,
    lang: str,
    output_mp3: Path,
    tts_engine: str = "edge",
    edge_voice: str | None = None,
    edge_rate: str = "+0%",
    edge_pitch: str = "+0Hz",
    edge_volume: str = "+0%",
    coqui_speaker_wav: Path | None = None,
) -> None:
    last_error: Exception | None = None
    coqui_failed = False

    if tts_engine == "coqui":
        for attempt in range(2):
            try:
                coqui_tts_segment(text, lang, output_mp3, speaker_wav=coqui_speaker_wav)
                return
            except Exception as exc:
                coqui_failed = True
                last_error = exc
                time.sleep(0.18 * (attempt + 1))

        fallback_voice = edge_voice or DEFAULT_EDGE_VOICES.get(lang)
        if fallback_voice:
            for attempt in range(2):
                try:
                    edge_tts_segment(
                        text, fallback_voice, output_mp3,
                        rate=edge_rate, pitch=edge_pitch, volume=edge_volume,
                    )
                    return
                except Exception as exc:
                    last_error = exc
                    time.sleep(0.18 * (attempt + 1))

    if tts_engine == "edge":
        voice = edge_voice or DEFAULT_EDGE_VOICES.get(lang, DEFAULT_EDGE_VOICES["en"])
        for attempt in range(3):
            try:
                edge_tts_segment(
                    text, voice, output_mp3,
                    rate=edge_rate, pitch=edge_pitch, volume=edge_volume,
                )
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.18 * (attempt + 1))

    if tts_engine == "edge_human":
        voice = edge_voice or DEFAULT_EDGE_VOICES.get(lang, DEFAULT_EDGE_VOICES["en"])
        for attempt in range(3):
            try:
                edge_human_tts_segment(
                    text, voice, output_mp3,
                    rate=edge_rate, pitch=edge_pitch, volume=edge_volume,
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

    if coqui_failed:
        raise RuntimeError(
            "Coqui synthesis failed and fallback engines did not succeed. "
            f"Last error: {last_error}"
        )
    raise RuntimeError(f"TTS synthesis failed after retries: {last_error}")