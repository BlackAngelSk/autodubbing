"""Edge TTS synthesis: edge-tts and edge_human (sentence-level prosody)."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import cast

from pydub import AudioSegment

from autodub.segments import safe_text
from autodub.tts_postprocess import sanitize_tts_text, format_edge_rate, format_edge_pitch

logger = logging.getLogger(__name__)


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

    def patched_default_exception_handler(self: asyncio.BaseEventLoop, context: dict) -> None:
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


def _parse_edge_rate_percent(rate: str) -> int:
    match = re.match(r"\s*([+-]?\d+)\s*%\s*$", safe_text(rate))
    if not match:
        return 0
    return int(match.group(1))


def _parse_edge_pitch_hz(pitch: str) -> int:
    match = re.match(r"\s*([+-]?\d+)\s*hz\s*$", safe_text(pitch).lower())
    if not match:
        return 0
    return int(match.group(1))


def edge_human_tts_segment(
    text: str,
    voice: str,
    output_mp3: Path,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
) -> None:
    """Synthesize sentence-by-sentence with gentle prosody variation for a more human feel."""
    cleaned = sanitize_tts_text(text)
    if not cleaned:
        raise RuntimeError("No text available for edge_human synthesis")

    raw_sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?;:])\s+", cleaned) if chunk.strip()]
    sentences: list[str] = []
    for sentence in raw_sentences:
        if len(sentence) <= 130:
            sentences.append(sentence)
            continue
        clauses = [part.strip() for part in re.split(r"(?<=,)\s+", sentence) if part.strip()]
        if len(clauses) <= 1:
            sentences.append(sentence)
        else:
            sentences.extend(clauses)

    if len(sentences) <= 1:
        edge_tts_segment(cleaned, voice, output_mp3, rate=rate, pitch=pitch, volume=volume)
        return

    base_rate = _parse_edge_rate_percent(rate)
    base_pitch = _parse_edge_pitch_hz(pitch)

    with tempfile.TemporaryDirectory(prefix="edge_human_") as tmp_dir:
        temp_dir = Path(tmp_dir)
        combined = AudioSegment.silent(duration=0)

        total = len(sentences)
        for index, sentence in enumerate(sentences, start=1):
            words = len(sentence.split())
            local_rate = base_rate
            local_pitch = base_pitch

            if words >= 22:
                local_rate -= 4
            elif words >= 14:
                local_rate -= 2
            elif words <= 4:
                local_rate += 1

            if sentence.endswith("?"):
                local_pitch += 3
            elif sentence.endswith("!"):
                local_pitch += 2
            elif index % 3 == 0:
                local_pitch += 1

            if sentence.endswith(".") and words >= 8:
                local_pitch -= 1

            local_rate = max(min(local_rate, 18), -18)
            local_pitch = max(min(local_pitch, 10), -8)

            sentence_mp3 = temp_dir / f"sentence_{index:04d}.mp3"
            edge_tts_segment(
                sentence, voice, sentence_mp3,
                rate=format_edge_rate(local_rate),
                pitch=format_edge_pitch(local_pitch),
                volume=volume,
            )

            piece = AudioSegment.from_file(sentence_mp3)
            if len(combined) > 0:
                pause_ms = 170
                if sentence.endswith("?") or sentence.endswith("!"):
                    pause_ms = 210
                elif sentence.endswith(","):
                    pause_ms = 120
                elif words <= 5:
                    pause_ms = 130

                crossfade_ms = min(36, max(len(piece) // 35, 14), max(len(combined) // 35, 14))
                combined = combined.append(piece, crossfade=crossfade_ms)
                combined += AudioSegment.silent(duration=pause_ms)
            else:
                combined += piece

            if index < total and index % 4 == 0:
                combined += AudioSegment.silent(duration=110)

        combined.export(output_mp3, format="mp3")