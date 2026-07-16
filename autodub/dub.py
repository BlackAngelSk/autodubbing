"""Build the dubbed audio track from segments using TTS synthesis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from pydub import AudioSegment

from autodub.segments import Segment, safe_text
from autodub.tts import tts_segment
from autodub.tts_postprocess import (
    build_edge_tts_profile,
    build_page_tts_profile,
    format_edge_rate,
    format_edge_pitch,
    fit_audio_to_duration_with_controls,
    enhance_tts_audio,
)

logger = logging.getLogger(__name__)


def build_dubbed_track(
    segments: list[Segment],
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
    coqui_speaker_wav: Path | None = None,
    use_page_tts_profile: bool = True,
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
            spoken_text, target_lang, mp3_path,
            tts_engine=tts_engine, edge_voice=edge_voice,
            edge_rate=rate, edge_pitch=pitch, edge_volume=volume,
            coqui_speaker_wav=coqui_speaker_wav,
        )
        cached_audio = AudioSegment.from_file(mp3_path)
        cached_audio = enhance_tts_audio(cached_audio, tts_engine)
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

            if tts_engine in {"edge", "edge_human"} and use_page_tts_profile:
                spoken_text, base_rate_percent, pitch_hz, volume = build_page_tts_profile(line_text)
            else:
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
            edge_should_adapt_rate = (
                tts_engine == "edge_human" or (tts_engine == "edge" and not use_page_tts_profile)
            )
            if edge_should_adapt_rate and slot_ms > 0:
                desired_ratio = (
                    len(
                        get_tts_audio(
                            spoken_text,
                            rate=format_edge_rate(base_rate_percent),
                            pitch=format_edge_pitch(pitch_hz),
                            volume=volume,
                        )
                    )
                    / slot_ms
                )
                if desired_ratio > 1.08:
                    edge_rate_percent += int(min((desired_ratio - 1.0) * 18, 10))
                elif desired_ratio < 0.74:
                    edge_rate_percent -= int(min((1.0 - desired_ratio) * 8, 4))

                if tts_engine == "edge_human":
                    edge_rate_percent = max(min(edge_rate_percent, 20), -18)
                else:
                    edge_rate_percent = max(min(edge_rate_percent, 12), -12)

            edge_rate = format_edge_rate(edge_rate_percent)
            edge_pitch = format_edge_pitch(pitch_hz)
            voice = get_tts_audio(spoken_text, rate=edge_rate, pitch=edge_pitch, volume=volume)

            if edge_should_adapt_rate and slot_ms > 0:
                natural_ratio = len(voice) / slot_ms
                if natural_ratio > max_stretch_speed + 0.12:
                    adaptive_bump = int(min((natural_ratio - 1.0) * 10, 8))
                    if tts_engine == "edge_human":
                        adaptive_rate = format_edge_rate(max(min(edge_rate_percent + adaptive_bump, 22), -18))
                    else:
                        adaptive_rate = format_edge_rate(max(min(edge_rate_percent + adaptive_bump, 14), -12))
                    voice = get_tts_audio(spoken_text, rate=adaptive_rate, pitch=edge_pitch, volume=volume)

            voice = fit_audio_to_duration_with_controls(
                voice, slot_ms, temp_dir, i,
                min_stretch_speed=max(min_stretch_speed, 0.93) if tts_engine == "edge_human" else min_stretch_speed,
                max_stretch_speed=min(max_stretch_speed, 1.18) if tts_engine == "edge_human" else max_stretch_speed,
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