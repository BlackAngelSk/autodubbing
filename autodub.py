#!/usr/bin/env python3
"""Auto-dub a video by transcribing, translating, and re-synthesizing speech."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, cast

from deep_translator import GoogleTranslator
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


@dataclass
class Segment:
    start_s: float
    end_s: float
    source_text: str
    translated_text: str = ""


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


def transcribe_segments(audio_path: Path, model_name: str, device: str) -> List[Segment]:
    model = WhisperModel(model_name, device=device, compute_type="int8")

    def normalize_text(text: str) -> str:
        lowered = text.strip().lower()
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
            if len(cand.source_text.strip()) < 2:
                continue

            overlaps = [existing for existing in merged if has_time_overlap(existing, cand)]
            if not overlaps:
                merged.append(cand)
                continue

            if any(is_same_text(existing.source_text, cand.source_text) for existing in overlaps):
                continue

            # Keep richer relaxed segment when it clearly carries more content.
            best = max(overlaps, key=lambda seg: seg.end_s - seg.start_s)
            best_dur = best.end_s - best.start_s
            cand_dur = cand.end_s - cand.start_s
            if cand_dur >= best_dur * 1.35 and len(cand.source_text) >= len(best.source_text) + 8:
                merged.remove(best)
                merged.append(cand)

        return sorted(merged, key=lambda seg: seg.start_s)

    def collect_segments(target_audio_path: Path, vad_filter: bool, relaxed: bool = False) -> List[Segment]:
        transcribe_kwargs: dict[str, Any] = {
            "vad_filter": vad_filter,
            "word_timestamps": True,
            "condition_on_previous_text": False,
        }
        if relaxed:
            # Favor recall for quiet/overlapped speech in the second pass.
            transcribe_kwargs["no_speech_threshold"] = 0.92

        whisper_segments, _info = model.transcribe(
            str(target_audio_path),
            **transcribe_kwargs,
        )
        collected: List[Segment] = []
        for seg in whisper_segments:
            text = seg.text.strip()
            if not text:
                continue
            start_s, end_s = resolve_segment_bounds(seg)
            if end_s <= start_s + 0.09:
                continue
            collected.append(Segment(start_s=start_s, end_s=end_s, source_text=text))
        return collected

    def recover_tail_segments(existing_segments: List[Segment]) -> List[Segment]:
        if not existing_segments:
            return existing_segments

        audio_duration_s = len(AudioSegment.from_wav(audio_path)) / 1000.0
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
            tail_audio = AudioSegment.from_wav(audio_path)[int(tail_start_s * 1000) :]
            tail_path = audio_path.parent / f"tail_recheck_{probe_index}.wav"
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

    primary_segments = collect_segments(audio_path, vad_filter=True)
    recall_segments = collect_segments(audio_path, vad_filter=False, relaxed=True)

    if primary_segments:
        merged_segments = merge_recall_segments(primary_segments, recall_segments)
        return recover_tail_segments(merged_segments)

    # If VAD pass found nothing, use relaxed no-VAD pass.
    return recover_tail_segments(recall_segments)


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
        replacement = word_translator.translate(token)
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
    translator: GoogleTranslator,
    fallback_translator: GoogleTranslator,
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


def translate_segments(segments: Iterable[Segment], target_lang: str) -> None:
    translator = GoogleTranslator(source="auto", target=target_lang)
    for seg in tqdm(segments, desc="Translating"):
        seg.translated_text = translator.translate(seg.source_text)


def translate_segments_with_progress(
    segments: List[Segment],
    target_lang: str,
    segment_progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    translator = GoogleTranslator(source="auto", target=target_lang)
    fallback_translator = GoogleTranslator(source="en", target=target_lang)
    word_translator = GoogleTranslator(source="en", target=target_lang)

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

    total = len(segments)
    unchanged_count = 0
    for idx, seg in enumerate(tqdm(segments, desc="Translating"), start=1):
        translated = safe_translate(seg.source_text, translator, fallback_translator)

        if should_retry_with_english_source(seg.source_text, translated) or has_untranslated_english_tokens(
            seg.source_text, translated, target_lang
        ):
            retry = safe_translate(seg.source_text, fallback_translator, translator)
            if retry is not None and retry.strip():
                translated = retry

        if has_untranslated_english_tokens(seg.source_text, translated, target_lang):
            translated = replace_untranslated_tokens(seg.source_text, translated, word_translator)

        if target_lang != "en":
            source_clean = re.sub(r"\s+", " ", seg.source_text).strip().lower()
            translated_clean = re.sub(r"\s+", " ", translated).strip().lower()
            if source_clean and source_clean == translated_clean:
                unchanged_count += 1

        seg.translated_text = translated
        if segment_progress_callback is not None:
            segment_progress_callback(idx, total)

    if target_lang != "en" and total > 0 and unchanged_count / total > 0.45:
        raise RuntimeError(
            "Translation provider returned too many unchanged lines. "
            "Try again in a minute or use a different target language code."
        )


def edge_tts_segment(text: str, voice: str, output_mp3: Path, rate: str = "+0%") -> None:
    edge_tts = importlib.import_module("edge_tts")

    async def synthesize() -> None:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        await communicate.save(str(output_mp3))

    asyncio.run(synthesize())


def tts_segment(
    text: str,
    lang: str,
    output_mp3: Path,
    tts_engine: str = "edge",
    edge_voice: str | None = None,
    edge_rate: str = "+0%",
) -> None:
    if tts_engine == "edge":
        voice = edge_voice or DEFAULT_EDGE_VOICES.get(lang, DEFAULT_EDGE_VOICES["en"])
        try:
            edge_tts_segment(text, voice, output_mp3, rate=edge_rate)
            return
        except Exception:
            # Fallback keeps pipeline working if Edge service is unavailable.
            pass

    tts = gTTS(text=text, lang=lang)
    tts.save(str(output_mp3))


def sanitize_tts_text(text: str) -> str:
    """Clean text so TTS gets stable, natural input."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"([!?.,])\1{1,}", r"\1", cleaned)
    return cleaned


def format_edge_rate(percent: int) -> str:
    bounded = max(min(percent, 40), -50)
    if bounded >= 0:
        return f"+{bounded}%"
    return f"{bounded}%"


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


def fit_audio_to_duration(audio: AudioSegment, target_ms: int, temp_dir: Path, segment_index: int) -> AudioSegment:
    """Fit audio into target slot while avoiding robotic speed and pitch artifacts."""
    if target_ms <= 0:
        return AudioSegment.silent(duration=0)

    current_ms = len(audio)
    if current_ms <= 0:
        return AudioSegment.silent(duration=target_ms)

    required_speed = current_ms / max(target_ms, 1)
    clamped_speed = min(max(required_speed, 0.85), 1.20)

    if abs(clamped_speed - 1.0) > 0.03:
        audio = stretch_audio_preserve_pitch(audio, clamped_speed, temp_dir, f"seg_{segment_index:05d}")

    if len(audio) > target_ms:
        fade_ms = min(80, max(target_ms // 6, 20))
        clipped = cast(AudioSegment, audio[:target_ms])
        return clipped.fade_out(fade_ms)
    if len(audio) < target_ms:
        return audio + AudioSegment.silent(duration=target_ms - len(audio))
    return audio


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


def fit_audio_to_duration_with_controls(
    audio: AudioSegment,
    target_ms: int,
    temp_dir: Path,
    segment_index: int,
    min_stretch_speed: float,
    max_stretch_speed: float,
    silence_trim_ms: int,
) -> AudioSegment:
    if silence_trim_ms > 0:
        audio = trim_segment_silence(audio, silence_trim_ms)

    if target_ms <= 0:
        return AudioSegment.silent(duration=0)

    current_ms = len(audio)
    if current_ms <= 0:
        return AudioSegment.silent(duration=target_ms)

    required_speed = current_ms / max(target_ms, 1)
    clamped_speed = min(max(required_speed, min_stretch_speed), max_stretch_speed)

    if abs(clamped_speed - 1.0) > 0.03:
        audio = stretch_audio_preserve_pitch(audio, clamped_speed, temp_dir, f"seg_{segment_index:05d}")

    # Safety pass: if audio is still too long, allow one extra stretch to avoid dropping words.
    if len(audio) > target_ms:
        overflow_speed = len(audio) / max(target_ms, 1)
        if overflow_speed > 1.03:
            # Keep emergency compression bounded so a single difficult line does
            # not become unnaturally fast.
            safety_speed = min(max(overflow_speed, 1.0), max(max_stretch_speed + 0.35, 2.4))
            if abs(safety_speed - 1.0) > 0.03:
                audio = stretch_audio_preserve_pitch(audio, safety_speed, temp_dir, f"seg_{segment_index:05d}_safe")

    if len(audio) > target_ms:
        fade_ms = min(80, max(target_ms // 6, 20))
        clipped = cast(AudioSegment, audio[:target_ms])
        return clipped.fade_out(fade_ms)
    if len(audio) < target_ms:
        return audio + AudioSegment.silent(duration=target_ms - len(audio))
    return audio


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
) -> AudioSegment:
    dubbed = AudioSegment.silent(duration=total_duration_ms)
    total = len(segments)

    for i, seg in enumerate(tqdm(segments, desc="Generating TTS"), start=1):
        if not seg.translated_text:
            if segment_progress_callback is not None:
                segment_progress_callback(i, total)
            continue

        spoken_text = sanitize_tts_text(seg.translated_text)
        if not spoken_text:
            continue

        mp3_path = temp_dir / f"tts_{i:05d}.mp3"
        start_ms = max(int(seg.start_s * 1000), 0)
        original_end_ms = max(int(seg.end_s * 1000), start_ms + 120)
        if i < total:
            next_start_ms = max(int(segments[i].start_s * 1000) - 60, original_end_ms)
            allowed_end_ms = min(next_start_ms, total_duration_ms)
        else:
            allowed_end_ms = total_duration_ms
        slot_ms = max(allowed_end_ms - start_ms, 120)

        tts_segment(
            spoken_text,
            target_lang,
            mp3_path,
            tts_engine=tts_engine,
            edge_voice=edge_voice,
        )

        voice = AudioSegment.from_file(mp3_path)

        if tts_engine == "edge" and slot_ms > 0:
            natural_ratio = len(voice) / slot_ms
            if natural_ratio > max_stretch_speed + 0.12:
                adaptive_rate = format_edge_rate(int(min((natural_ratio - 1.0) * 55, 32)))
                tts_segment(
                    spoken_text,
                    target_lang,
                    mp3_path,
                    tts_engine=tts_engine,
                    edge_voice=edge_voice,
                    edge_rate=adaptive_rate,
                )
                voice = AudioSegment.from_file(mp3_path)

        if len(voice) > slot_ms and i < total:
            overflow_ms = len(voice) - slot_ms
            if overflow_ms > 0:
                slot_ms = min(slot_ms + min(overflow_ms, 180), total_duration_ms - start_ms)

        voice = fit_audio_to_duration_with_controls(
            voice,
            slot_ms,
            temp_dir,
            i,
            min_stretch_speed=min_stretch_speed,
            max_stretch_speed=max_stretch_speed,
            silence_trim_ms=silence_trim_ms,
        )
        dubbed = dubbed.overlay(voice, position=start_ms)
        if segment_progress_callback is not None:
            segment_progress_callback(i, total)

    return dubbed


def mux_video_with_dub(
    input_video: Path,
    dubbed_wav: Path,
    output_video: Path,
    background_mix_level: float = 0.08,
) -> None:
    # Keep original ambience quietly under dubbed speech for more natural output.
    mix = min(max(background_mix_level, 0.0), 1.0)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(dubbed_wav),
            "-filter_complex",
            f"[0:a]volume={mix:.3f}[bg];[bg][1:a]amix=inputs=2:duration=first:normalize=0[aout]",
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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-dub a video to a target language")
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument("--output", required=True, type=Path, help="Output dubbed video path")
    parser.add_argument("--target-lang", required=True, help="Target language code, e.g. es, hi, fr")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--tts-engine", default="edge", choices=["edge", "gtts"], help="Speech synthesis engine")
    parser.add_argument("--edge-voice", default=None, help="Edge voice name, e.g. en-US-AriaNeural")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds for dubbing window")
    parser.add_argument("--end-time", type=float, default=None, help="Optional end time in seconds for dubbing window")
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate files")
    return parser.parse_args()


def autodub_video(
    input_path: Path,
    output_path: Path,
    target_lang: str,
    whisper_model: str = "small",
    device: str = "auto",
    tts_engine: str = "edge",
    edge_voice: str | None = None,
    background_mix_level: float = 0.08,
    min_stretch_speed: float = 0.85,
    max_stretch_speed: float = 1.80,
    silence_trim_ms: int = 0,
    start_time_s: float = 0.0,
    end_time_s: float | None = None,
    keep_temp: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    progress_percent_callback: Callable[[float, str], None] | None = None,
) -> int:
    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    def report_progress(value: float, label: str) -> None:
        if progress_percent_callback is not None:
            bounded = min(max(value, 0.0), 1.0)
            progress_percent_callback(bounded, label)

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

    temp_base = Path(tempfile.mkdtemp(prefix="autodub_"))
    try:
        working_video = input_path
        extracted_wav = temp_base / "extracted.wav"
        dubbed_wav = temp_base / "dubbed.wav"
        segments_json = temp_base / "segments.json"

        if start_time_s > 0 or end_time_s is not None:
            report("[0/5] Trimming selected video range...")
            report_progress(0.01, "Preparing selected time range")
            clipped_video = temp_base / "trimmed_input.mp4"
            trim_video(input_path, clipped_video, start_time_s, end_time_s)
            working_video = clipped_video

        report("[1/5] Extracting audio...")
        report_progress(0.03, "Extracting audio")
        extract_audio(working_video, extracted_wav)
        report_progress(0.10, "Audio extracted")

        report("[2/5] Transcribing with Whisper...")
        report_progress(0.12, "Transcribing speech")
        segments = transcribe_segments(extracted_wav, whisper_model, device)
        if not segments:
            raise RuntimeError(
                "No speech segments found in the selected range. "
                "Try a different time window, use a larger Whisper model, or increase spoken content."
            )
        report_progress(0.35, f"Transcription complete ({len(segments)} segments)")

        report("[3/5] Translating segments...")

        def translation_progress(done: int, total: int) -> None:
            start = 0.35
            end = 0.60
            fraction = done / max(total, 1)
            report_progress(start + (end - start) * fraction, f"Translating ({done}/{total})")

        translate_segments_with_progress(segments, target_lang, segment_progress_callback=translation_progress)

        report("[4/5] Generating dubbed track...")
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
            min_stretch_speed=min_stretch_speed,
            max_stretch_speed=max_stretch_speed,
            silence_trim_ms=silence_trim_ms,
            segment_progress_callback=tts_progress,
        )
        dubbed_track.export(dubbed_wav, format="wav")

        segments_json.write_text(json.dumps([asdict(s) for s in segments], indent=2), encoding="utf-8")

        report("[5/5] Muxing dubbed audio into video...")
        report_progress(0.92, "Muxing audio and video")
        mux_video_with_dub(
            working_video,
            dubbed_wav,
            output_path,
            background_mix_level=background_mix_level,
        )
        report_progress(1.0, "Completed")

        report(f"Done. Output written to: {output_path}")
        if keep_temp:
            report(f"Temp files kept at: {temp_base}")
        return 0
    finally:
        if not keep_temp:
            shutil.rmtree(temp_base, ignore_errors=True)


def main() -> int:
    args = parse_args()
    return autodub_video(
        input_path=args.input,
        output_path=args.output,
        target_lang=args.target_lang,
        whisper_model=args.whisper_model,
        device=args.device,
        tts_engine=args.tts_engine,
        edge_voice=args.edge_voice,
        start_time_s=args.start_time,
        end_time_s=args.end_time,
        keep_temp=args.keep_temp,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
