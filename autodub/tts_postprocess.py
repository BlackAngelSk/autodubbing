"""TTS audio post-processing: text sanitization, audio enhancement, stretching, and fitting."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import cast

from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, high_pass_filter, low_pass_filter, normalize
from pydub.silence import detect_nonsilent

from autodub.segments import safe_text

logger = logging.getLogger(__name__)


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
    if word_count >= 20:
        rate_percent = -6
    elif word_count >= 13:
        rate_percent = -4
    elif word_count >= 8:
        rate_percent = -2

    if punctuation_count >= 2:
        rate_percent -= 1

    pitch_hz = 0
    if spoken_text.endswith("?"):
        pitch_hz = 5
    elif spoken_text.endswith("!"):
        pitch_hz = 3
    elif word_count <= 4:
        pitch_hz = 1

    return spoken_text, rate_percent, pitch_hz, "+0%"


def build_page_tts_profile(text: str) -> tuple[str, int, int, str]:
    """Match the UI Text-to-Speech tab defaults for more natural speech."""
    spoken_text = sanitize_tts_text(text)
    return spoken_text, 0, 0, "+0%"


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

    from autodub.ffmpeg_utils import run_cmd

    run_cmd([
        "ffmpeg", "-y", "-i", str(in_wav),
        "-filter:a", build_atempo_filter(speed),
        str(out_wav),
    ])

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

    if len(audio) > target_ms:
        overflow_speed = len(audio) / max(target_ms, 1)
        if overflow_speed > 1.08:
            safety_speed = min(max(overflow_speed, 1.0), max(max_stretch_speed + 0.08, 1.42))
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


def enhance_coqui_audio(voice: AudioSegment) -> AudioSegment:
    """Apply light mastering to Coqui output so speech sounds less flat."""
    enhanced = normalize(voice, headroom=1.2)
    enhanced = compress_dynamic_range(enhanced, threshold=-24.0, ratio=2.0, attack=10, release=120)
    return enhanced


def soften_sibilance(voice: AudioSegment, attenuation_db: float = 4.5) -> AudioSegment:
    """Apply a lightweight static de-esser to reduce harsh 's' consonants."""
    if len(voice) <= 0:
        return voice

    low_band = low_pass_filter(voice, 4200)
    high_band = high_pass_filter(voice, 4200) - attenuation_db
    combined = low_band.overlay(high_band)

    return low_pass_filter(combined, 8200)


def enhance_tts_audio(voice: AudioSegment, tts_engine: str) -> AudioSegment:
    """Apply gentle mastering so synthesized voices are easier to listen to."""
    if len(voice) <= 0:
        return voice

    enhanced = voice.set_frame_rate(48000)
    engine = safe_text(tts_engine).strip().lower()

    if engine == "coqui":
        enhanced = enhance_coqui_audio(enhanced)
        enhanced = soften_sibilance(enhanced, attenuation_db=2.8)
    elif engine == "gtts":
        enhanced = low_pass_filter(high_pass_filter(enhanced, 105), 7000)
        enhanced = soften_sibilance(enhanced, attenuation_db=4.2)
        enhanced = compress_dynamic_range(enhanced, threshold=-23.0, ratio=2.6, attack=9, release=130)
        enhanced = normalize(enhanced, headroom=1.5)
    elif engine == "edge_human":
        enhanced = low_pass_filter(high_pass_filter(enhanced, 90), 8600)
        enhanced = soften_sibilance(enhanced, attenuation_db=3.4)
        enhanced = compress_dynamic_range(enhanced, threshold=-25.5, ratio=1.7, attack=12, release=160)
        enhanced = normalize(enhanced, headroom=1.8)
    else:
        enhanced = low_pass_filter(high_pass_filter(enhanced, 95), 7600)
        enhanced = soften_sibilance(enhanced, attenuation_db=4.8)
        enhanced = compress_dynamic_range(enhanced, threshold=-24.5, ratio=2.1, attack=8, release=120)
        enhanced = normalize(enhanced, headroom=1.4)

    fade_in_ms = min(16, max(len(enhanced) // 24, 6))
    fade_out_ms = min(28, max(len(enhanced) // 18, 10))
    return enhanced.fade_in(fade_in_ms).fade_out(fade_out_ms)


def post_process_dubbed_track(dubbed: AudioSegment, tts_engine: str) -> AudioSegment:
    """Apply final gentle mastering to the combined dub track before muxing."""
    if len(dubbed) <= 0:
        return dubbed

    engine = safe_text(tts_engine).strip().lower()
    mastered = low_pass_filter(high_pass_filter(dubbed, 80), 8500)
    if engine == "edge_human":
        mastered = soften_sibilance(mastered, attenuation_db=2.4)
        mastered = compress_dynamic_range(mastered, threshold=-27.0, ratio=1.45, attack=14, release=170)
    else:
        if engine == "edge":
            mastered = soften_sibilance(mastered, attenuation_db=3.2)
        mastered = compress_dynamic_range(mastered, threshold=-26.0, ratio=1.8, attack=12, release=160)

    headroom = 1.6 if engine == "coqui" else 1.4
    if engine == "edge_human":
        headroom = 1.9
    mastered = normalize(mastered, headroom=headroom)
    return mastered