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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, List, cast

from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from gtts import gTTS
from pydub import AudioSegment
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
    whisper_segments, _info = model.transcribe(str(audio_path), vad_filter=True)

    segments: List[Segment] = []
    for seg in whisper_segments:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(Segment(start_s=float(seg.start), end_s=float(seg.end), source_text=text))
    return segments


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
    total = len(segments)
    for idx, seg in enumerate(tqdm(segments, desc="Translating"), start=1):
        seg.translated_text = translator.translate(seg.source_text)
        if segment_progress_callback is not None:
            segment_progress_callback(idx, total)


def edge_tts_segment(text: str, voice: str, output_mp3: Path) -> None:
    edge_tts = importlib.import_module("edge_tts")

    async def synthesize() -> None:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%")
        await communicate.save(str(output_mp3))

    asyncio.run(synthesize())


def tts_segment(
    text: str,
    lang: str,
    output_mp3: Path,
    tts_engine: str = "edge",
    edge_voice: str | None = None,
) -> None:
    if tts_engine == "edge":
        voice = edge_voice or DEFAULT_EDGE_VOICES.get(lang, DEFAULT_EDGE_VOICES["en"])
        try:
            edge_tts_segment(text, voice, output_mp3)
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


def build_dubbed_track(
    segments: List[Segment],
    total_duration_ms: int,
    temp_dir: Path,
    target_lang: str,
    tts_engine: str = "edge",
    edge_voice: str | None = None,
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
        tts_segment(
            spoken_text,
            target_lang,
            mp3_path,
            tts_engine=tts_engine,
            edge_voice=edge_voice,
        )

        voice = AudioSegment.from_file(mp3_path)
        slot_ms = max(int((seg.end_s - seg.start_s) * 1000), 120)
        voice = fit_audio_to_duration(voice, slot_ms, temp_dir, i)
        dubbed = dubbed.overlay(voice, position=max(int(seg.start_s * 1000), 0))
        if segment_progress_callback is not None:
            segment_progress_callback(i, total)

    return dubbed


def mux_video_with_dub(input_video: Path, dubbed_wav: Path, output_video: Path) -> None:
    # Keep original ambience quietly under dubbed speech for more natural output.
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(dubbed_wav),
            "-filter_complex",
            "[0:a]volume=0.22[bg];[bg][1:a]amix=inputs=2:duration=first:normalize=0[aout]",
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
            raise RuntimeError("No speech segments found.")
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
            segment_progress_callback=tts_progress,
        )
        dubbed_track.export(dubbed_wav, format="wav")

        segments_json.write_text(json.dumps([asdict(s) for s in segments], indent=2), encoding="utf-8")

        report("[5/5] Muxing dubbed audio into video...")
        report_progress(0.92, "Muxing audio and video")
        mux_video_with_dub(working_video, dubbed_wav, output_path)
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
