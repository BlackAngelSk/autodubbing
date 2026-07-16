"""Main pipeline orchestration: the autodub_video function."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, cast

from pydub import AudioSegment

from autodub.config import LARGE_WHISPER_MODELS, COQUI_DEFAULT_MODEL
from autodub.device import configure_hf_hub_access
from autodub.ffmpeg_utils import ensure_ffmpeg, probe_media_duration, extract_audio, trim_video
from autodub.segments import (
    Segment, safe_text,
    save_segments_to_json, load_segments_from_json,
    collapse_consecutive_duplicate_segments,
    write_srt, write_transcript_txt,
)
from autodub.translate import (
    translate_segments_with_progress,
    cached_translation_looks_poor, translation_looks_wrong_language,
)
from autodub.asr_whisper import transcribe_segments_whisper
from autodub.asr_stable_ts import stable_ts_available, _transcribe_with_stable_ts
from autodub.dub import build_dubbed_track
from autodub.mux import mux_video_with_dub
from autodub.tts_postprocess import post_process_dubbed_track, has_meaningful_audio
from autodub.tts_coqui import coqui_tts_segment
from autodub.cache import build_dub_cache_signature, build_resume_dir

logger = logging.getLogger(__name__)


def resolve_asr_engine(asr_engine: str | None, status_callback: Callable[[str], None] | None = None) -> str:
    normalized = safe_text(asr_engine).strip().lower() or "auto"
    from autodub.config import ASR_ENGINE_CHOICES
    if normalized not in ASR_ENGINE_CHOICES:
        raise ValueError(f"Unsupported ASR engine: {asr_engine}")

    if normalized != "auto":
        return normalized

    if stable_ts_available():
        if status_callback is not None:
            status_callback("[asr] Auto-selected stable-ts for stronger speech detection and timestamp alignment.")
        return "stable-ts"

    if status_callback is not None:
        status_callback("[asr] stable-ts is unavailable; falling back to Whisper.")
    return "whisper"


def resolve_processing_profile(
    selected_profile: str,
    clip_duration_s: float | None,
    whisper_model: str,
    device: str,
    min_stretch_speed: float,
    max_stretch_speed: float,
    silence_trim_ms: int,
) -> dict[str, Any]:
    from autodub.device import resolve_device_selection

    applied_profile = selected_profile
    if selected_profile == "auto":
        if clip_duration_s is not None and clip_duration_s <= 150:
            applied_profile = "short"
        elif clip_duration_s is not None and clip_duration_s >= 12 * 60:
            applied_profile = "long"
        else:
            applied_profile = "balanced"

    resolved_device = resolve_device_selection(device)
    resolved_model = whisper_model
    resolved_min = min_stretch_speed
    resolved_max = max_stretch_speed
    resolved_trim = silence_trim_ms
    transcribe_chunk_s: float | None = None
    tts_chunk_window_s: float | None = 75.0

    if applied_profile == "short":
        resolved_min = max(resolved_min, 0.90)
        resolved_max = min(max(resolved_max, 1.28), 1.42)
        resolved_trim = max(resolved_trim, 18)
        tts_chunk_window_s = 60.0
    elif applied_profile == "long":
        if resolved_device != "cuda" and whisper_model in ({"small", "medium"} | LARGE_WHISPER_MODELS):
            resolved_model = "base"
        resolved_min = min(resolved_min, 0.92)
        resolved_max = max(resolved_max, 1.45)
        resolved_trim = max(resolved_trim, 12)
        transcribe_chunk_s = 420.0
        tts_chunk_window_s = 120.0
    else:
        transcribe_chunk_s = 300.0 if clip_duration_s is not None and clip_duration_s >= 8 * 60 else None
        tts_chunk_window_s = 90.0

    return {
        "label": applied_profile,
        "device": resolved_device,
        "whisper_model": resolved_model,
        "min_stretch_speed": resolved_min,
        "max_stretch_speed": resolved_max,
        "silence_trim_ms": resolved_trim,
        "clip_duration_s": clip_duration_s,
        "transcribe_chunk_s": transcribe_chunk_s,
        "tts_chunk_window_s": tts_chunk_window_s,
    }


def build_coqui_speaker_reference(audio_path: Path, output_wav: Path, max_ref_ms: int = 9_000) -> Path | None:
    from pydub.silence import detect_nonsilent
    if not audio_path.exists():
        return None
    try:
        audio = AudioSegment.from_wav(audio_path)
    except Exception:
        return None
    if len(audio) < 1_200:
        return None
    silence_floor = audio.dBFS - 19 if audio.dBFS != float("-inf") else -45
    speech_ranges = detect_nonsilent(audio, min_silence_len=180, silence_thresh=max(silence_floor, -45))
    if not speech_ranges:
        return None
    merged_ranges: list[tuple[int, int]] = []
    for start, end in speech_ranges:
        if not merged_ranges:
            merged_ranges.append((start, end))
            continue
        prev_start, prev_end = merged_ranges[-1]
        if start <= prev_end + 260:
            merged_ranges[-1] = (prev_start, max(prev_end, end))
        else:
            merged_ranges.append((start, end))
    if not merged_ranges:
        return None
    best_start, best_end = max(merged_ranges, key=lambda item: item[1] - item[0])
    if best_end <= best_start + 900:
        return None
    sample = cast(AudioSegment, audio[best_start:best_end])
    if len(sample) > max_ref_ms:
        mid = len(sample) // 2
        half = max_ref_ms // 2
        sample = cast(AudioSegment, sample[max(mid - half, 0):min(mid + half, len(sample))])
    sample = sample.set_channels(1).set_frame_rate(22050)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sample.export(output_wav, format="wav")
    return output_wav if output_wav.exists() else None


def transcribe_segments(
    audio_path: Path,
    model_name: str,
    device: str,
    chunk_length_s: float | None = None,
    cache_dir: Path | None = None,
    chunk_progress_callback: Callable[[int, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    hf_token: str | None = None,
    asr_engine: str = "auto",
) -> list[Segment]:
    resolved_asr_engine = resolve_asr_engine(asr_engine, status_callback=status_callback)

    if resolved_asr_engine == "stable-ts":
        if not stable_ts_available():
            logging.warning(
                "stable-ts is not installed (pip install stable-ts). "
                "Falling back to standard Whisper for this job."
            )
            if status_callback is not None:
                status_callback("[asr] stable-ts not found, falling back to Whisper.")
        else:
            return _transcribe_with_stable_ts(
                audio_path=audio_path,
                model_name=model_name,
                device=device,
                chunk_length_s=chunk_length_s,
                cache_dir=cache_dir,
                chunk_progress_callback=chunk_progress_callback,
                status_callback=status_callback,
                hf_token=hf_token,
            )
    return transcribe_segments_whisper(
        audio_path=audio_path,
        model_name=model_name,
        device=device,
        chunk_length_s=chunk_length_s,
        cache_dir=cache_dir,
        chunk_progress_callback=chunk_progress_callback,
        status_callback=status_callback,
        hf_token=hf_token,
    )


def autodub_video(
    input_path: Path,
    output_path: Path,
    target_lang: str,
    whisper_model: str = "small",
    device: str = "auto",
    translation_provider: str = "google",
    hf_token: str | None = None,
    tts_engine: str = "edge",
    use_page_tts_profile: bool = False,
    edge_voice: str | None = None,
    background_mix_level: float = 0.03,
    include_original_audio: bool = True,
    min_stretch_speed: float = 0.85,
    max_stretch_speed: float = 1.35,
    silence_trim_ms: int = 0,
    optimization_profile: str = "auto",
    export_srt: bool = True,
    resume_enabled: bool = True,
    glossary_text: str = "",
    asr_engine: str = "auto",
    start_time_s: float = 0.0,
    end_time_s: float | None = None,
    keep_temp: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    progress_percent_callback: Callable[[float, str], None] | None = None,
) -> int:
    started_at = time.perf_counter()

    def has_weak_opening_coverage(segments: list[Segment]) -> bool:
        if not segments:
            return True
        first_start = segments[0].start_s
        early_segments = [seg for seg in segments if seg.start_s < 12.0]
        return first_start > 1.4 or len(early_segments) < 2

    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    def report_progress(value: float, label: str) -> None:
        if progress_percent_callback is not None:
            bounded = min(max(value, 0.0), 1.0)
            progress_percent_callback(bounded, label)

    has_hf_token = configure_hf_hub_access(hf_token)
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

    resolved_asr_engine = resolve_asr_engine(asr_engine, status_callback=report)

    temp_base = Path(tempfile.mkdtemp(prefix="autodub_"))
    try:
        resume_dir = None
        if resume_enabled:
            resume_dir = build_resume_dir(
                input_path=input_path,
                output_dir=output_path.parent,
                target_lang=target_lang,
                whisper_model=whisper_model,
                translation_provider=translation_provider,
                tts_engine=tts_engine,
                edge_voice=edge_voice,
                optimization_profile=optimization_profile,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                glossary_text=glossary_text,
                asr_engine=resolved_asr_engine,
            )
            resume_dir.mkdir(parents=True, exist_ok=True)
            report(f"[resume] Cache directory: {resume_dir}")

        working_video = input_path
        extracted_wav = (resume_dir / "extracted.wav") if resume_dir is not None else (temp_base / "extracted.wav")
        dubbed_wav = (resume_dir / "dubbed.wav") if resume_dir is not None else (temp_base / "dubbed.wav")
        segments_json = (resume_dir / "segments.json") if resume_dir is not None else (temp_base / "segments.json")
        asr_cache_dir = (resume_dir / "asr") if resume_dir is not None else None
        tts_cache_dir = (resume_dir / "tts_chunks") if resume_dir is not None else None
        dub_meta_path = (resume_dir / "dubbed_meta.json") if resume_dir is not None else (temp_base / "dubbed_meta.json")
        subtitle_path = output_path.with_suffix(".srt")
        transcript_path = output_path.with_suffix(".transcript.txt")
        translated_transcript_path = output_path.with_suffix(".translated.txt")
        glossary_overrides = __import__("autodub.translate", fromlist=["parse_glossary_overrides"]).parse_glossary_overrides(glossary_text)

        if start_time_s > 0 or end_time_s is not None:
            clipped_video = (resume_dir / "trimmed_input.mp4") if resume_dir is not None else (temp_base / "trimmed_input.mp4")
            if clipped_video.exists():
                report("[resume] Reusing trimmed video range...")
            else:
                report("[0/5] Trimming selected video range...")
                report_progress(0.01, "Preparing selected time range")
                trim_video(input_path, clipped_video, start_time_s, end_time_s)
            working_video = clipped_video

        resolved_settings = resolve_processing_profile(
            optimization_profile,
            probe_media_duration(working_video),
            whisper_model, device,
            min_stretch_speed, max_stretch_speed, silence_trim_ms,
        )
        clip_duration_s = resolved_settings["clip_duration_s"]
        clip_label = "unknown length"
        if isinstance(clip_duration_s, float):
            clip_label = f"{clip_duration_s / 60:.1f} min" if clip_duration_s >= 60 else f"{clip_duration_s:.0f} sec"
        report(
            "[opt] "
            f"Profile={resolved_settings['label']} | clip={clip_label} | "
            f"Device={resolved_settings['device']} | Whisper={resolved_settings['whisper_model']} | "
            f"Translate={translation_provider}"
        )
        if cast(str, resolved_settings["whisper_model"]) in LARGE_WHISPER_MODELS:
            report("[whisper] Large models may download several GB on first use.")
            if not has_hf_token:
                report("[hf] Optional: set `HF_TOKEN` or use the UI token field for higher rate limits and faster downloads.")
        if cast(str, resolved_settings["device"]) != "cuda" and cast(str, resolved_settings["whisper_model"]) in LARGE_WHISPER_MODELS:
            report("[hw] Large Whisper on CPU can be slow; use CUDA or switch to 'medium' for faster runs.")

        report("[1/5] Extracting audio...")
        if extracted_wav.exists():
            report("[resume] Reusing extracted audio cache...")
            report_progress(0.10, "Audio extracted")
        else:
            report_progress(0.03, "Extracting audio")
            extract_audio(working_video, extracted_wav)
            report_progress(0.10, "Audio extracted")

        _asr_label = "stable-ts" if resolved_asr_engine == "stable-ts" else "Whisper"
        report(f"[2/5] Transcribing with {_asr_label}...")

        def asr_chunk_progress(done: int, total: int) -> None:
            start = 0.12
            end = 0.35
            fraction = done / max(total, 1)
            report_progress(start + (end - start) * fraction, f"Transcribing chunks ({done}/{total})")

        if segments_json.exists():
            segments = load_segments_from_json(segments_json)
            report(f"[resume] Reusing cached segments ({len(segments)} segments)...")
            if has_weak_opening_coverage(segments):
                report("[resume] Cached ASR seems to miss opening speech; rebuilding transcription...")
                report_progress(0.12, "Re-transcribing opening coverage")
                segments = transcribe_segments(
                    extracted_wav,
                    cast(str, resolved_settings["whisper_model"]),
                    cast(str, resolved_settings["device"]),
                    chunk_length_s=cast(float | None, resolved_settings["transcribe_chunk_s"]),
                    cache_dir=asr_cache_dir,
                    chunk_progress_callback=asr_chunk_progress,
                    status_callback=report,
                    hf_token=hf_token,
                    asr_engine=resolved_asr_engine,
                )
                save_segments_to_json(segments, segments_json)
            report_progress(0.35, f"Transcription complete ({len(segments)} segments)")
        else:
            report_progress(0.12, "Transcribing speech")
            segments = transcribe_segments(
                extracted_wav,
                cast(str, resolved_settings["whisper_model"]),
                cast(str, resolved_settings["device"]),
                chunk_length_s=cast(float | None, resolved_settings["transcribe_chunk_s"]),
                cache_dir=asr_cache_dir,
                chunk_progress_callback=asr_chunk_progress,
                status_callback=report,
                hf_token=hf_token,
                asr_engine=resolved_asr_engine,
            )
            save_segments_to_json(segments, segments_json)
            report_progress(0.35, f"Transcription complete ({len(segments)} segments)")

        if not segments:
            raise RuntimeError(
                "No speech segments found in the selected range. "
                "Try a different time window, use a larger Whisper model, or increase spoken content."
            )

        report("[3/5] Translating segments...")

        def translation_progress(done: int, total: int) -> None:
            start = 0.35
            end = 0.60
            fraction = done / max(total, 1)
            report_progress(start + (end - start) * fraction, f"Translating ({done}/{total})")

        needs_translation = any(not safe_text(seg.translated_text).strip() for seg in segments)
        if not needs_translation and cached_translation_looks_poor(segments, target_lang):
            report("[resume] Cached translations look low quality; re-translating segments...")
            needs_translation = True
        if not needs_translation and translation_looks_wrong_language(segments, target_lang):
            report("[resume] Cached translations appear to be in the wrong language; re-translating segments...")
            needs_translation = True
        if needs_translation:
            translate_segments_with_progress(
                segments, target_lang,
                segment_progress_callback=translation_progress,
                glossary_overrides=glossary_overrides,
                translation_provider=translation_provider,
            )
            if translation_looks_wrong_language(segments, target_lang):
                report("[translate] Output still looks wrong-language; forcing explicit English->target translation...")
                translate_segments_with_progress(
                    segments, target_lang,
                    segment_progress_callback=translation_progress,
                    glossary_overrides=glossary_overrides,
                    translation_provider=translation_provider,
                    force_english_source=True,
                )
            if translation_looks_wrong_language(segments, target_lang):
                raise RuntimeError(
                    "Translation appears to remain in the wrong language for the selected target. "
                    "Please retry with --no-resume or switch translation provider."
                )
            save_segments_to_json(segments, segments_json)
        else:
            report("[resume] Reusing cached translations...")
            report_progress(0.60, f"Translating ({len(segments)}/{len(segments)})")

        deduped_segments = collapse_consecutive_duplicate_segments(segments)
        removed_count = len(segments) - len(deduped_segments)
        if removed_count > 0:
            report(f"[clean] Collapsed {removed_count} consecutive duplicate subtitle segment(s).")
            segments = deduped_segments
            save_segments_to_json(segments, segments_json)

        write_transcript_txt(segments, transcript_path, translated=False)
        write_transcript_txt(segments, translated_transcript_path, translated=True)
        report(f"[transcript] Source transcript written to: {transcript_path}")
        report(f"[transcript] Translated transcript written to: {translated_transcript_path}")

        if export_srt:
            write_srt(segments, subtitle_path)
            report(f"[srt] Subtitle file written to: {subtitle_path}")

        report("[4/5] Generating dubbed track...")
        active_min_stretch = cast(float, resolved_settings["min_stretch_speed"])
        active_max_stretch = cast(float, resolved_settings["max_stretch_speed"])
        active_silence_trim = cast(int, resolved_settings["silence_trim_ms"])
        coqui_speaker_wav: Path | None = None
        coqui_speaker_fingerprint = ""
        coqui_model_name = ""
        if tts_engine == "coqui":
            coqui_model_name = os.environ.get("AUTODUB_COQUI_MODEL", "").strip() or COQUI_DEFAULT_MODEL
            coqui_ref_path = (resume_dir / "coqui_speaker_ref.wav") if resume_dir is not None else (temp_base / "coqui_speaker_ref.wav")
            if coqui_ref_path.exists() and has_meaningful_audio(coqui_ref_path, min_nonsilent_ms=700):
                coqui_speaker_wav = coqui_ref_path
            else:
                coqui_speaker_wav = build_coqui_speaker_reference(extracted_wav, coqui_ref_path)
            if coqui_speaker_wav is not None and coqui_speaker_wav.exists():
                coqui_speaker_fingerprint = f"{coqui_speaker_wav.stat().st_size}:{int(coqui_speaker_wav.stat().st_mtime)}"
                report("[coqui] Using auto speaker reference from source audio.")
            else:
                report("[coqui] Speaker reference unavailable; using default model speaker.")

        dub_signature = build_dub_cache_signature(
            segments,
            target_lang=target_lang,
            tts_engine=tts_engine,
            use_page_tts_profile=use_page_tts_profile,
            edge_voice=edge_voice,
            min_stretch_speed=active_min_stretch,
            max_stretch_speed=active_max_stretch,
            silence_trim_ms=active_silence_trim,
            coqui_model=coqui_model_name,
            coqui_speaker_fingerprint=coqui_speaker_fingerprint,
        )

        can_reuse_dubbed_audio = False
        if dubbed_wav.exists() and dub_meta_path.exists():
            try:
                dub_meta = json.loads(dub_meta_path.read_text(encoding="utf-8"))
                can_reuse_dubbed_audio = (
                    dub_meta.get("signature") == dub_signature
                    and has_meaningful_audio(dubbed_wav)
                )
            except Exception:
                can_reuse_dubbed_audio = False

        if can_reuse_dubbed_audio:
            report("[resume] Reusing cached dubbed audio...")
            report_progress(0.90, "Synthesizing voice (cached)")
        else:
            if dubbed_wav.exists():
                report("[resume] Dubbed audio cache is stale or silent; regenerating from current segments...")
            report_progress(0.62, "Generating neural voice")

            def tts_progress(done: int, total: int) -> None:
                start = 0.62
                end = 0.90
                fraction = done / max(total, 1)
                report_progress(start + (end - start) * fraction, f"Synthesizing voice ({done}/{total})")

            base_audio = AudioSegment.from_wav(extracted_wav)
            dubbed_track = build_dubbed_track(
                segments, len(base_audio),
                temp_base, target_lang,
                tts_engine=tts_engine, edge_voice=edge_voice,
                min_stretch_speed=active_min_stretch,
                max_stretch_speed=active_max_stretch,
                silence_trim_ms=active_silence_trim,
                segment_progress_callback=tts_progress,
                chunk_window_s=cast(float | None, resolved_settings["tts_chunk_window_s"]),
                cache_dir=tts_cache_dir,
                coqui_speaker_wav=coqui_speaker_wav,
                use_page_tts_profile=use_page_tts_profile,
            )
            dubbed_track = post_process_dubbed_track(dubbed_track, tts_engine)
            dubbed_track.export(dubbed_wav, format="wav")
            if not has_meaningful_audio(dubbed_wav):
                raise RuntimeError(
                    "Generated dubbed audio is silent. "
                    "Try switching TTS engine/voice and rerun with --no-resume."
                )
            dub_meta_path.write_text(
                json.dumps({"signature": dub_signature}, indent=2),
                encoding="utf-8",
            )

        save_segments_to_json(segments, segments_json)

        report("[5/5] Muxing dubbed audio into video...")
        report_progress(0.92, "Muxing audio and video")
        if not include_original_audio:
            report("[mix] Original source audio disabled; exporting dubbed speech only.")
        mux_video_with_dub(
            working_video, dubbed_wav, output_path,
            background_mix_level=background_mix_level,
            include_original_audio=include_original_audio,
        )
        report_progress(1.0, "Completed")

        report(f"Done. Output written to: {output_path}")
        elapsed_s = time.perf_counter() - started_at
        elapsed_m, elapsed_rem_s = divmod(int(round(elapsed_s)), 60)
        report(f"Total processing time: {elapsed_m}m {elapsed_rem_s:02d}s")
        if keep_temp:
            report(f"Temp files kept at: {temp_base}")
        return 0
    finally:
        if not keep_temp:
            shutil.rmtree(temp_base, ignore_errors=True)