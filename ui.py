#!/usr/bin/env python3
"""Web UI for the auto-dubbing pipeline."""

from __future__ import annotations

import math
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable
from xml.etree import ElementTree as ET

import gradio as gr
from deep_translator import GoogleTranslator, MyMemoryTranslator
from pydub import AudioSegment

from autodub import DEFAULT_EDGE_VOICES, autodub_video, detect_cuda_available, detect_rocm_available, tts_segment, sanitize_tts_text


LANGUAGE_CHOICES = [
    ("English (en)", "en"),
    ("Spanish (es)", "es"),
    ("French (fr)", "fr"),
    ("German (de)", "de"),
    ("Hindi (hi)", "hi"),
    ("Japanese (ja)", "ja"),
    ("Portuguese (pt)", "pt"),
    ("Russian (ru)", "ru"),
    ("Slovak (sk)", "sk"),
]


EDGE_VOICE_CHOICES = [
    ("English - Aria (en-US-AriaNeural)", "en-US-AriaNeural"),
    ("Spanish - Elvira (es-ES-ElviraNeural)", "es-ES-ElviraNeural"),
    ("French - Denise (fr-FR-DeniseNeural)", "fr-FR-DeniseNeural"),
    ("German - Katja (de-DE-KatjaNeural)", "de-DE-KatjaNeural"),
    ("Hindi - Swara (hi-IN-SwaraNeural)", "hi-IN-SwaraNeural"),
    ("Japanese - Nanami (ja-JP-NanamiNeural)", "ja-JP-NanamiNeural"),
    ("Portuguese - Francisca (pt-BR-FranciscaNeural)", "pt-BR-FranciscaNeural"),
    ("Russian - Svetlana (ru-RU-SvetlanaNeural)", "ru-RU-SvetlanaNeural"),
    ("Slovak - Viktoria (sk-SK-ViktoriaNeural)", "sk-SK-ViktoriaNeural"),
]

WHISPER_MODEL_CHOICES = [
    ("tiny (fastest)", "tiny"),
    ("base", "base"),
    ("small", "small"),
    ("medium", "medium"),
    ("large-v2", "large-v2"),
    ("large-v3 (best accuracy, GPU recommended)", "large-v3"),
    ("large-v3-turbo (faster high quality)", "large-v3-turbo"),
    ("distil-large-v3 (fast large model)", "distil-large-v3"),
]

QUALITY_PRESET_CHOICES = [
    ("Fast", "fast"),
    ("Balanced", "balanced"),
    ("Best Quality", "best"),
]

TRANSLATION_PROVIDER_CHOICES = [
    ("Google", "google"),
    ("MyMemory", "mymemory"),
]

ASR_ENGINE_CHOICES = [
    ("Auto (recommended: stable-ts when available)", "auto"),
    ("Whisper (fallback, compatible)", "whisper"),
    ("stable-ts (accurate timestamps, fixes missed opening)", "stable-ts"),
]

AUTO_DEVICE = "cuda" if detect_cuda_available() else ("rocm" if detect_rocm_available() else "cpu")

SUPPORTED_TTS_UPLOAD_TYPES = [
    ".txt",
    ".text",
    ".asc",
    ".md",
    ".markdown",
    ".srt",
    ".vtt",
    ".sub",
    ".ass",
    ".ssa",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".rtf",
    ".log",
    ".ini",
    ".cfg",
    ".docx",
]





def voice_for_language(lang: str) -> str:
    return DEFAULT_EDGE_VOICES.get(lang, DEFAULT_EDGE_VOICES["en"])


def _update_edge_voice_dropdown(lang: str, tts_engine: str) -> gr.Dropdown:
    return gr.Dropdown(
        choices=EDGE_VOICE_CHOICES,
        value=voice_for_language(lang),
        interactive=tts_engine == "edge",
    )


def on_quality_preset_change(preset: str, tts_engine: str) -> tuple[gr.Dropdown, gr.Dropdown, gr.Radio]:
    if preset == "fast":
        whisper_value = "base"
        profile_value = "long"
        device_value = "auto"
    elif preset == "best":
        whisper_value = "large-v3"
        profile_value = "short"
        device_value = AUTO_DEVICE if AUTO_DEVICE in {"cuda", "rocm"} else "auto"
    else:
        whisper_value = "small"
        profile_value = "auto"
        device_value = "auto"

    return (
        gr.Dropdown(choices=WHISPER_MODEL_CHOICES, value=whisper_value),
        gr.Dropdown(
            choices=[
                ("Auto (recommended)", "auto"),
                ("Balanced", "balanced"),
                ("Short video quality", "short"),
                ("Long video stability", "long"),
            ],
            value=profile_value,
        ),
        gr.Radio(choices=["auto", "cpu", "cuda", "rocm"], value=device_value),
    )


def parse_optional_number(value: object, *, default: float | None) -> float | None:
    if value in (None, ""):
        return default

    if not isinstance(value, (int, float, str)):
        return default

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(parsed) or math.isinf(parsed):
        return default

    return parsed


def split_text_for_tts(text: str, max_chunk_chars: int = 2500) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    if len(normalized) <= max_chunk_chars:
        return [normalized]

    chunks: list[str] = []
    current = ""

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    for sentence in sentences:
        candidate = sentence.strip()
        if not candidate:
            continue

        if len(candidate) > max_chunk_chars:
            words = candidate.split()
            long_current = ""
            for word in words:
                proposed = f"{long_current} {word}".strip()
                if len(proposed) > max_chunk_chars and long_current:
                    chunks.append(long_current)
                    long_current = word
                else:
                    long_current = proposed
            if long_current:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(long_current)
            continue

        proposed = f"{current} {candidate}".strip()
        if len(proposed) > max_chunk_chars and current:
            chunks.append(current)
            current = candidate
        else:
            current = proposed

    if current:
        chunks.append(current)

    return chunks


def translate_text_unlimited(
    text: str,
    source_language: str,
    target_language: str,
    translation_provider: str,
    max_chunk_chars: int = 4500,
    progress_callback: Callable[[float, str], object] | None = None,
) -> str:
    chunks = split_text_for_tts(text, max_chunk_chars=max_chunk_chars)
    if not chunks:
        return ""

    provider = (translation_provider or "google").strip().lower()
    translated_chunks: list[str] = []

    total_chunks = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        if provider == "mymemory":
            try:
                translator = MyMemoryTranslator(source=source_language, target=target_language)
            except Exception:
                translator = GoogleTranslator(source=source_language, target=target_language)
        else:
            translator = GoogleTranslator(source=source_language, target=target_language)

        translated_piece = translator.translate(chunk)
        translated_chunks.append(translated_piece if isinstance(translated_piece, str) else " ".join(translated_piece))

        if progress_callback is not None:
            progress_callback(index / max(total_chunks, 1), f"Translating chunk {index}/{total_chunks}")

    return "\n".join(part.strip() for part in translated_chunks if part and part.strip())


def synthesize_text_unlimited(
    text: str,
    lang: str,
    output_mp3: Path,
    tts_engine: str,
    edge_voice: str | None,
    edge_rate: str,
    edge_pitch: str,
    edge_volume: str,
    progress_callback: Callable[[float, str], object] | None = None,
) -> int:
    chunks = split_text_for_tts(text)
    if not chunks:
        raise ValueError("No text available for synthesis.")

    if len(chunks) == 1:
        tts_segment(
            text=chunks[0],
            lang=lang,
            output_mp3=output_mp3,
            tts_engine=tts_engine,
            edge_voice=edge_voice,
            edge_rate=edge_rate,
            edge_pitch=edge_pitch,
            edge_volume=edge_volume,
        )
        if progress_callback is not None:
            progress_callback(1.0, "Synthesis complete")
        return 1

    with tempfile.TemporaryDirectory(prefix="tts_chunks_") as tmp_dir:
        temp_dir = Path(tmp_dir)
        combined = AudioSegment.silent(duration=0)

        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            chunk_path = temp_dir / f"chunk_{index:04d}.mp3"
            tts_segment(
                text=chunk,
                lang=lang,
                output_mp3=chunk_path,
                tts_engine=tts_engine,
                edge_voice=edge_voice,
                edge_rate=edge_rate,
                edge_pitch=edge_pitch,
                edge_volume=edge_volume,
            )

            piece = AudioSegment.from_file(chunk_path, format="mp3")
            if len(combined) > 0:
                combined += AudioSegment.silent(duration=140)
            combined += piece

            if progress_callback is not None:
                progress_callback(index / max(total_chunks, 1), f"Synthesizing chunk {index}/{total_chunks}")

        combined.export(output_mp3, format="mp3")

    return len(chunks)


def _extract_docx_text(file_path: Path) -> str:
    with zipfile.ZipFile(file_path) as archive:
        with archive.open("word/document.xml") as doc_xml:
            root = ET.parse(doc_xml).getroot()

    text_parts = [node.text for node in root.iter() if node.tag.endswith("}t") and node.text]
    return "\n".join(text_parts)


def _extract_uploaded_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".docx":
        return _extract_docx_text(file_path)

    raw = file_path.read_bytes()
    decoded = raw.decode("utf-8", errors="ignore")

    if suffix == ".json":
        try:
            parsed = json.loads(decoded)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            return decoded

    if suffix == ".rtf":
        without_groups = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", decoded)
        without_braces = without_groups.replace("{", " ").replace("}", " ")
        return re.sub(r"\s+", " ", without_braces).strip()

    if suffix in {".xml", ".html", ".htm"}:
        no_tags = re.sub(r"<[^>]+>", " ", decoded)
        return re.sub(r"\s+", " ", no_tags).strip()

    return decoded


def run_dub(
    input_video: str | None,
    youtube_url: str,
    target_lang: str,
    whisper_model: str,
    device: str,
    optimization_profile: str,
    tts_engine: str,
    edge_voice: str,
    translation_provider: str,
    hf_token: str,
    include_original_audio: bool,
    export_srt: bool,
    resume_enabled: bool,
    glossary_text: str,
    use_time_range: bool,
    start_time_s: float,
    end_time_s: float | None,
    keep_temp: bool,
    asr_engine: str = "auto",
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str, str, str | None]:
    logs: list[str] = []
    temp_download_dir: Path | None = None

    yt_url = youtube_url.strip() if youtube_url else ""

    if yt_url:
        if shutil.which("yt-dlp") is None:
            return None, "Missing tool", "Please install yt-dlp first: pip install yt-dlp", None

        temp_download_dir = Path(tempfile.mkdtemp(prefix="autodub_yt_"))
        downloaded_video = temp_download_dir / "source.mp4"
        logs.append("Downloading video from YouTube...")

        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "-f",
                    "mp4/best",
                    "--merge-output-format",
                    "mp4",
                    "-o",
                    str(downloaded_video),
                    yt_url,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            error_text = exc.stderr.strip() or exc.stdout.strip() or "Unknown yt-dlp error"
            if temp_download_dir is not None:
                shutil.rmtree(temp_download_dir, ignore_errors=True)
            return None, "Download failed", f"Could not download YouTube video.\n{error_text}", None

        input_path = downloaded_video
    elif isinstance(input_video, str):
        input_path = Path(input_video)
    elif isinstance(input_video, dict) and "path" in input_video:
        input_path = Path(str(input_video["path"]))
    elif input_video is not None and hasattr(input_video, "path"):
        path_attr = getattr(input_video, "path", None)
        if path_attr is None:
            return None, "Invalid input", "Uploaded video path is missing.", None
        input_path = Path(str(path_attr))
    else:
        return None, "No input", "Upload a video file or paste a YouTube URL.", None

    if not input_path.exists():
        if temp_download_dir is not None:
            shutil.rmtree(temp_download_dir, ignore_errors=True)
        return None, "Invalid input", f"Input file does not exist: {input_path}", None

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    safe_stem = input_path.stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = outputs_dir / f"{safe_stem}_{target_lang}_{timestamp}.mp4"

    normalized_start_time = parse_optional_number(start_time_s, default=0.0)
    normalized_end_time = parse_optional_number(end_time_s, default=None)

    if not use_time_range:
        normalized_start_time = 0.0
        normalized_end_time = None
    elif normalized_start_time is None:
        normalized_start_time = 0.0

    if normalized_start_time < 0:
        return None, "Invalid time range", "Start time must be 0 or greater.", None
    if normalized_end_time is not None and normalized_end_time <= normalized_start_time:
        return None, "Invalid time range", "End time must be greater than start time.", None

    start_time = perf_counter()

    def add_log(message: str) -> None:
        logs.append(message)

    def update_progress(value: float, stage: str) -> None:
        elapsed = perf_counter() - start_time
        eta_s: float | None = None
        if value > 0.01:
            eta_s = elapsed * (1.0 - value) / value

        if eta_s is None:
            desc = f"{stage} | elapsed {elapsed/60:.1f}m"
        else:
            desc = f"{stage} | elapsed {elapsed/60:.1f}m | ETA {eta_s/60:.1f}m"

        progress(value, desc=desc)

    try:
        autodub_video(
            input_path=input_path,
            output_path=output_path,
            target_lang=target_lang,
            whisper_model=whisper_model,
            device=device,
            optimization_profile=optimization_profile,
            translation_provider=translation_provider,
            hf_token=(hf_token or "").strip() or None,
            tts_engine=tts_engine,
            edge_voice=edge_voice if tts_engine == "edge" else None,
            include_original_audio=include_original_audio,
            export_srt=export_srt,
            resume_enabled=resume_enabled,
            glossary_text=glossary_text,
            start_time_s=normalized_start_time,
            end_time_s=normalized_end_time,
            keep_temp=keep_temp,
            asr_engine=asr_engine,
            progress_callback=add_log,
            progress_percent_callback=update_progress,
        )
    except Exception as exc:
        logs.append(f"Error: {exc}")
        if temp_download_dir is not None:
            shutil.rmtree(temp_download_dir, ignore_errors=True)
        return None, "Failed", "\n".join(logs), None

    if temp_download_dir is not None:
        shutil.rmtree(temp_download_dir, ignore_errors=True)

    subtitle_output = str(output_path.with_suffix(".srt")) if export_srt and output_path.with_suffix(".srt").exists() else None
    return str(output_path), "Success", "\n".join(logs), subtitle_output


def convert_text_to_speech(
    text: str,
    text_file: object | None,
    auto_translate: bool,
    source_language: str,
    language: str,
    translation_provider: str,
    tts_engine: str,
    edge_voice: str,
    edge_rate: int = 0,
    edge_pitch: int = 0,
    edge_volume: int = 0,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str]:
    """Convert text to speech and return audio file path and status message."""
    input_text = text or ""
    source_name = "text box"

    uploaded_path: Path | None = None
    if isinstance(text_file, str):
        uploaded_path = Path(text_file)
    elif isinstance(text_file, dict) and "path" in text_file:
        uploaded_path = Path(str(text_file["path"]))
    elif text_file is not None and hasattr(text_file, "name"):
        name_attr = getattr(text_file, "name", None)
        if name_attr:
            uploaded_path = Path(str(name_attr))
    elif text_file is not None and hasattr(text_file, "path"):
        path_attr = getattr(text_file, "path", None)
        if path_attr:
            uploaded_path = Path(str(path_attr))

    if uploaded_path is not None and uploaded_path.exists():
        try:
            input_text = _extract_uploaded_text(uploaded_path)
            source_name = f"file {uploaded_path.name}"
        except Exception as exc:
            return None, f"Error: could not read uploaded file ({exc})"

    if not input_text or not input_text.strip():
        return None, "Error: Please enter text or upload a supported file."
    
    try:
        progress(0.05, desc="Preparing input")
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_audio = outputs_dir / f"tts_{language}_{timestamp}.mp3"
        
        # Format edge-tts parameters
        rate_str = f"{edge_rate:+d}%" if edge_rate != 0 else "+0%"
        pitch_str = f"{edge_pitch:+d}Hz" if edge_pitch != 0 else "+0Hz"
        volume_str = f"{edge_volume:+d}%" if edge_volume != 0 else "+0%"
        
        cleaned_text = sanitize_tts_text(input_text)

        text_for_tts = cleaned_text
        if auto_translate and source_language != language:
            translated_text_str = translate_text_unlimited(
                text=cleaned_text,
                source_language=source_language,
                target_language=language,
                translation_provider=translation_provider,
                progress_callback=lambda value, stage: progress(0.10 + (value * 0.40), desc=stage),
            )
            text_for_tts = sanitize_tts_text(translated_text_str)
        else:
            progress(0.50, desc="Skipping translation")
        
        chunk_count = synthesize_text_unlimited(
            text=text_for_tts,
            lang=language,
            output_mp3=output_audio,
            tts_engine=tts_engine,
            edge_voice=edge_voice if tts_engine == "edge" else None,
            edge_rate=rate_str,
            edge_pitch=pitch_str,
            edge_volume=volume_str,
            progress_callback=lambda value, stage: progress(0.50 + (value * 0.50), desc=stage),
        )

        progress(1.0, desc="Done")

        if auto_translate and source_language != language:
            return str(output_audio), f"Success: translated {source_language} -> {language}, synthesized {chunk_count} chunk(s), and generated {output_audio.name}"
        return str(output_audio), f"Success: synthesized {chunk_count} chunk(s) from {source_name} as {output_audio.name}"
    except Exception as exc:
        return None, f"Error: {str(exc)}"


def _update_edge_voice_for_tts(lang: str, tts_engine: str) -> gr.Dropdown:
    """Update edge voice dropdown based on language and TTS engine."""
    return gr.Dropdown(
        choices=EDGE_VOICE_CHOICES,
        value=voice_for_language(lang),
        interactive=tts_engine == "edge",
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Auto Dubbing Studio") as demo:
        gr.Markdown("# Auto Dubbing Studio")

        with gr.Tabs():
            # Video Dubbing Tab
            with gr.TabItem("Video Dubbing"):
                gr.Markdown("Upload a video (or paste a YouTube URL), pick a language, and click **Generate**.")

                with gr.Row():
                    with gr.Column(scale=3):
                        input_video = gr.Video(label="Upload Video")
                        youtube_url = gr.Textbox(
                            label="Or paste a YouTube URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                        )

                    with gr.Column(scale=2):
                        target_lang = gr.Dropdown(
                            label="Target Language",
                            choices=LANGUAGE_CHOICES,
                            value="es",
                        )
                        quality_preset = gr.Radio(
                            label="Quality Preset",
                            choices=QUALITY_PRESET_CHOICES,
                            value="balanced",
                            info="Fast = quick; Balanced = default; Best Quality = slower and more accurate",
                        )
                        run_button = gr.Button("Generate Dubbed Video", variant="primary", size="lg")

                with gr.Accordion("Settings", open=False):
                    with gr.Row():
                        whisper_model = gr.Dropdown(
                            label="Whisper Model",
                            choices=WHISPER_MODEL_CHOICES,
                            value="small",
                        )
                        device = gr.Radio(
                            label="Device",
                            choices=["auto", "cpu", "cuda", "rocm"],
                            value="auto",
                            info=f"Auto → {AUTO_DEVICE}",
                        )
                        optimization_profile = gr.Dropdown(
                            label="Optimization Profile",
                            choices=[
                                ("Auto (recommended)", "auto"),
                                ("Balanced", "balanced"),
                                ("Short video quality", "short"),
                                ("Long video stability", "long"),
                            ],
                            value="auto",
                        )

                    with gr.Row():
                        asr_engine = gr.Radio(
                            label="ASR Engine",
                            choices=ASR_ENGINE_CHOICES,
                            value="auto",
                        )
                        tts_engine = gr.Radio(
                            label="TTS Engine",
                            choices=["edge", "gtts"],
                            value="edge",
                            info="Edge sounds more natural in most cases",
                        )
                        edge_voice = gr.Dropdown(
                            label="Edge Voice",
                            choices=EDGE_VOICE_CHOICES,
                            value=voice_for_language("es"),
                        )

                    with gr.Row():
                        translation_provider = gr.Radio(
                            label="Translation Provider",
                            choices=TRANSLATION_PROVIDER_CHOICES,
                            value="google",
                        )
                        glossary_text = gr.Textbox(
                            label="Glossary Overrides",
                            lines=3,
                            placeholder="death => morti\nwar => vojna",
                            info="Optional forced translations, one per line",
                        )

                    with gr.Row():
                        include_original_audio = gr.Checkbox(label="Keep original audio in background", value=True)
                        export_srt = gr.Checkbox(label="Export subtitles (.srt)", value=True)
                        resume_enabled = gr.Checkbox(label="Resume previous job", value=True)
                        keep_temp = gr.Checkbox(label="Keep temp files", value=False)

                    with gr.Row():
                        hf_token = gr.Textbox(
                            label="Hugging Face Token (optional)",
                            type="password",
                            placeholder="hf_...",
                        )
                        use_time_range = gr.Checkbox(label="Custom time range", value=False)
                        start_time_s = gr.Number(label="Start (seconds)", value=0, precision=0)
                        end_time_s = gr.Number(label="End (seconds, optional)", value=None, precision=0)

                with gr.Row():
                    output_video = gr.Video(label="Dubbed Output")
                    output_srt = gr.File(label="Subtitle File (.srt)")

                status = gr.Textbox(label="Status", interactive=False)
                logs = gr.Textbox(label="Logs", lines=10, interactive=False)

                target_lang.change(fn=_update_edge_voice_dropdown, inputs=[target_lang, tts_engine], outputs=[edge_voice])
                tts_engine.change(fn=_update_edge_voice_dropdown, inputs=[target_lang, tts_engine], outputs=[edge_voice])
                quality_preset.change(
                    fn=on_quality_preset_change,
                    inputs=[quality_preset, tts_engine],
                    outputs=[whisper_model, optimization_profile, device],
                )

                run_button.click(
                    fn=run_dub,
                    inputs=[
                        input_video,
                        youtube_url,
                        target_lang,
                        whisper_model,
                        device,
                        optimization_profile,
                        tts_engine,
                        edge_voice,
                        translation_provider,
                        hf_token,
                        include_original_audio,
                        export_srt,
                        resume_enabled,
                        glossary_text,
                        use_time_range,
                        start_time_s,
                        end_time_s,
                        keep_temp,
                        asr_engine,
                    ],
                    outputs=[output_video, status, logs, output_srt],
                )

            # Text-to-Speech Tab
            with gr.TabItem("Text-to-Speech"):
                gr.Markdown("Convert text to speech with optional auto-translation.")

                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="Text to Convert",
                            lines=6,
                            placeholder="Enter the text you want to convert to speech...",
                            value="Hello, this is a test of the text-to-speech system.",
                        )
                        tts_text_file = gr.File(
                            label="Or upload text file",
                            file_types=SUPPORTED_TTS_UPLOAD_TYPES,
                            type="filepath",
                        )
                        tts_auto_translate = gr.Checkbox(
                            label="Auto-translate text before TTS",
                            value=False,
                        )
                        tts_source_language = gr.Dropdown(
                            label="Source Language",
                            choices=LANGUAGE_CHOICES,
                            value="en",
                        )
                        tts_language = gr.Dropdown(
                            label="Target Language",
                            choices=LANGUAGE_CHOICES,
                            value="en",
                        )

                    with gr.Column(scale=1):
                        tts_engine_tab = gr.Radio(
                            label="TTS Engine",
                            choices=["edge", "gtts"],
                            value="edge",
                            info="Edge sounds more natural",
                        )
                        tts_translation_provider = gr.Radio(
                            label="Translation Provider",
                            choices=TRANSLATION_PROVIDER_CHOICES,
                            value="google",
                        )
                        tts_edge_voice = gr.Dropdown(
                            label="Voice (Edge only)",
                            choices=EDGE_VOICE_CHOICES,
                            value=voice_for_language("en"),
                        )

                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        tts_edge_rate = gr.Slider(
                            label="Rate (speed)",
                            minimum=-50,
                            maximum=50,
                            value=0,
                            step=5,
                            info="Negative = slower, Positive = faster",
                        )
                        tts_edge_pitch = gr.Slider(
                            label="Pitch",
                            minimum=-20,
                            maximum=20,
                            value=0,
                            step=2,
                            info="Negative = lower, Positive = higher",
                        )
                        tts_edge_volume = gr.Slider(
                            label="Volume",
                            minimum=-100,
                            maximum=100,
                            value=0,
                            step=5,
                            info="Negative = quieter, Positive = louder",
                        )

                tts_convert_button = gr.Button("Convert to Speech", variant="primary", size="lg")

                with gr.Row():
                    tts_audio_output = gr.Audio(label="Generated Audio", interactive=False)

                tts_status = gr.Textbox(label="Status", interactive=False)

                tts_language.change(
                    fn=_update_edge_voice_for_tts,
                    inputs=[tts_language, tts_engine_tab],
                    outputs=[tts_edge_voice],
                )
                tts_engine_tab.change(
                    fn=_update_edge_voice_for_tts,
                    inputs=[tts_language, tts_engine_tab],
                    outputs=[tts_edge_voice],
                )

                tts_convert_button.click(
                    fn=convert_text_to_speech,
                    inputs=[
                        tts_text,
                        tts_text_file,
                        tts_auto_translate,
                        tts_source_language,
                        tts_language,
                        tts_translation_provider,
                        tts_engine_tab,
                        tts_edge_voice,
                        tts_edge_rate,
                        tts_edge_pitch,
                        tts_edge_volume,
                    ],
                    outputs=[tts_audio_output, tts_status],
                )

    return demo


def main() -> None:
    app = build_ui()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
