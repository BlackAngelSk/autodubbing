#!/usr/bin/env python3
"""Web UI for the auto-dubbing pipeline."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter

import gradio as gr

from autodub import DEFAULT_EDGE_VOICES, autodub_video


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


def voice_for_language(lang: str) -> str:
    return DEFAULT_EDGE_VOICES.get(lang, DEFAULT_EDGE_VOICES["en"])


def on_language_change(lang: str) -> gr.Dropdown:
    return gr.Dropdown(value=voice_for_language(lang))


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


def run_dub(
    input_video: str | None,
    youtube_url: str,
    target_lang: str,
    whisper_model: str,
    device: str,
    optimization_profile: str,
    tts_engine: str,
    edge_voice: str,
    export_srt: bool,
    resume_enabled: bool,
    glossary_text: str,
    use_time_range: bool,
    start_time_s: float,
    end_time_s: float | None,
    keep_temp: bool,
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
            tts_engine=tts_engine,
            edge_voice=edge_voice if tts_engine == "edge" else None,
            export_srt=export_srt,
            resume_enabled=resume_enabled,
            glossary_text=glossary_text,
            start_time_s=normalized_start_time,
            end_time_s=normalized_end_time,
            keep_temp=keep_temp,
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


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Auto Dubbing Studio") as demo:
        gr.Markdown(
            """
# Auto Dubbing Studio
Upload a video, choose target language, and generate a dubbed version.
            """.strip()
        )

        with gr.Row():
            with gr.Column(scale=2):
                input_video = gr.Video(label="1) Upload Video")
                youtube_url = gr.Textbox(
                    label="1b) Or Paste YouTube Link",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                run_button = gr.Button("2) Generate Dubbed Video", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### 3) Settings")
                target_lang = gr.Dropdown(
                    label="Target Language",
                    choices=LANGUAGE_CHOICES,
                    value="es",
                    info="Pick the language for dubbed speech",
                )
                whisper_model = gr.Dropdown(
                    label="Whisper Model",
                    choices=["tiny", "base", "small", "medium"],
                    value="small",
                )
                device = gr.Radio(label="Device", choices=["auto", "cpu", "cuda"], value="auto")
                optimization_profile = gr.Dropdown(
                    label="Optimization Profile",
                    choices=[
                        ("Auto (recommended)", "auto"),
                        ("Short video quality", "short"),
                        ("Long video stability", "long"),
                    ],
                    value="auto",
                    info="Auto tunes for short clips or long videos",
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
                export_srt = gr.Checkbox(
                    label="Export translated subtitles (.srt)",
                    value=True,
                )
                resume_enabled = gr.Checkbox(
                    label="Resume previous job if possible",
                    value=True,
                    info="Reuses cached chunks and translations on reruns",
                )
                glossary_text = gr.Textbox(
                    label="Glossary Overrides",
                    lines=4,
                    placeholder="death => smrti\nwar => vojna",
                    info="Optional forced translations, one rule per line",
                )
                with gr.Accordion("Custom time range (optional)", open=False):
                    use_time_range = gr.Checkbox(
                        label="Enable custom time range",
                        value=False,
                        info="Turn on to dub only a selected part of the video",
                    )
                    start_time_s = gr.Number(
                        label="Start Time (seconds)",
                        value=0,
                        precision=0,
                        info="Set start point for dubbing window",
                    )
                    end_time_s = gr.Number(
                        label="End Time (seconds, optional)",
                        value=None,
                        precision=0,
                        info="Leave empty to dub until video end",
                    )
                keep_temp = gr.Checkbox(label="Keep temp files", value=False)

        target_lang.change(fn=on_language_change, inputs=[target_lang], outputs=[edge_voice])

        with gr.Row():
            output_video = gr.Video(label="4) Dubbed Output")
            output_srt = gr.File(label="5) Subtitle File (.srt)")

        status = gr.Textbox(label="Status", interactive=False)
        logs = gr.Textbox(label="Logs", lines=12, interactive=False)

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
                export_srt,
                resume_enabled,
                glossary_text,
                use_time_range,
                start_time_s,
                end_time_s,
                keep_temp,
            ],
            outputs=[output_video, status, logs, output_srt],
        )

    return demo


def main() -> None:
    app = build_ui()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
