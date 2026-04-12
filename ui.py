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

from autodub import DEFAULT_EDGE_VOICES, autodub_video, detect_cuda_available, detect_rocm_available


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
    ("Whisper (default, reliable)", "whisper"),
    ("stable-ts (accurate timestamps, fixes missed opening)", "stable-ts"),
]

LARGE_WHISPER_MODELS = {"large-v2", "large-v3", "large-v3-turbo", "distil-large-v3"}
AUTO_DEVICE = "cuda" if detect_cuda_available() else ("rocm" if detect_rocm_available() else "cpu")


APP_CSS = """
:root {
    --ad-bg: #f4f6fb;
    --ad-panel: #ffffff;
    --ad-text: #111827;
    --ad-muted: #374151;
    --ad-border: #cbd5e1;
    --ad-input-bg: #ffffff;
    --ad-input-text: #111827;
    --ad-accent: #2563eb;
    --ad-accent-text: #ffffff;
    --ad-option-bg: #f8fafc;
    --ad-option-text: #111827;
}

html.ad-dark,
body.ad-dark {
    --ad-bg: #0b1220;
    --ad-panel: #111827;
    --ad-text: #f3f4f6;
    --ad-muted: #d1d5db;
    --ad-border: #334155;
    --ad-input-bg: #0f172a;
    --ad-input-text: #f8fafc;
    --ad-accent: #3b82f6;
    --ad-accent-text: #ffffff;
    --ad-option-bg: #1e293b;
    --ad-option-text: #f8fafc;
}

.gradio-container {
    background: var(--ad-bg) !important;
    color: var(--ad-text) !important;

    --body-background-fill: var(--ad-bg);
    --body-text-color: var(--ad-text);
    --body-text-color-subdued: var(--ad-muted);
    --background-fill-primary: var(--ad-bg);
    --background-fill-secondary: var(--ad-panel);
    --block-background-fill: var(--ad-panel);
    --block-border-color: var(--ad-border);
    --block-title-text-color: var(--ad-text);
    --block-label-text-color: var(--ad-text);
    --block-info-text-color: var(--ad-muted);
    --input-background-fill: var(--ad-input-bg);
    --input-border-color: var(--ad-border);
    --input-text-color: var(--ad-input-text);
    --input-placeholder-color: var(--ad-muted);
    --button-primary-background-fill: var(--ad-accent);
    --button-primary-text-color: var(--ad-accent-text);
    --button-secondary-background-fill: var(--ad-option-bg);
    --button-secondary-text-color: var(--ad-option-text);
    --button-secondary-border-color: var(--ad-border);
}

#theme-mode {
    border: 1px solid var(--ad-border);
    border-radius: 10px;
    padding: 8px;
    background: var(--ad-panel);
}

#theme-mode label {
    font-size: 12px;
    color: var(--ad-text);
}

/* Force option text visibility in setting button groups/radios. */
.gradio-container .gradio-radio label,
.gradio-container .gradio-radio label *,
.gradio-container .gradio-checkbox label,
.gradio-container .gradio-checkbox label * {
    color: var(--ad-option-text) !important;
    opacity: 1 !important;
}

.gradio-container .gradio-radio label {
    background: var(--ad-option-bg) !important;
    border: 1px solid var(--ad-border) !important;
}

.gradio-container .gradio-radio label:has(input:checked),
.gradio-container .gradio-radio label[aria-pressed="true"],
.gradio-container button[aria-pressed="true"] {
    background: var(--ad-accent) !important;
    border-color: var(--ad-accent) !important;
}

.gradio-container .gradio-radio label:has(input:checked) *,
.gradio-container .gradio-radio label[aria-pressed="true"] *,
.gradio-container button[aria-pressed="true"] * {
    color: var(--ad-accent-text) !important;
}

.gradio-container .info,
.gradio-container .hint,
.gradio-container [data-testid="block-info"],
.gradio-container [class*="description"] {
    color: var(--ad-muted) !important;
    opacity: 1 !important;
}
"""


def voice_for_language(lang: str) -> str:
    return DEFAULT_EDGE_VOICES.get(lang, DEFAULT_EDGE_VOICES["en"])


def build_runtime_note(
    whisper_model: str,
    device: str,
    optimization_profile: str,
    tts_engine: str,
) -> str:
    resolved_device = AUTO_DEVICE if device == "auto" else device
    speed_label = "fast" if whisper_model in {"tiny", "base"} else "balanced" if whisper_model in {"small", "medium", "distil-large-v3"} else "slowest"

    lines = [
        f"Auto-detected hardware target: `{AUTO_DEVICE}`.",
        f"Expected transcription pace: **{speed_label}**.",
        f"Optimization profile: `{optimization_profile}`.",
    ]
    if whisper_model in LARGE_WHISPER_MODELS:
        lines.append("First use of this model may download several GB of Whisper weights.")
        lines.append("Optional: set `HF_TOKEN` or paste a Hugging Face token below to avoid the auth warning and improve rate limits.")
    if resolved_device not in {"cuda", "rocm"} and whisper_model in LARGE_WHISPER_MODELS:
        lines.append("Large Whisper models on CPU can take a long time; `medium` is usually the safer fallback.")
    if tts_engine != "edge":
        lines.append("`gtts` is simpler but usually sounds less natural than `edge`.")

    return "### Smart guidance\n" + "\n".join(f"- {line}" for line in lines)


def _update_edge_voice_dropdown(lang: str, tts_engine: str) -> gr.Dropdown:
    return gr.Dropdown(
        choices=EDGE_VOICE_CHOICES,
        value=voice_for_language(lang),
        interactive=tts_engine == "edge",
    )


def on_quality_preset_change(preset: str, tts_engine: str) -> tuple[gr.Dropdown, gr.Dropdown, gr.Radio, gr.Markdown]:
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
        gr.Radio(choices=["auto", "cpu", "cuda"], value=device_value),
        gr.Markdown(value=build_runtime_note(whisper_value, device_value, profile_value, tts_engine)),
    )


def on_settings_change(
    whisper_model: str,
    device: str,
    optimization_profile: str,
    tts_engine: str,
) -> gr.Markdown:
    return gr.Markdown(value=build_runtime_note(whisper_model, device, optimization_profile, tts_engine))


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
    asr_engine: str = "whisper",
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


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Auto Dubbing Studio") as demo:
        with gr.Row():
            theme_mode = gr.Radio(
                label="Theme",
                choices=[("Light", "light"), ("Dark", "dark")],
                value="light",
                elem_id="theme-mode",
                scale=1,
            )

        def keep_theme_value(mode: str) -> str:
            return mode

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
                with gr.Accordion("Core", open=True):
                    quality_preset = gr.Radio(
                        label="Quality Preset",
                        choices=QUALITY_PRESET_CHOICES,
                        value="balanced",
                        info="Quickly choose fast, balanced, or best-quality defaults",
                    )
                    target_lang = gr.Dropdown(
                        label="Target Language",
                        choices=LANGUAGE_CHOICES,
                        value="es",
                        info="Pick the language for dubbed speech",
                    )
                    whisper_model = gr.Dropdown(
                        label="Whisper Model",
                        choices=WHISPER_MODEL_CHOICES,
                        value="small",
                        info="Larger models are more accurate; large-v3 is best on GPU",
                    )
                    asr_engine = gr.Radio(
                        label="ASR Engine",
                        choices=ASR_ENGINE_CHOICES,
                        value="whisper",
                        info="stable-ts fixes missed opening words and produces more precise timestamps. Install with: pip install stable-ts",
                    )
                    device = gr.Radio(
                        label="Device",
                        choices=["auto", "cpu", "cuda", "rocm"],
                        value="auto",
                        info=f"Auto currently resolves to {AUTO_DEVICE} on this machine",
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
                        info="Auto tunes for short clips or long videos",
                    )
                    hardware_hint = gr.Markdown(build_runtime_note("small", "auto", "auto", "edge"))

                with gr.Accordion("Voice", open=True):
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

                with gr.Accordion("Translation", open=False):
                    translation_provider = gr.Radio(
                        label="Translation Provider",
                        choices=TRANSLATION_PROVIDER_CHOICES,
                        value="google",
                        info="Google is the default; MyMemory is a good alternate when needed",
                    )
                    glossary_text = gr.Textbox(
                        label="Glossary Overrides",
                        lines=4,
                        placeholder="death => smrti\nwar => vojna",
                        value="",
                        info="Optional forced translations, one rule per line",
                    )

                with gr.Accordion("Output", open=False):
                    include_original_audio = gr.Checkbox(
                        label="Keep original audio quietly in background",
                        value=True,
                        info="Turn off for dubbed voice only",
                    )
                    export_srt = gr.Checkbox(
                        label="Export translated subtitles (.srt)",
                        value=True,
                    )

                with gr.Accordion("Advanced", open=False):
                    hf_token = gr.Textbox(
                        label="Hugging Face Token (optional)",
                        type="password",
                        placeholder="hf_...",
                        info="Avoids the HF Hub auth warning and can improve Whisper download rate limits",
                    )
                    resume_enabled = gr.Checkbox(
                        label="Resume previous job if possible",
                        value=True,
                        info="Reuses cached chunks and translations on reruns",
                    )
                    keep_temp = gr.Checkbox(label="Keep temp files", value=False)

                with gr.Accordion("Custom Time Range", open=False):
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

        target_lang.change(fn=_update_edge_voice_dropdown, inputs=[target_lang, tts_engine], outputs=[edge_voice])
        tts_engine.change(fn=_update_edge_voice_dropdown, inputs=[target_lang, tts_engine], outputs=[edge_voice])
        quality_preset.change(
            fn=on_quality_preset_change,
            inputs=[quality_preset, tts_engine],
            outputs=[whisper_model, optimization_profile, device, hardware_hint],
        )
        whisper_model.change(
            fn=on_settings_change,
            inputs=[whisper_model, device, optimization_profile, tts_engine],
            outputs=[hardware_hint],
        )
        device.change(
            fn=on_settings_change,
            inputs=[whisper_model, device, optimization_profile, tts_engine],
            outputs=[hardware_hint],
        )
        optimization_profile.change(
            fn=on_settings_change,
            inputs=[whisper_model, device, optimization_profile, tts_engine],
            outputs=[hardware_hint],
        )
        tts_engine.change(
            fn=on_settings_change,
            inputs=[whisper_model, device, optimization_profile, tts_engine],
            outputs=[hardware_hint],
        )

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

        demo.load(
            fn=lambda: "light",
            inputs=None,
            outputs=[theme_mode],
            js="""
() => {
    const key = 'autodub-theme-mode';
    const saved = localStorage.getItem(key) === 'dark' ? 'dark' : 'light';
    const dark = saved === 'dark';
    document.documentElement.classList.toggle('ad-dark', dark);
    document.body.classList.toggle('ad-dark', dark);
    return [saved];
}
            """,
        )

        theme_mode.change(
            fn=keep_theme_value,
            inputs=[theme_mode],
            outputs=[theme_mode],
            js="""
(mode) => {
    const normalized = mode === 'dark' ? 'dark' : 'light';
    const dark = normalized === 'dark';
    document.documentElement.classList.toggle('ad-dark', dark);
    document.body.classList.toggle('ad-dark', dark);
    localStorage.setItem('autodub-theme-mode', normalized);
    return [normalized];
}
            """,
        )

    return demo


def main() -> None:
    app = build_ui()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True, css=APP_CSS)


if __name__ == "__main__":
    main()
