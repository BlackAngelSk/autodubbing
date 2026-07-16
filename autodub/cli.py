"""Command-line interface for the auto-dubbing pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from autodub.pipeline import autodub_video
from autodub.config import SUPPORTED_LANGUAGES


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autodub",
        description="Automatic video dubbing: transcribe, translate, and re-synthesize speech.",
    )
    parser.add_argument("input", type=Path, help="Path to the input video file.")
    parser.add_argument("output", type=Path, help="Path to the output video file.")
    parser.add_argument(
        "--lang", "-l", type=str, default="en",
        help=f"Target language code. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="small",
        help="Whisper model name (e.g. 'base', 'small', 'medium', 'large-v3').",
    )
    parser.add_argument(
        "--device", "-d", type=str, default="auto",
        help="Compute device: 'auto', 'cuda', 'cpu', or 'rocm'.",
    )
    parser.add_argument(
        "--translation-provider", "-t", type=str, default="google",
        choices=["google", "mymemory"],
        help="Translation provider to use.",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="Hugging Face token for faster Whisper model downloads.",
    )
    parser.add_argument(
        "--tts-engine", type=str, default="edge",
        choices=["edge", "edge_human", "gtts", "coqui"],
        help="TTS engine to use for speech synthesis.",
    )
    parser.add_argument(
        "--use-page-tts-profile", action="store_true",
        help="Use the UI Text-to-Speech tab defaults for TTS profile.",
    )
    parser.add_argument(
        "--edge-voice", type=str, default=None,
        help="Edge TTS voice override (e.g. 'en-US-AriaNeural').",
    )
    parser.add_argument(
        "--background-mix-level", type=float, default=0.03,
        help="Mix level for original background audio (0.0 to 1.0).",
    )
    parser.add_argument(
        "--no-original-audio", action="store_true",
        help="Mute original audio in output, using only dubbed speech.",
    )
    parser.add_argument(
        "--min-stretch-speed", type=float, default=0.85,
        help="Minimum stretching speed for audio fitting.",
    )
    parser.add_argument(
        "--max-stretch-speed", type=float, default=1.35,
        help="Maximum stretching speed for audio fitting.",
    )
    parser.add_argument(
        "--silence-trim-ms", type=int, default=0,
        help="Silence trimming amount in milliseconds.",
    )
    parser.add_argument(
        "--profile", type=str, default="auto",
        choices=["auto", "short", "balanced", "long"],
        help="Optimization profile for processing.",
    )
    parser.add_argument(
        "--no-srt", action="store_true",
        help="Do not export SRT subtitle file.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Disable cache/resume functionality.",
    )
    parser.add_argument(
        "--glossary", type=str, default="",
        help="Glossary overrides text (source => translation, one per line).",
    )
    parser.add_argument(
        "--asr-engine", type=str, default="auto",
        choices=["auto", "whisper", "stable-ts"],
        help="ASR engine to use for transcription.",
    )
    parser.add_argument(
        "--start-time", type=float, default=0.0,
        help="Start time in seconds for processing a sub-clip.",
    )
    parser.add_argument(
        "--end-time", type=float, default=None,
        help="End time in seconds for processing a sub-clip.",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep temporary files after processing.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        result = autodub_video(
            input_path=args.input,
            output_path=args.output,
            target_lang=args.lang,
            whisper_model=args.model,
            device=args.device,
            translation_provider=args.translation_provider,
            hf_token=args.hf_token,
            tts_engine=args.tts_engine,
            use_page_tts_profile=args.use_page_tts_profile,
            edge_voice=args.edge_voice,
            background_mix_level=args.background_mix_level,
            include_original_audio=not args.no_original_audio,
            min_stretch_speed=args.min_stretch_speed,
            max_stretch_speed=args.max_stretch_speed,
            silence_trim_ms=args.silence_trim_ms,
            optimization_profile=args.profile,
            export_srt=not args.no_srt,
            resume_enabled=not args.no_resume,
            glossary_text=args.glossary,
            asr_engine=args.asr_engine,
            start_time_s=args.start_time,
            end_time_s=args.end_time,
            keep_temp=args.keep_temp,
        )
        return result
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        if args.verbose:
            logging.exception(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())