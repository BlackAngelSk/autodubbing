"""FFmpeg/FFprobe utility functions for audio extraction, video trimming, and command execution."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


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


def probe_media_duration(media_path: Path) -> float | None:
    """Best-effort media duration probe used for auto optimization decisions."""
    if shutil.which("ffprobe") is None:
        return None

    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    value = completed.stdout.strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def extract_audio(video_path: Path, audio_out: Path) -> None:
    run_cmd([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(audio_out),
    ])


def trim_video(input_video: Path, output_video: Path, start_time_s: float, end_time_s: float | None) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(input_video), "-ss", f"{start_time_s}"]
    if end_time_s is not None:
        cmd.extend(["-to", f"{end_time_s}"])
    cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(output_video)])
    run_cmd(cmd)