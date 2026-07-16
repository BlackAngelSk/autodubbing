"""Video muxing: merge dubbed audio track back into video."""

from __future__ import annotations

import logging
from pathlib import Path

from autodub.ffmpeg_utils import run_cmd

logger = logging.getLogger(__name__)


def mux_video_with_dub(
    input_video: Path,
    dubbed_wav: Path,
    output_video: Path,
    background_mix_level: float = 0.03,
    include_original_audio: bool = True,
) -> None:
    def build_dub_only_cmd() -> list[str]:
        return [
            "ffmpeg", "-y", "-i", str(input_video), "-i", str(dubbed_wav),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output_video),
        ]

    if include_original_audio:
        mix = min(max(background_mix_level, 0.0), 1.0)
        advanced_cmd = [
            "ffmpeg", "-y", "-i", str(input_video), "-i", str(dubbed_wav),
            "-filter_complex",
            (
                f"[1:a]aresample=48000,acompressor=threshold=-22dB:ratio=2.8:attack=12:release=180,"
                f"alimiter=limit=0.97,volume=2.6[dub];"
                f"[0:a]aresample=48000,highpass=f=120,lowpass=f=6800,volume={mix:.3f}[bg];"
                f"[bg][dub]sidechaincompress=threshold=0.015:ratio=14:attack=16:release=320:makeup=1.25[ducked];"
                f"[ducked][dub]amix=inputs=2:weights='0.12 1.88':duration=first:normalize=0[aout]"
            ),
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output_video),
        ]
        simple_cmd = [
            "ffmpeg", "-y", "-i", str(input_video), "-i", str(dubbed_wav),
            "-filter_complex",
            f"[0:a]volume={mix:.3f}[bg];[1:a]volume=2.6[dub];[bg][dub]amix=inputs=2:weights='0.10 1.90':duration=first:normalize=0[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output_video),
        ]

        try:
            run_cmd(advanced_cmd)
            return
        except RuntimeError as exc:
            logger.warning("Advanced audio ducking mix failed; retrying with simple amix. Error: %s", exc)
            try:
                run_cmd(simple_cmd)
                return
            except RuntimeError as fallback_exc:
                logger.warning("Simple amix fallback failed; exporting dubbed-only audio. Error: %s", fallback_exc)
                run_cmd(build_dub_only_cmd())
                return
    else:
        run_cmd(build_dub_only_cmd())