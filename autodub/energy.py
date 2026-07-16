"""Audio energy utilities: normalize video energy and adjust segment volumes."""

from __future__ import annotations

import logging
from pathlib import Path

from pydub import AudioSegment
from pydub.effects import normalize

logger = logging.getLogger(__name__)


def normalize_video_energy(video_path: Path, output_path: Path) -> None:
    """Normalize vol/video loudness for stable dubbing.
    
    This re-encodes the audio track after normalizing dB levels to
    achieve more consistent dubbing quality.
    """
    try:
        from autodub.ffmpeg_utils import run_cmd

        run_cmd([
            "ffmpeg", "-y", "-i", str(video_path),
            "-af", "loudnorm=I=-18:TP=-2:LRA=11",
            "-c:v", "copy", str(output_path),
        ])
    except Exception as exc:
        logger.error("Failed to normalize video energy: %s", exc)
        raise


def adjust_segment_energy(
    segment: AudioSegment,
    target_db: float = -20.0,
    headroom: float = 1.0,
) -> AudioSegment:
    """Adjust a single audio segment to a target dB level with headroom."""
    if len(segment) <= 0:
        return segment

    target_db = min(max(target_db, -60.0), 0.0)
    headroom = max(min(headroom, 3.0), 0.5)

    normalized = normalize(segment, headroom=headroom)
    offset_db = target_db - normalized.dBFS
    normalized = normalized + offset_db

    return normalized


def detect_meaningful_energy(
    audio: AudioSegment,
    min_db: float = -35.0,
) -> bool:
    """Return True if the audio contains enough energy to be considered speech."""
    if len(audio) <= 0:
        return False

    from pydub.silence import detect_nonsilent

    ranges = detect_nonsilent(audio, min_silence_len=150, silence_thresh=int(min_db))
    nonsilent_ms = sum(max(end - start, 0) for start, end in ranges)
    return nonsilent_ms >= 400