"""Segment data model and segment-related utility functions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Segment:
    start_s: float
    end_s: float
    source_text: str
    translated_text: str = ""


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: Iterable[Segment], srt_path: Path) -> None:
    lines: List[str] = []
    for index, seg in enumerate(segments, start=1):
        subtitle_text = (safe_text(seg.translated_text) or safe_text(seg.source_text)).strip()
        if not subtitle_text:
            continue
        lines.extend(
            [
                str(index),
                f"{format_srt_timestamp(seg.start_s)} --> {format_srt_timestamp(seg.end_s)}",
                subtitle_text,
                "",
            ]
        )
    srt_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_transcript_txt(
    segments: Iterable[Segment],
    transcript_path: Path,
    *,
    translated: bool = False,
) -> None:
    lines: List[str] = []
    for seg in segments:
        text = safe_text(seg.translated_text if translated else seg.source_text).strip()
        if not text:
            continue
        lines.append(text)

    transcript_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def normalize_subtitle_for_dedupe(text: str) -> str:
    normalized = re.sub(r"\s+", " ", safe_text(text)).strip().lower()
    return re.sub(r"[^\w\s]", "", normalized)


def collapse_consecutive_duplicate_segments(
    segments: List[Segment], max_gap_s: float = 0.35
) -> List[Segment]:
    if not segments:
        return segments

    ordered = sorted(segments, key=lambda seg: seg.start_s)
    merged: List[Segment] = [ordered[0]]

    for seg in ordered[1:]:
        prev = merged[-1]
        prev_text = normalize_subtitle_for_dedupe(prev.translated_text or prev.source_text)
        current_text = normalize_subtitle_for_dedupe(seg.translated_text or seg.source_text)

        is_duplicate_text = bool(prev_text) and prev_text == current_text
        close_enough = seg.start_s <= prev.end_s + max_gap_s

        if is_duplicate_text and close_enough:
            prev.end_s = max(prev.end_s, seg.end_s)
            if len(safe_text(seg.translated_text).strip()) > len(safe_text(prev.translated_text).strip()):
                prev.translated_text = safe_text(seg.translated_text)
            if len(safe_text(seg.source_text).strip()) > len(safe_text(prev.source_text).strip()):
                prev.source_text = safe_text(seg.source_text)
            continue

        merged.append(seg)

    return merged


def save_segments_to_json(segments: Iterable[Segment], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(seg) for seg in segments], indent=2), encoding="utf-8"
    )


def load_segments_from_json(input_path: Path) -> List[Segment]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    return [
        Segment(
            start_s=float(item["start_s"]),
            end_s=float(item["end_s"]),
            source_text=safe_text(item.get("source_text", "")),
            translated_text=safe_text(item.get("translated_text", "")),
        )
        for item in raw
    ]