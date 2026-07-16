"""Cache signature and resume directory management."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from autodub.segments import Segment
from autodub.config import TTS_POSTPROCESS_VERSION


def build_dub_cache_signature(
    segments: Iterable[Segment],
    target_lang: str,
    tts_engine: str,
    use_page_tts_profile: bool,
    edge_voice: str | None,
    min_stretch_speed: float,
    max_stretch_speed: float,
    silence_trim_ms: int,
    coqui_model: str | None = None,
    coqui_speaker_fingerprint: str | None = None,
) -> str:
    payload = {
        "tts_postprocess_version": TTS_POSTPROCESS_VERSION,
        "target_lang": target_lang,
        "tts_engine": tts_engine,
        "use_page_tts_profile": bool(use_page_tts_profile),
        "edge_voice": edge_voice or "",
        "coqui_model": coqui_model or "",
        "coqui_speaker_fingerprint": coqui_speaker_fingerprint or "",
        "min_stretch_speed": round(min_stretch_speed, 4),
        "max_stretch_speed": round(max_stretch_speed, 4),
        "silence_trim_ms": int(silence_trim_ms),
        "segments": [asdict(seg) for seg in segments],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def build_resume_dir(
    input_path: Path,
    output_dir: Path,
    target_lang: str,
    whisper_model: str,
    translation_provider: str,
    tts_engine: str,
    edge_voice: str | None,
    optimization_profile: str,
    start_time_s: float,
    end_time_s: float | None,
    glossary_text: str,
    asr_engine: str = "whisper",
) -> Path:
    input_fingerprint = "unknown"
    try:
        file_size = input_path.stat().st_size
        hasher = hashlib.sha1()
        hasher.update(str(file_size).encode("utf-8"))
        with input_path.open("rb") as handle:
            head = handle.read(1_048_576)
            hasher.update(head)
            if file_size > 1_048_576:
                handle.seek(max(file_size - 1_048_576, 0))
                tail = handle.read(1_048_576)
                hasher.update(tail)
        input_fingerprint = hasher.hexdigest()[:16]
    except OSError:
        input_fingerprint = "unreadable"

    job_signature = "|".join([
        str(input_path.resolve()),
        input_fingerprint,
        target_lang,
        whisper_model,
        asr_engine,
        translation_provider,
        tts_engine,
        edge_voice or "",
        optimization_profile,
        f"{start_time_s:.3f}",
        "none" if end_time_s is None else f"{end_time_s:.3f}",
        hashlib.sha1((glossary_text or "").encode("utf-8")).hexdigest()[:10],
    ])
    short_hash = hashlib.sha1(job_signature.encode("utf-8")).hexdigest()[:12]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", input_path.stem)[:40] or "video"
    return output_dir / ".autodub_resume" / f"{safe_stem}_{target_lang}_{short_hash}"