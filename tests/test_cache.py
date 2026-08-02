"""Tests for autodub.cache module."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodub.segments import Segment
from autodub.cache import build_dub_cache_signature, build_resume_dir


class TestBuildDubCacheSignature:
    def test_returns_string(self, sample_segments):
        sig = build_dub_cache_signature(
            sample_segments,
            target_lang="sk",
            tts_engine="edge",
            use_page_tts_profile=False,
            edge_voice=None,
            min_stretch_speed=0.85,
            max_stretch_speed=1.35,
            silence_trim_ms=0,
        )
        assert isinstance(sig, str)
        assert len(sig) == 40  # SHA1 hex digest

    def test_same_input_same_sig(self, sample_segments):
        sig1 = build_dub_cache_signature(
            sample_segments, "sk", "edge", False, None, 0.85, 1.35, 0
        )
        sig2 = build_dub_cache_signature(
            sample_segments, "sk", "edge", False, None, 0.85, 1.35, 0
        )
        assert sig1 == sig2

    def test_different_lang_different_sig(self, sample_segments):
        sig1 = build_dub_cache_signature(
            sample_segments, "sk", "edge", False, None, 0.85, 1.35, 0
        )
        sig2 = build_dub_cache_signature(
            sample_segments, "fr", "edge", False, None, 0.85, 1.35, 0
        )
        assert sig1 != sig2


class TestBuildResumeDir:
    def test_returns_path(self, tmp_path, sample_segments):
        result = build_resume_dir(
            input_path=Path("nonexistent.mp4"),
            output_dir=tmp_path,
            target_lang="sk",
            whisper_model="small",
            translation_provider="google",
            tts_engine="edge",
            edge_voice=None,
            optimization_profile="auto",
            start_time_s=0.0,
            end_time_s=None,
            glossary_text="",
            asr_engine="whisper",
        )
        assert isinstance(result, Path)
        assert ".autodub_resume" in str(result)