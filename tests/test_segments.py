"""Tests for autodub.segments module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from autodub.segments import (
    Segment,
    safe_text,
    format_srt_timestamp,
    write_srt,
    write_transcript_txt,
    normalize_subtitle_for_dedupe,
    collapse_consecutive_duplicate_segments,
    save_segments_to_json,
    load_segments_from_json,
)


class TestSafeText:
    def test_none_returns_empty(self):
        assert safe_text(None) == ""

    def test_string_passthrough(self):
        assert safe_text("hello") == "hello"

    def test_non_string_converted(self):
        assert safe_text(123) == "123"


class TestFormatSrtTimestamp:
    def test_zero(self):
        assert format_srt_timestamp(0) == "00:00:00,000"

    def test_large_value(self):
        assert format_srt_timestamp(3723.456) == "01:02:03,456"

    def test_negative_clamped(self):
        assert format_srt_timestamp(-5) == "00:00:00,000"


class TestWriteSrt:
    def test_creates_file(self, tmp_path):
        segs = [
            Segment(start_s=0.0, end_s=2.0, source_text="Hello", translated_text="Ahoj"),
            Segment(start_s=3.0, end_s=5.0, source_text="World", translated_text="Svet"),
        ]
        out = tmp_path / "out.srt"
        write_srt(segs, out)
        text = out.read_text(encoding="utf-8")
        assert "1\n" in text
        assert "Ahoj" in text
        assert "Svet" in text

    def test_skips_empty(self, tmp_path):
        segs = [
            Segment(start_s=0.0, end_s=2.0, source_text="", translated_text="")
        ]
        out = tmp_path / "empty.srt"
        write_srt(segs, out)
        text = out.read_text(encoding="utf-8").strip()
        assert text == ""


class TestWriteTranscriptTxt:
    def test_source_text(self, tmp_path):
        segs = [Segment(start_s=0, end_s=1, source_text="Hi"), Segment(start_s=1, end_s=2, source_text="Bye")]
        out = tmp_path / "transcript.txt"
        write_transcript_txt(segs, out)
        assert "Hi" in out.read_text()
        assert "Bye" in out.read_text()

    def test_translated_text(self, tmp_path):
        segs = [Segment(start_s=0, end_s=1, source_text="Hi", translated_text="Ahoj")]
        out = tmp_path / "translated.txt"
        write_transcript_txt(segs, out, translated=True)
        assert "Ahoj" in out.read_text()


class TestNormalizeSubtitleForDedupe:
    def test_normalizes(self):
        assert normalize_subtitle_for_dedupe("  Hello   World! ") == "hello world"

    def test_empty(self):
        assert normalize_subtitle_for_dedupe("") == ""


class TestCollapseConsecutiveDuplicateSegments:
    def test_collapses(self):
        segs = [
            Segment(start_s=0, end_s=1, source_text="A", translated_text="B"),
            Segment(start_s=0.5, end_s=2, source_text="a", translated_text="b"),
        ]
        assert len(collapse_consecutive_duplicate_segments(segs)) == 1

    def test_keeps_different(self):
        segs = [
            Segment(start_s=0, end_s=1, source_text="A", translated_text="B"),
            Segment(start_s=2, end_s=3, source_text="C", translated_text="D"),
        ]
        assert len(collapse_consecutive_duplicate_segments(segs)) == 2

    def test_empty_list(self):
        assert collapse_consecutive_duplicate_segments([]) == []


class TestSaveLoadJson:
    def test_round_trip(self, tmp_path):
        segs = [
            Segment(start_s=0, end_s=1, source_text="A", translated_text="B"),
            Segment(start_s=2, end_s=3, source_text="C", translated_text="D"),
        ]
        out = tmp_path / "segments.json"
        save_segments_to_json(segs, out)
        loaded = load_segments_from_json(out)
        assert len(loaded) == 2
        assert loaded[0].source_text == "A"
        assert loaded[1].translated_text == "D"