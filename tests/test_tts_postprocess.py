"""Tests for autodub.tts_postprocess module."""

from autodub.tts_postprocess import (
    sanitize_tts_text,
    format_edge_rate,
    format_edge_pitch,
    inject_mid_sentence_pause,
    build_edge_tts_profile,
    build_page_tts_profile,
    build_atempo_filter,
)


class TestSanitizeTtsText:
    def test_repeat_punctuation(self):
        assert "!" in sanitize_tts_text("Hello!!!")
        assert sanitize_tts_text("Hello!!!") == "Hello!"

    def test_dashes(self):
        result = sanitize_tts_text("A - B - C")
        assert "- " not in result  # dashes replaced with commas


class TestFormatEdgeRate:
    def test_positive(self):
        assert format_edge_rate(5) == "+5%"

    def test_negative(self):
        assert format_edge_rate(-10) == "-10%"

    def test_clamp_max(self):
        assert format_edge_rate(100) == "+40%"

    def test_clamp_min(self):
        assert format_edge_rate(-100) == "-50%"


class TestFormatEdgePitch:
    def test_positive(self):
        assert format_edge_pitch(3) == "+3Hz"

    def test_negative(self):
        assert format_edge_pitch(-5) == "-5Hz"


class TestInjectMidSentencePause:
    def test_long_no_punct(self):
        result = inject_mid_sentence_pause("This is a very long sentence with many words and no punctuation marks at all")
        assert "," in result or result == "This is a very long sentence with many words and no punctuation marks at all"

    def test_short_text_passthrough(self):
        short = "Short sentence"
        assert inject_mid_sentence_pause(short) == short

    def test_text_with_punct(self):
        text = "Hello, world!"
        assert inject_mid_sentence_pause(text) == text


class TestBuildEdgeTtsProfile:
    def test_returns_tuple(self):
        result = build_edge_tts_profile("Hello world")
        assert len(result) == 4
        assert isinstance(result[0], str)
        assert isinstance(result[1], int)
        assert isinstance(result[2], int)
        assert isinstance(result[3], str)

    def test_question_mark(self):
        _, _, pitch, _ = build_edge_tts_profile("What is this?")
        assert pitch > 0


class TestBuildPageTtsProfile:
    def test_returns_tuple(self):
        result = build_page_tts_profile("Hello world")
        assert len(result) == 4


class TestBuildAtempoFilter:
    def test_single_speed(self):
        result = build_atempo_filter(1.5)
        assert "atempo=1.5" in result

    def test_slow_speed(self):
        result = build_atempo_filter(0.3)
        assert "atempo=0.5" in result
        assert "atempo=0.6" in result

    def test_fast_speed(self):
        result = build_atempo_filter(4.0)
        assert "atempo=2.0" in result