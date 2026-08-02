"""Tests for autodub.translate module."""

from unittest.mock import MagicMock, patch

import pytest

from autodub.segments import Segment
from autodub.translate import (
    build_translator,
    parse_glossary_overrides,
    apply_glossary_overrides,
    english_word_tokens,
    has_untranslated_english_tokens,
    replace_untranslated_tokens,
    split_for_translation,
    safe_translate,
    cached_translation_looks_poor,
    translation_looks_wrong_language,
)


class TestBuildTranslator:
    def test_google(self):
        t = build_translator("google", source="en", target="sk")
        assert t is not None

    def test_mymemory(self):
        t = build_translator("mymemory", source="en", target="sk")
        assert t is not None

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            build_translator("unsupported", source="en", target="sk")


class TestParseGlossaryOverrides:
    def test_none(self):
        assert parse_glossary_overrides(None) == {}

    def test_empty(self):
        assert parse_glossary_overrides("") == {}

    def test_with_arrow(self):
        result = parse_glossary_overrides("apple => jablko\nbanana => banan")
        assert result["apple"] == "jablko"
        assert result["banana"] == "banan"

    def test_with_hash_comments(self):
        result = parse_glossary_overrides("# comment\npear => hruska")
        assert result["pear"] == "hruska"

    def test_skips_invalid(self):
        result = parse_glossary_overrides("invalid line\nfoo => bar")
        assert len(result) == 1


class TestApplyGlossaryOverrides:
    def test_replaces(self):
        text = "I like apple and banana"
        overrides = {"apple": "jablko", "banana": "banan"}
        result = apply_glossary_overrides(text, overrides)
        assert "jablko" in result
        assert "banan" in result


class TestEnglishWordTokens:
    def test_basic(self):
        assert "hello" in english_word_tokens("Hello world!")
        assert len(english_word_tokens("Hi")) == 0  # 2 chars minimum


class TestHasUntranslatedEnglishTokens:
    def test_same_is_untranslated(self):
        assert has_untranslated_english_tokens("Hello world", "Hello world", "sk")

    def test_different_is_fine(self):
        assert not has_untranslated_english_tokens("Hello world", "Ahoj svet", "sk")

    def test_english_to_english(self):
        assert not has_untranslated_english_tokens("Hello", "Hello", "en")


class TestSplitForTranslation:
    def test_short_text(self):
        assert split_for_translation("Hello world") == ["Hello world"]

    def test_long_text(self):
        text = "This is a very long sentence. " * 5
        chunks = split_for_translation(text)
        assert len(chunks) > 1


class TestSafeTranslate:
    def test_empty_returns_empty(self):
        result = safe_translate("", MagicMock(), MagicMock())
        assert result == ""


class TestCachedTranslationLooksPoor:
    def test_empty(self):
        assert not cached_translation_looks_poor([], "sk")

    def test_english(self):
        segs = [Segment(0, 1, "Hello", "Hello")]
        assert not cached_translation_looks_poor(segs, "en")

    def test_poor(self):
        segs = [Segment(0, 1, "Hello", "Hello"), Segment(1, 2, "World", "World")]
        assert cached_translation_looks_poor(segs, "sk")


class TestTranslationLooksWrongLanguage:
    def test_english_fine(self):
        assert not translation_looks_wrong_language([], "en")

    def test_empty_segments(self):
        segs = [Segment(0, 1, "Hello", "")]
        assert translation_looks_wrong_language(segs, "sk")

    def test_script_pattern(self):
        segs = [Segment(0, 1, "Hello", "Привет мирolutely long text to check")]
        assert not translation_looks_wrong_language(segs, "ru")