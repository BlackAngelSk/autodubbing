"""Translation logic: Google/MyMemory translation, glossary overrides, retry logic, and validation."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List

from deep_translator import GoogleTranslator, MyMemoryTranslator
from tqdm import tqdm

from autodub.config import TRANSLATION_PROVIDERS
from autodub.segments import Segment, safe_text

logger = logging.getLogger(__name__)


def build_translator(provider: str, source: str, target: str) -> Any:
    normalized = provider.strip().lower() if provider else "google"
    if normalized not in TRANSLATION_PROVIDERS:
        raise ValueError(f"Unsupported translation provider: {provider}")
    if normalized == "mymemory":
        try:
            return MyMemoryTranslator(source=source, target=target)
        except Exception:
            return GoogleTranslator(source=source, target=target)
    return GoogleTranslator(source=source, target=target)


def parse_glossary_overrides(glossary_text: str | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not glossary_text:
        return overrides

    for raw_line in glossary_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        separator = "=>" if "=>" in line else "->" if "->" in line else "=" if "=" in line else None
        if separator is None:
            continue

        source, replacement = (part.strip() for part in line.split(separator, 1))
        if source and replacement:
            overrides[source.lower()] = replacement

    return overrides


def apply_glossary_overrides(text: str, overrides: dict[str, str]) -> str:
    adjusted = text
    for source, replacement in sorted(overrides.items(), key=lambda item: len(item[0]), reverse=True):
        escaped = re.escape(source)
        pattern = escaped if " " in source else rf"\b{escaped}\b"
        adjusted = re.sub(pattern, replacement, adjusted, flags=re.IGNORECASE)
    return adjusted


def english_word_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
    }


def has_untranslated_english_tokens(source_text: str, translated_text: str, target_lang: str) -> bool:
    if target_lang == "en":
        return False

    source_tokens = english_word_tokens(source_text)
    translated_tokens = english_word_tokens(translated_text)
    if not source_tokens or not translated_tokens:
        return False

    common_tokens = source_tokens & translated_tokens
    ignored_tokens = {"oh", "yeah", "hey", "la", "na"}
    return any(token not in ignored_tokens for token in common_tokens)


def replace_untranslated_tokens(
    source_text: str,
    translated_text: str,
    word_translator: Any,
) -> str:
    source_tokens = english_word_tokens(source_text)
    translated_tokens = english_word_tokens(translated_text)
    common_tokens = [token for token in source_tokens & translated_tokens if token not in {"oh", "yeah", "hey", "la", "na"}]

    repaired_text = translated_text
    for token in sorted(common_tokens, key=len, reverse=True):
        try:
            replacement = word_translator.translate(token)
        except Exception:
            continue
        if replacement is None:
            continue
        replacement = replacement.strip()
        if not replacement or replacement.lower() == token.lower():
            continue
        repaired_text = re.sub(
            rf"\b{re.escape(token)}\b",
            replacement,
            repaired_text,
            flags=re.IGNORECASE,
        )
    return repaired_text


def split_for_translation(text: str, max_chars: int = 420) -> List[str]:
    """Split long text into smaller chunks to avoid provider payload failures."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: List[str] = []
    cursor = 0
    while cursor < len(normalized):
        window = normalized[cursor : cursor + max_chars]
        if len(window) < max_chars:
            chunks.append(window.strip())
            break

        split_at = max(
            window.rfind(". "),
            window.rfind("? "),
            window.rfind("! "),
            window.rfind(", "),
            window.rfind("; "),
        )
        if split_at < 60:
            split_at = window.rfind(" ")
        if split_at < 30:
            split_at = len(window)

        piece = normalized[cursor : cursor + split_at].strip()
        if piece:
            chunks.append(piece)
        cursor += split_at
        while cursor < len(normalized) and normalized[cursor] == " ":
            cursor += 1

    return chunks or [normalized]


def safe_translate(
    text: str,
    translator: Any,
    fallback_translator: Any,
) -> str:
    """Translate text with retries and chunk fallback to avoid hard pipeline failures."""
    import time

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""

    errors: list[str] = []
    for attempt in range(3):
        try:
            result = translator.translate(normalized)
            if result is not None and result.strip():
                return result.strip()
        except Exception as exc:
            errors.append(str(exc))
        time.sleep(0.12 * (attempt + 1))

    for attempt in range(2):
        try:
            result = fallback_translator.translate(normalized)
            if result is not None and result.strip():
                return result.strip()
        except Exception as exc:
            errors.append(str(exc))
        time.sleep(0.12 * (attempt + 1))

    chunks = split_for_translation(normalized)
    if len(chunks) > 1:
        translated_chunks: List[str] = []
        for chunk in chunks:
            chunk_result = ""
            for candidate in (translator, fallback_translator):
                try:
                    translated = candidate.translate(chunk)
                    if translated is not None and translated.strip():
                        chunk_result = translated.strip()
                        break
                except Exception as exc:
                    errors.append(str(exc))
            if not chunk_result:
                chunk_result = chunk
            translated_chunks.append(chunk_result)
        combined = " ".join(part for part in translated_chunks if part).strip()
        if combined:
            return combined

    return normalized


def cached_translation_looks_poor(segments: list[Segment], target_lang: str) -> bool:
    if target_lang == "en":
        return False

    items = list(segments)
    if not items:
        return False

    unchanged = 0
    empty = 0
    for seg in items:
        source_clean = re.sub(r"\s+", " ", safe_text(seg.source_text)).strip().lower()
        translated_clean = re.sub(r"\s+", " ", safe_text(seg.translated_text)).strip().lower()
        if not translated_clean:
            empty += 1
            continue
        if source_clean and source_clean == translated_clean:
            unchanged += 1

    total = len(items)
    unchanged_ratio = unchanged / total
    empty_ratio = empty / total
    return unchanged_ratio > 0.35 or empty_ratio > 0.12


def translation_looks_wrong_language(segments: Iterable[Segment], target_lang: str) -> bool:
    if target_lang == "en":
        return False

    stopword_hints: dict[str, set[str]] = {
        "es": {"el", "la", "los", "las", "que", "de", "por", "para", "con", "una", "un", "como"},
        "fr": {"le", "la", "les", "des", "une", "que", "pour", "avec", "pas", "est", "dans"},
        "de": {"der", "die", "das", "und", "nicht", "mit", "ist", "für", "ein", "eine", "ich"},
        "pt": {"de", "do", "da", "que", "para", "com", "não", "uma", "um", "como", "você"},
        "sk": {"som", "si", "je", "sme", "ste", "sa", "že", "ako", "čo", "pre", "to", "nie"},
        "ru": {"и", "в", "не", "на", "что", "это", "как", "для", "с", "я", "ты"},
        "hi": {"है", "और", "नहीं", "के", "यह", "से", "मैं", "आप", "हम", "क्या"},
        "ja": {"です", "ます", "して", "ない", "する", "これ", "それ", "から", "まで", "よう"},
    }
    script_patterns: dict[str, str] = {
        "ru": r"[А-Яа-яЁё]",
        "hi": r"[\u0900-\u097F]",
        "ja": r"[\u3040-\u30FF\u4E00-\u9FFF]",
    }
    english_hints = {
        "the", "and", "you", "that", "this", "with", "for", "not", "are", "was", "have", "will", "what",
        "your", "from", "they", "can", "about", "just", "like", "there",
    }

    target_hints = stopword_hints.get(target_lang, set())
    script_pattern = script_patterns.get(target_lang)

    items = [seg for seg in segments if safe_text(seg.translated_text).strip()]
    if not items:
        return True

    checked = 0
    unchanged = 0
    english_like = 0
    target_like = 0

    for seg in items:
        source_clean = re.sub(r"\s+", " ", safe_text(seg.source_text)).strip().lower()
        text = safe_text(seg.translated_text).strip().lower()
        if len(text) < 8:
            continue
        checked += 1
        if source_clean and source_clean == text:
            unchanged += 1

        target_match = False
        if script_pattern is not None and re.search(script_pattern, text):
            target_match = True

        latin_tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ']+", text)
        en_hits = sum(1 for token in latin_tokens if token in english_hints)
        target_hits = sum(1 for token in latin_tokens if token in target_hints)

        if target_hits >= 1:
            target_match = True

        if target_match:
            target_like += 1
        if en_hits >= 2:
            english_like += 1

    if checked <= 0:
        return True

    unchanged_ratio = unchanged / checked
    english_ratio = english_like / checked
    target_ratio = target_like / checked

    if unchanged_ratio >= 0.55:
        return True

    if script_pattern is not None:
        return target_ratio < 0.30

    if target_hints:
        if target_ratio >= 0.28:
            return False
        return english_ratio >= 0.50 or unchanged_ratio >= 0.35

    return unchanged_ratio >= 0.45


def translate_segments_with_progress(
    segments: list[Segment],
    target_lang: str,
    segment_progress_callback: Callable[[int, int], None] | None = None,
    glossary_overrides: dict[str, str] | None = None,
    translation_provider: str = "google",
    force_english_source: bool = False,
) -> None:
    normalized_provider = translation_provider.strip().lower() if translation_provider else "google"
    translator = build_translator(normalized_provider, source="auto", target=target_lang)
    fallback_provider = "mymemory" if normalized_provider == "google" else "google"
    fallback_translator = build_translator(fallback_provider, source="auto", target=target_lang)
    explicit_en_translator = build_translator(normalized_provider, source="en", target=target_lang)
    explicit_en_fallback_translator = build_translator(fallback_provider, source="en", target=target_lang)
    word_translator = build_translator("google", source="en", target=target_lang)
    translation_cache: dict[str, str] = {}

    def recommended_translation_workers(item_count: int, provider: str) -> int:
        if item_count < 20:
            return 1
        if provider == "google":
            return 2
        if provider == "mymemory":
            return 3
        return 1

    def should_retry_with_english_source(source_text: str, translated_text: str) -> bool:
        if target_lang == "en":
            return False
        source_clean = re.sub(r"\s+", " ", source_text).strip().lower()
        translated_clean = re.sub(r"\s+", " ", translated_text).strip().lower()
        if not source_clean or not translated_clean:
            return False
        if source_clean != translated_clean:
            return False
        alpha_chars = re.findall(r"[A-Za-z]", source_clean)
        word_count = len(source_clean.split())
        return len(alpha_chars) >= 4 and word_count >= 2

    def looks_untranslated(source_text: str, translated_text: str) -> bool:
        if target_lang == "en":
            return False
        translated_clean = re.sub(r"\s+", " ", translated_text).strip()
        if not translated_clean:
            return True
        if should_retry_with_english_source(source_text, translated_text):
            return True
        if has_untranslated_english_tokens(source_text, translated_text, target_lang):
            return True
        return False

    def translate_source_text(source_text: str) -> str:
        if force_english_source:
            translated = safe_translate(source_text, explicit_en_translator, explicit_en_fallback_translator)
        else:
            translated = safe_translate(source_text, translator, fallback_translator)

        if not force_english_source and looks_untranslated(source_text, translated):
            retry = safe_translate(source_text, explicit_en_translator, explicit_en_fallback_translator)
            if retry is not None and retry.strip():
                translated = retry

        if not force_english_source and looks_untranslated(source_text, translated):
            retry = safe_translate(source_text, fallback_translator, translator)
            if retry is not None and retry.strip():
                translated = retry

        if looks_untranslated(source_text, translated):
            retry = safe_translate(source_text, explicit_en_fallback_translator, explicit_en_translator)
            if retry is not None and retry.strip():
                translated = retry

        if has_untranslated_english_tokens(source_text, translated, target_lang):
            translated = replace_untranslated_tokens(source_text, translated, word_translator)

        return translated

    total = len(segments)
    unchanged_count = 0
    if total <= 0:
        return

    segment_keys: list[str] = []
    source_by_key: dict[str, str] = {}
    key_counts: dict[str, int] = {}
    for seg in segments:
        source_key = re.sub(r"\s+", " ", seg.source_text).strip().lower()
        segment_keys.append(source_key)
        if source_key not in source_by_key:
            source_by_key[source_key] = seg.source_text
            key_counts[source_key] = 0
        key_counts[source_key] += 1

    unique_items = list(source_by_key.items())
    worker_count = min(recommended_translation_workers(total, normalized_provider), len(unique_items))

    if worker_count > 1 and len(unique_items) > 1:
        completed_segments = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(translate_source_text, source_text): source_key
                for source_key, source_text in unique_items
            }
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="Translating"):
                source_key = future_map[future]
                source_text = source_by_key[source_key]
                try:
                    translation_cache[source_key] = future.result()
                except Exception:
                    translation_cache[source_key] = source_text
                completed_segments += key_counts.get(source_key, 1)
                if segment_progress_callback is not None:
                    segment_progress_callback(min(completed_segments, total), total)
    else:
        completed_segments = 0
        for idx, (source_key, source_text) in enumerate(tqdm(unique_items, desc="Translating"), start=1):
            try:
                translation_cache[source_key] = translate_source_text(source_text)
            except Exception:
                translation_cache[source_key] = source_text
            completed_segments += key_counts.get(source_key, 1)
            if segment_progress_callback is not None:
                segment_progress_callback(min(completed_segments, total), total)

    for seg, source_key in zip(segments, segment_keys):
        translated = translation_cache.get(source_key, seg.source_text) or ""

        if glossary_overrides:
            translated = apply_glossary_overrides(translated, glossary_overrides)

        if target_lang != "en":
            source_clean = re.sub(r"\s+", " ", seg.source_text).strip().lower()
            translated_clean = re.sub(r"\s+", " ", translated).strip().lower()
            if source_clean and source_clean == translated_clean:
                unchanged_count += 1

        seg.translated_text = translated

    if target_lang != "en" and total > 0 and unchanged_count / total > 0.45:
        raise RuntimeError(
            "Translation provider returned too many unchanged lines. "
            "Try again in a minute or use a different target language code."
        )