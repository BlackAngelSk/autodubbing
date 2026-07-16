"""Configuration constants and defaults for the auto-dubbing pipeline."""

from __future__ import annotations

DEFAULT_EDGE_VOICES: dict[str, str] = {
    "en": "en-US-AriaNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "ja": "ja-JP-NanamiNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "sk": "sk-SK-ViktoriaNeural",
}

LARGE_WHISPER_MODELS = {
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v2",
    "distil-large-v3",
}

TRANSLATION_PROVIDERS = {"google", "mymemory"}
ASR_ENGINE_CHOICES = {"auto", "whisper", "stable-ts"}
HF_UNAUTH_WARNING_TEXT = "unauthenticated requests to the hf hub"
COQUI_DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
COQUI_XTTS_SUPPORTED_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
}
COQUI_LANGUAGE_ALIASES = {
    "pt-br": "pt",
    "pt-pt": "pt",
    "zh": "zh-cn",
    "cz": "cs",
    "jp": "ja",
    "kr": "ko",
    "sk": "cs",
}
TTS_POSTPROCESS_VERSION = 5

# Supported languages for the UI and pipeline
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "ja": "Japanese",
    "pt": "Portuguese",
    "ru": "Russian",
    "sk": "Slovak",
}