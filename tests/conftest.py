"""Shared pytest fixtures for autodub tests."""

from pathlib import Path

import pytest

from autodub.segments import Segment


@pytest.fixture
def sample_segments() -> list[Segment]:
    """Return a small list of sample segments for testing."""
    return [
        Segment(start_s=0.0, end_s=2.5, source_text="Hello world", translated_text="Ahoj svet"),
        Segment(start_s=2.5, end_s=5.0, source_text="This is a test", translated_text="Toto je test"),
        Segment(start_s=5.0, end_s=8.0, source_text="Good morning everyone", translated_text="Dobré ráno všetkým"),
    ]


@pytest.fixture
def tmp_audio_dir(tmp_path) -> Path:
    """Return a temporary directory for audio files."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path