"""Tests for autodub.ffmpeg_utils module."""

import subprocess

import pytest

from autodub.ffmpeg_utils import run_cmd


class TestRunCmd:
    def test_run_cmd_success(self):
        """Test that run_cmd succeeds with a valid command."""
        # This should succeed on any system with ls
        run_cmd(["ls", "-l", "."])

    def test_run_cmd_missing_binary(self):
        """Test that run_cmd raises when binary is not found."""
        with pytest.raises(RuntimeError, match="Required executable not found"):
            run_cmd(["nonexistent_binary_12345"])

    def test_run_cmd_fails(self):
        """Test that run_cmd raises when command exits with error."""
        with pytest.raises(RuntimeError, match="Command failed"):
            run_cmd(["ls", "--nonexistent-argument"])