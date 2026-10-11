"""Smoke tests for migrated legacy experiment entrypoints.

No model weights or datasets are required.
"""
import importlib
import subprocess
import sys


def test_migrated_modules_import():
    for name in (
        "mami.models.llava",
        "mami.models.smolvlm",
        "mami.binary.score_thresholds",
    ):
        assert importlib.import_module(name)


def test_migrated_entrypoints_have_help():
    for name in (
        "mami.models.llava",
        "mami.models.smolvlm",
        "mami.binary.score_thresholds",
    ):
        result = subprocess.run(
            [sys.executable, "-m", name, "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (name, result.stderr)
        assert "usage:" in result.stdout.lower()
