"""Tests for DEFAULT_NOISE_PATTERNS — the ingest-time sample filter."""
from __future__ import annotations

import pytest

from crucible.insight.engine import DEFAULT_NOISE_PATTERNS, _is_noisy


class TestDefaultNoisePatterns:
    @pytest.mark.parametrize(
        "path",
        [
            ".history/foo.md",
            ".lh/bar.md",
            ".venv/lib/site-packages/x.py",
            "node_modules/pkg/index.js",
            "__pycache__/module.cpython-313.pyc",
            "dist/bundle.js",
            "build/lib/x.py",
            ".git/objects/pack.idx",
            "docs/diffs/2026-04-22.md",
            "data.json",
        ],
    )
    def test_dev_clutter_is_filtered(self, path):
        assert _is_noisy(path, DEFAULT_NOISE_PATTERNS), f"{path!r} should be noisy"

    @pytest.mark.parametrize(
        "path",
        [
            "src/module.py",
            "docs/real-content.md",
            "notes/2026-04-22.md",
            "README.md",
        ],
    )
    def test_real_content_survives(self, path):
        assert not _is_noisy(path, DEFAULT_NOISE_PATTERNS)
