"""Shared pytest fixtures.

Every test runs against an isolated settings object and a temporary database so
that no test can observe another test's environment mutations.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.core.config import reset_settings_cache


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Snapshot the real environment and restore it after each test.

    Without this, a test that sets ``MAX_PAGES_PER_COMPETITOR`` leaks into the
    next test and the suite order silently changes behaviour.
    """
    before = dict(os.environ)
    reset_settings_cache()
    yield
    os.environ.clear()
    os.environ.update(before)
    reset_settings_cache()


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    return data
