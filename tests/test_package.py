"""Smoke test — the package imports and exposes a version."""

from __future__ import annotations

import anvil


def test_version_is_set() -> None:
    assert isinstance(anvil.__version__, str)
    assert anvil.__version__.count(".") == 2
