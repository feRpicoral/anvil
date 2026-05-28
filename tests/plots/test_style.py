from __future__ import annotations

import re
from collections.abc import Iterator

import matplotlib as mpl
import pytest
from matplotlib.colors import to_rgba

from anvil.plots.style import apply_style, palette

_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


@pytest.fixture(autouse=True)
def _restore_rcparams() -> Iterator[None]:
    with mpl.rc_context():
        yield


def test_palette_has_three_way_variant_keys() -> None:
    p = palette()

    assert {"base", "finetuned", "gpt_4o"}.issubset(p.keys())


def test_palette_values_are_hex_strings() -> None:
    for name, value in palette().items():
        assert _HEX_COLOR_RE.fullmatch(value) is not None, f"{name}: invalid hex color {value!r}"


def test_palette_assigns_distinct_colors_to_variants() -> None:
    p = palette()

    assert p["base"] != p["finetuned"]
    assert p["finetuned"] != p["gpt_4o"]
    assert p["base"] != p["gpt_4o"]


def test_apply_style_strips_top_and_right_spines() -> None:
    apply_style()

    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False


def test_apply_style_sets_savefig_dpi() -> None:
    apply_style()

    assert mpl.rcParams["savefig.dpi"] == 160


def test_apply_style_sets_white_facecolor() -> None:
    apply_style()

    assert to_rgba(mpl.rcParams["figure.facecolor"]) == to_rgba("white")
    assert to_rgba(mpl.rcParams["axes.facecolor"]) == to_rgba("white")


def test_apply_style_enables_grid_with_low_alpha() -> None:
    apply_style()

    assert mpl.rcParams["axes.grid"] is True
    assert 0.0 < mpl.rcParams["grid.alpha"] < 1.0


def test_apply_style_is_idempotent() -> None:
    apply_style()
    first = dict(mpl.rcParams)
    apply_style()
    second = dict(mpl.rcParams)

    assert first == second
