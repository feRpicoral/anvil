from __future__ import annotations

import matplotlib as mpl

from anvil.plots.style import apply_style, palette


def test_palette_has_three_way_variant_keys() -> None:
    p = palette()

    assert {"base", "finetuned", "gpt_4o"}.issubset(p.keys())


def test_palette_values_are_hex_strings() -> None:
    for name, value in palette().items():
        assert value.startswith("#"), f"{name}: {value!r} should start with '#'"
        assert len(value) == 7, f"{name}: expected 7-char hex, got {value!r}"


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

    # `to_rgba` normalizes any matplotlib color spec to a tuple; "white" → (1,1,1,1).
    from matplotlib.colors import to_rgba

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
