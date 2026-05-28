# Adapted from forge@2b9d733fb6b0df49a2c55fca7d879a4240843b72
# (forge/plots/style.py). When Forge updates the stylesheet, refresh this
# file from the upstream revision and bump the SHA in this header.
"""Matplotlib stylesheet for Anvil charts.

The goal is engineering-blog aesthetics: clean axes, readable typography,
no chartjunk. Calling ``apply_style()`` once at the start of a script
configures matplotlib's rcParams; subsequent plots inherit them.

Colors are a color-blind-friendly palette adapted from the Okabe-Ito set,
with deliberate semantic mapping: blue = full precision / base, orange =
quantized / fine-tuned, purple = GPT-4o, green = Claude, gray = baselines.
"""

from __future__ import annotations

import matplotlib as mpl


def palette() -> dict[str, str]:
    """Named colors used throughout the charts.

    Returns a dict for explicit per-series color binding; avoid relying on
    cycler order so the same series always gets the same color regardless
    of plot order.
    """
    return {
        "base": "#0072B2",
        "finetuned": "#D55E00",
        "gpt_4o": "#7570b3",
        "claude": "#1b9e77",
        "api_other": "#666666",
        "baseline": "#444444",
        "muted": "#888888",
        "train_loss": "#0072B2",
        "val_loss": "#D55E00",
        "self_hosted": "#0072B2",
        "api_blended": "#7570b3",
    }


def apply_style() -> None:
    """Install the Anvil matplotlib defaults onto the global rcParams."""
    mpl.rcParams.update(
        {
            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            # Axes & spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            # Lines & markers
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            # Figure sizing & DPI
            "figure.figsize": (8.0, 5.0),
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            # Legend
            "legend.frameon": False,
            "legend.loc": "best",
        }
    )
