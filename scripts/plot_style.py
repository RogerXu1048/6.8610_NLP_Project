"""Centralized matplotlib styling for the milestone notebook + poster export.

Usage in a notebook:

    from scripts.plot_style import setup_style, MODEL_COLORS, MODEL_ORDER, save_for_poster
    setup_style()                              # call once at the top of the notebook
    fig, ax = plt.subplots(figsize=(8, 5))
    ...
    save_for_poster(fig, "tax_overview")       # writes both .png (150 dpi) and .pdf (vector)

Design principles:
- One stable color per model across every plot (brand-suggestive but distinct)
- Generous default font sizes (poster legible from ~1m at 4x scale)
- Minimal chrome (no top/right spines), explicit grid on y only
- Both raster (.png at 150 dpi for notebook, 300 for poster) and vector (.pdf) outputs
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Stable model order (used by every plot) ───────────────────────────────────
MODEL_ORDER = (
    "gpt-5.5",
    "claude-sonnet",
    "claude-opus",
    "gemini-3.1-pro",
    "deepseek-v4-pro",
)

# ── Stable per-model colors ───────────────────────────────────────────────────
# Picked to be (a) brand-suggestive, (b) distinguishable on B/W print fallback,
# (c) friendly to common color-vision deficiencies (no red-green confusion pair).
MODEL_COLORS = {
    "gpt-5.5":         "#10A37F",   # OpenAI teal
    "claude-sonnet":   "#D97757",   # Anthropic warm orange
    "claude-opus":     "#7B3F00",   # darker brown (distinguish from sonnet)
    "gemini-3.1-pro":  "#4285F4",   # Google blue
    "deepseek-v4-pro": "#553C9A",   # purple
}

# ── Stable colors for ambiguity types ─────────────────────────────────────────
# Used in the type × model heatmap, type-grouped bar charts, etc.
TYPE_COLORS = {
    "coreferential":           "#2E86AB",
    "syntactic":               "#A23B72",
    "scopal":                  "#F18F01",
    "collective_distributive": "#3E885B",
    "elliptical":              "#7B4F8F",
}

# ── Behavior labels (SA / EA / AC) ────────────────────────────────────────────
BEHAVIOR_COLORS = {
    "SA": "#888888",        # neutral grey — silent (default)
    "EA": "#F39C12",        # amber — explicit (caution)
    "AC": "#27AE60",        # green — clarification (good)
    "unclassifiable": "#BDC3C7",
    "error":          "#E74C3C",
}

# ── Diverging palette for tax (negative=blue=help, positive=red=hurt) ────────
DIVERGING_CMAP = "RdBu_r"   # for heatmaps where center=0 matters

# ── Typography sizes ──────────────────────────────────────────────────────────
# Notebook view at default; poster export multiplies by ~1.5
BASE_SIZES = {
    "title":   14,
    "label":   12,
    "tick":    11,
    "legend":  10,
    "annot":    9,
}


def setup_style(scale: float = 1.0) -> None:
    """Apply project-wide matplotlib defaults. Call once at notebook top.

    `scale` multiplies all font sizes — 1.0 for notebook view, ~1.4 for poster.
    """
    mpl.rcParams.update({
        # Font
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          BASE_SIZES["label"] * scale,
        "axes.titlesize":     BASE_SIZES["title"] * scale,
        "axes.labelsize":     BASE_SIZES["label"] * scale,
        "xtick.labelsize":    BASE_SIZES["tick"]  * scale,
        "ytick.labelsize":    BASE_SIZES["tick"]  * scale,
        "legend.fontsize":    BASE_SIZES["legend"] * scale,
        "figure.titlesize":   BASE_SIZES["title"] * scale + 1,

        # Layout
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.linestyle":     ":",
        "grid.alpha":         0.4,
        "grid.linewidth":     0.8,
        "axes.axisbelow":     True,

        # Misc
        "figure.dpi":         100,           # notebook display
        "savefig.dpi":        300,           # default for save_for_poster .png
        "savefig.bbox":       "tight",
        "pdf.fonttype":       42,            # embed TrueType (editable in Illustrator)
        "ps.fonttype":        42,
        "axes.titlepad":      10,
        "figure.autolayout":  False,         # we use bbox_inches="tight" on save
    })


def model_palette(models: Iterable[str] = None) -> list[str]:
    """Return colors for the given model list (defaults to MODEL_ORDER)."""
    models = list(models) if models is not None else list(MODEL_ORDER)
    return [MODEL_COLORS.get(m, "#777777") for m in models]


def model_order(models: Iterable[str]) -> list[str]:
    """Reorder a list of models to match the canonical project order."""
    seen = set(models)
    canonical = [m for m in MODEL_ORDER if m in seen]
    extras = [m for m in models if m not in MODEL_ORDER]
    return canonical + extras


def annotate_bars(
    ax: plt.Axes,
    fmt: str = "{:+.1f}",
    offset_pp: float = 1.0,
    fontsize: Optional[int] = None,
) -> None:
    """Write the value above each bar in the axes."""
    fontsize = fontsize or BASE_SIZES["annot"]
    for p in ax.patches:
        height = p.get_height()
        if height == 0:
            continue
        x = p.get_x() + p.get_width() / 2
        y = height + offset_pp if height >= 0 else height - offset_pp * 1.5
        ax.annotate(
            fmt.format(height),
            xy=(x, height),
            xytext=(0, 6 if height >= 0 else -12),
            textcoords="offset points",
            ha="center", va="bottom" if height >= 0 else "top",
            fontsize=fontsize,
            color="#333333",
        )


def add_zero_line(ax: plt.Axes, color: str = "#444444", lw: float = 0.8) -> None:
    """Add a horizontal y=0 reference line. Useful for tax / delta plots."""
    ax.axhline(0, color=color, linewidth=lw, zorder=1)


def shorten_model_name(m: str) -> str:
    """Shorter display label for tight axes."""
    return {
        "gpt-5.5":         "GPT-5.5",
        "claude-sonnet":   "Sonnet 4.6",
        "claude-opus":     "Opus 4.6",
        "gemini-3.1-pro":  "Gemini 3.1 Pro",
        "deepseek-v4-pro": "DeepSeek V4 Pro",
    }.get(m, m)


def _project_root() -> Path:
    """Walk up from this file to the project root (looks for `data/` and `scripts/`)."""
    here = Path(__file__).resolve().parent
    for p in [here] + list(here.parents):
        if (p / "scripts").is_dir() and (p / "data").is_dir():
            return p
    return here.parent


def save_for_poster(
    fig: plt.Figure,
    name: str,
    out_dir: Path | str | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> dict[str, Path]:
    """Save the figure in both raster (300 dpi) and vector formats.

    `out_dir` defaults to <project_root>/plots/findings — the curated location
    that's referenced by README and the poster. Anchored to the project root
    so it works whether the notebook lives in /notebooks or anywhere else.
    """
    if out_dir is None:
        out_dir = _project_root() / "plots" / "findings"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for ext in formats:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        paths[ext] = path
    return paths