"""Shared, presentation-ready plotting conventions for explainability notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler


MODEL_COLORS = {
    "M7 Late Fusion": "#6B7280",
    "M11 Scalar": "#3D8DFF",
    "M11 Vector": "#6DCBF4",
    "M33 Hierarchical": "#8B5CF6",
    "Original GMU": "#F59E0B",
    "GMU-GNN": "#10B981",
}

STATIC_COLOR = "#F59E0B"
DYNAMIC_COLOR = "#10B981"
OBSERVED_COLOR = "#111827"
PREDICTION_COLOR = "#3D8DFF"
RESIDUAL_COLOR = "#8B5CF6"
REFERENCE_COLOR = "#374151"

FIGSIZE_SINGLE = (8.0, 5.0)
FIGSIZE_DOUBLE = (13.6, 5.0)
FIGSIZE_TRIPLE = (16.0, 5.0)
TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 12
LEGEND_SIZE = 12
LINE_WIDTH = 2.2
MARKER_SIZE = 6
DPI = 300


def apply_presentation_style() -> None:
    """Apply a restrained 16:9-slide-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": TICK_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": LABEL_SIZE,
            "axes.labelcolor": "#111827",
            "axes.edgecolor": "#9CA3AF",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(
                color=[PREDICTION_COLOR, STATIC_COLOR, DYNAMIC_COLOR, RESIDUAL_COLOR]
            ),
            "grid.color": "#D1D5DB",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.8,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "legend.frameon": False,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
        }
    )


def save_presentation_figure(fig, path) -> None:
    """Save a figure with consistent DPI, whitespace, and a white canvas."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=DPI, bbox_inches="tight", facecolor="white")

