"""Shared matplotlib style for the replication figures.

Colours follow a validated categorical palette (fixed slot order, colour follows the entity):
positions keep one hue in every figure (PANL blue, CC orange, PANL+1 aqua, ...).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
BLUE_LIGHT, RED_LIGHT = "#86b6ef", "#f0a3a2"

# colour follows the entity: token positions
POS_COLOR = {
    "PANL": BLUE,
    "CC": ORANGE,
    "PANL+1": AQUA,
    "FCC": YELLOW,
    "AC": MAGENTA,
    "A_first": GREEN,
    "A_last": VIOLET,
    "QTT": MUTED,
}
POS_LABEL = {
    "PANL": "PANL (post-answer newline)",
    "CC": "CC (confidence colon)",
    "PANL+1": "PANL+1 (control)",
    "FCC": "FCC (first confidence colon, control)",
    "AC": "AC (answer colon)",
    "A_first": "first answer token",
    "A_last": "last answer token",
    "QTT": "QTT (3rd question token, control)",
}


def setup():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 2.5,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def layer_axis(ax, layers, every=1):
    """Layers are sampled non-uniformly (0, 10, 15, 20..35, 40, 50, 61): plot on an ordinal axis like the paper."""
    x = np.arange(len(layers))
    ax.set_xticks(x[::every])
    ax.set_xticklabels([str(l) for l in layers][::every], rotation=45)
    ax.set_xlim(-0.6, len(layers) - 0.4)
    ax.set_xlabel("Layer")
    return x


def line(ax, x, y, err=None, color=BLUE, label=None, ls="-", marker="o", ms=4.5, lw=2.0, alpha=1.0, zorder=3, hollow=False):
    """2px line, markers with a surface-coloured ring, optional SEM error bars."""
    y = np.asarray(y, dtype=float)
    if err is not None:
        ax.errorbar(x, y, yerr=np.asarray(err, dtype=float), fmt="none", ecolor=color, elinewidth=1.0, capsize=1.5, alpha=0.75 * alpha, zorder=zorder - 1)
    ax.plot(
        x, y, ls=ls, color=color, lw=lw, marker=marker, ms=ms, alpha=alpha, label=label, zorder=zorder,
        markerfacecolor=SURFACE if hollow else color, markeredgecolor=color if hollow else SURFACE, markeredgewidth=1.2 if hollow else 1.0,
    )


def hline(ax, y, color=MUTED, label=None, lw=1.2, ls="-"):
    ax.axhline(y, color=color, lw=lw, ls=ls, label=label, zorder=1)


def sem(a, axis=0):
    a = np.asarray(a, dtype=float)
    return a.std(axis=axis, ddof=1) / np.sqrt(a.shape[axis])


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)
