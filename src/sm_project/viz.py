"""Shared plotting configuration and save helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .config import FIGURES_DIR

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


def save_figure(fig, filename: str, directory: str | Path | None = None) -> Path:
    """Save a figure to outputs/figures (or a supplied directory)."""
    target_dir = Path(directory) if directory is not None else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return path


def finish_figure(fig, filename: str, directory: str | Path | None = None) -> Path:
    """Save, render inline, and close a figure."""
    path = save_figure(fig, filename, directory)
    plt.show()
    plt.close(fig)
    return path
