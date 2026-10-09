"""Shared publication styling and export helpers for performance figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "lightest": "#F7FCF0",
    "light": "#E0F3DB",
    "green_light": "#CCEBC5",
    "green": "#A8DDB5",
    "cyan": "#7BCCC4",
    "blue_light": "#4EB3D3",
    "blue": "#2B8CBE",
    "blue_dark": "#0868AC",
    "darkest": "#084081",
}

def apply_nature_style(font_size: float = 7.0) -> None:
    """Apply a compact, editable, journal-width Matplotlib style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": font_size,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def project_paths() -> tuple[Path, Path]:
    """Return script-relative source-data and output directories."""
    vis_dir = Path(__file__).parent.parent
    data_dir = vis_dir / "dataset"
    output_dir = vis_dir / "outputs" / "performance"
    return data_dir, output_dir


def ensure_columns(frame, required: list[str], source: Path) -> None:
    """Fail early with a readable message when an input schema changes."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {missing}")


def save_publication_figure(
    fig: plt.Figure,
    output_prefix: Path,
) -> list[Path]:
    """Export one standalone PDF, as requested for this figure set."""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    return [pdf_path]
