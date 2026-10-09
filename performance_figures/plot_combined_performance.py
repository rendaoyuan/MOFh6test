"""Combine scaling, publisher, paragraph, and structure evidence in one figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import ScaledTranslation

from nature_style import PALETTE, apply_nature_style, project_paths, save_publication_figure
from performance_data import (
    GROUPS,
    SHOT_COLUMNS,
    load_paragraph,
    load_shot,
    load_structure,
    summarize_coreference,
)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
EXPORT_SUFFIXES = (".pdf",)

PUBLISHER_ORDER = ["ACS", "Elsevier", "RSC", "Springer", "Wiley", "Overall"]
PARAGRAPH_ORDER = ["Overall", "ACS", "Elsevier", "RSC", "Springer", "Wiley"]
STRUCTURE_LABELS = {
    "Metal Source": "Metal source",
    "Organic Linkers Source": "Organic linker source",
    "Modulator Source": "Modulator source",
    "Solvent Source": "Solvent source",
    "Quantity of Metal": "Metal quantity",
    "Quantity of Organic Linkers": "Organic linker quantity",
    "Quantity of Modulator": "Modulator quantity",
    "Quantity of Solvent": "Solvent quantity",
    "pH": "pH",
    "Synthesis Temperature": "Synthesis temperature",
    "Synthesis Time": "Synthesis time",
    "Equipment": "Equipment",
    "Crystal Morphology": "Crystal morphology",
    "Yield": "Yield",
}
STRUCTURE_GROUP_LABELS = {
    "Chemical inputs": "Chemicals",
    "Synthesis conditions": "Conditions",
    "Crystallization outcomes": "Crystallization",
}


def add_panel_label(ax: plt.Axes, label: str, *, x_offset: float = -18, y_offset: float = 9) -> None:
    transform = ax.transAxes + ScaledTranslation(x_offset / 72, y_offset / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0,
        1,
        f"({label})",
        transform=transform,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )


def draw_shot(ax: plt.Axes, frame: pd.DataFrame) -> None:
    colors = [PALETTE["green"], PALETTE["blue_light"], PALETTE["darkest"]]
    x = np.arange(len(frame))
    ax.bar(
        x,
        frame["Cosine"],
        width=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks(x, labels=[str(int(value)) for value in frame["shot"]])
    ax.set_xlabel("Shots number")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Fine-tuning performance", fontsize=7, fontweight="bold", pad=4)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.55, zorder=0)
    ax.tick_params(axis="x", length=0)


def order_coreference(summary: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(PUBLISHER_ORDER)}
    return (
        summary.assign(_order=summary["pub"].replace({"Elsvier": "Elsevier"}).map(order).fillna(len(order)))
        .assign(pub=lambda frame: frame["pub"].replace({"Elsvier": "Elsevier"}))
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def draw_coreference(ax: plt.Axes, summary: pd.DataFrame) -> None:
    regular = ~summary["pub"].eq("Overall")
    overall = ~regular
    ax.scatter(
        summary.loc[regular, "total"],
        summary.loc[regular, "target_rate"],
        s=34,
        facecolors=to_rgba(PALETTE["darkest"], 0.70),
        edgecolors="black",
        linewidths=0.65,
        zorder=3,
    )
    ax.scatter(
        summary.loc[overall, "total"],
        summary.loc[overall, "target_rate"],
        s=48,
        marker="D",
        facecolors=to_rgba(PALETTE["cyan"], 0.70),
        edgecolor="black",
        linewidth=0.65,
        zorder=4,
    )
    offsets = {
        "ACS": (4, 5),
        "Elsevier": (4, -12),
        "RSC": (4, 4),
        "Springer": (4, -8),
        "Wiley": (4, -8),
        "Overall": (-4, 5),
    }
    for _, row in summary.iterrows():
        dx, dy = offsets.get(str(row["pub"]), (4, 4))
        ax.annotate(
            str(row["pub"]),
            (row["total"], row["target_rate"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
            va="bottom" if dy >= 0 else "top",
            fontsize=5.8,
        )
    overall_rate = float(summary.loc[overall, "target_rate"].iloc[0])
    ax.axhline(overall_rate, color=PALETTE["cyan"], linewidth=0.85, linestyle="--", zorder=1)
    ax.set_xlim(0, summary["total"].max() * 1.08)
    ax.set_ylim(0.78, 0.98)
    ax.set_xlabel("Number of identified coreference instances")
    ax.set_ylabel("Coreference resolution rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Coreference resolution performance", fontsize=7, fontweight="bold", pad=4)


def order_paragraph(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["pub"] = frame["pub"].replace({"Elsvier": "Elsevier", "Total": "Overall"})
    order = {name: index for index, name in enumerate(PARAGRAPH_ORDER)}
    return frame.assign(_order=frame["pub"].map(order).fillna(len(order))).sort_values("_order").drop(columns="_order")


def draw_paragraph(ax: plt.Axes, frame: pd.DataFrame) -> None:
    colors = [PALETTE["green"], PALETTE["cyan"], PALETTE["blue"], PALETTE["darkest"]]
    within_step = 0.74
    group_step = 4.15
    bar_width = 0.62
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for group_index, (_, row) in enumerate(frame.iterrows()):
        center = group_index * group_step
        positions = center + (np.arange(len(SHOT_COLUMNS)) - 1.5) * within_step
        ax.bar(
            positions,
            row[SHOT_COLUMNS].to_numpy(dtype=float),
            width=bar_width,
            color=colors,
            edgecolor="none",
            zorder=3,
        )
        tick_positions.extend(positions.tolist())
        tick_labels.extend([column.replace("-shot", "") for column in SHOT_COLUMNS])
        ax.text(
            center,
            1.03,
            str(row["pub"]),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold" if str(row["pub"]) == "Overall" else "normal",
            clip_on=False,
        )
        if group_index < len(frame) - 1:
            ax.axvline(center + group_step / 2, color="#D9D9D9", linewidth=0.65, zorder=1)
    ax.set_xticks(tick_positions, labels=tick_labels)
    ax.set_xlim(-2.05, (len(frame) - 1) * group_step + 2.05)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Cosine similarity")
    ax.set_xlabel("Shots number")
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.text(
        0,
        1.28,
        "Publisher-level extraction performance",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        clip_on=False,
    )
    handles = [
        Patch(facecolor=color, edgecolor="none", label=column.replace("-shot", " shots"))
        for color, column in zip(colors, SHOT_COLUMNS)
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.16),
        ncols=4,
        columnspacing=1.0,
        handlelength=1.1,
        handletextpad=0.35,
    )


def draw_structure(ax: plt.Axes, frame: pd.DataFrame) -> None:
    if not np.allclose(frame.loc["Accuracy"], frame.loc["Precision"]):
        raise ValueError("Accuracy no longer duplicates Precision; update the combined-figure contract")
    table = frame.loc[["Precision", "Recall", "F1"]].T
    y = np.arange(len(table))
    ax.hlines(y, table["Precision"], table["Recall"], color="#909090", linewidth=0.9, zorder=2)
    ax.scatter(table["Precision"], y, s=27, color=PALETTE["green"], zorder=3)
    ax.scatter(table["Recall"], y, s=27, color=PALETTE["cyan"], zorder=3)
    ax.scatter(
        table["F1"],
        y,
        s=25,
        marker="D",
        color=PALETTE["darkest"],
        edgecolor="white",
        linewidth=0.3,
        zorder=4,
    )
    labels = [STRUCTURE_LABELS[field] for field in table.index]
    ax.set_yticks(y, labels=labels)
    ax.set_ylim(len(table) - 0.5, -0.5)
    ax.set_xlim(0.80, 1.005)
    ax.set_xlabel("Performance")
    ax.xaxis.grid(True, color="#E3E3E3", linewidth=0.55, zorder=0)
    ax.tick_params(axis="y", length=0, labelsize=6)
    ax.text(
        0,
        1.09,
        "Structured conversion performance",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        clip_on=False,
    )

    boundaries = np.cumsum([len(fields) for fields in GROUPS.values()])[:-1]
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(table)]
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="#CFCFCF", linewidth=0.65, zorder=1)
    for index, (name, start, end) in enumerate(zip(GROUPS, starts, ends)):
        if index % 2 == 0:
            ax.axhspan(start - 0.5, end - 0.5, color=PALETTE["lightest"], zorder=0)
        ax.text(
            1.012,
            (start + end - 1) / 2,
            STRUCTURE_GROUP_LABELS[name],
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6,
            fontweight="bold",
            clip_on=False,
        )
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.5, color=PALETTE["green"], label="Precision"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.5, color=PALETTE["cyan"], label="Recall"),
        Line2D([0], [0], marker="D", linestyle="none", markersize=4.2, color=PALETTE["darkest"], label="F1"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.015), ncols=3)


def make_figure(
    shot: pd.DataFrame,
    coreference: pd.DataFrame,
    paragraph: pd.DataFrame,
    structure: pd.DataFrame,
) -> plt.Figure:
    apply_nature_style(6.5)
    fig = plt.figure(figsize=(7.2, 7.6))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.82, 0.90, 1.62],
        width_ratios=[0.88, 1.52],
        hspace=0.58,
        wspace=0.34,
    )
    axes = {
        "a": fig.add_subplot(grid[0, 0]),
        "b": fig.add_subplot(grid[0, 1]),
        "c": fig.add_subplot(grid[1, :]),
        "d": fig.add_subplot(grid[2, :]),
    }
    fig.subplots_adjust(left=0.13, right=0.84, bottom=0.07, top=0.965)

    draw_shot(axes["a"], shot)
    draw_coreference(axes["b"], coreference)
    draw_paragraph(axes["c"], paragraph)
    draw_structure(axes["d"], structure)
    for label, ax in axes.items():
        add_panel_label(ax, label)
    return fig


def main() -> None:
    data_dir, output_dir = project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=Path, default=data_dir / "shot.xlsx")
    parser.add_argument("--coreference", type=Path, default=data_dir / "cr.xlsx")
    parser.add_argument("--paragraph", type=Path, default=data_dir / "parag.xlsx")
    parser.add_argument("--structure", type=Path, default=data_dir / "structure.xlsx")
    parser.add_argument("--output", type=Path, default=output_dir / "combined_performance")
    args = parser.parse_args()

    figure = make_figure(
        load_shot(args.shot),
        order_coreference(summarize_coreference(args.coreference)),
        order_paragraph(load_paragraph(args.paragraph)),
        load_structure(args.structure),
    )
    figure.canvas.draw()
    a, b, c, d = [ax.get_position() for ax in figure.axes]
    if not (np.allclose([a.y0, a.y1], [b.y0, b.y1]) and
            np.allclose([c.x0, c.x1], [d.x0, d.x1])):
        raise ValueError("Combined figure panels are not aligned")
    save_publication_figure(figure, args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
