"""Export panel a: overall cosine similarity across shot counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from nature_style import apply_nature_style, project_paths, save_publication_figure
from performance_data import load_shot
from plot_combined_performance import draw_shot


def main() -> None:
    data_dir, output_dir = project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=data_dir / "shot.xlsx")
    parser.add_argument("--output", type=Path, default=output_dir / "individual_panels" / "panel_a_shot")
    args = parser.parse_args()

    apply_nature_style(7)
    figure, axis = plt.subplots(figsize=(3.5, 2.75))
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.89)
    draw_shot(axis, load_shot(args.input))
    save_publication_figure(figure, args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
