"""Export panel c: paragraph cosine similarity by publisher and shot count."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from nature_style import apply_nature_style, project_paths, save_publication_figure
from performance_data import load_paragraph
from plot_combined_performance import draw_paragraph, order_paragraph


def main() -> None:
    data_dir, output_dir = project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=data_dir / "parag.xlsx")
    parser.add_argument("--output", type=Path, default=output_dir / "individual_panels" / "panel_c_paragraph")
    args = parser.parse_args()

    apply_nature_style(7)
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.73)
    draw_paragraph(axis, order_paragraph(load_paragraph(args.input)))
    save_publication_figure(figure, args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
