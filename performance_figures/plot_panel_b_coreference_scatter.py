"""Export panel b: publisher record count versus target-label proportion."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from nature_style import apply_nature_style, project_paths, save_publication_figure
from performance_data import summarize_coreference
from plot_combined_performance import draw_coreference, order_coreference


def main() -> None:
    data_dir, output_dir = project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=data_dir / "cr.xlsx")
    parser.add_argument(
        "--output",
        type=Path,
        default=output_dir / "individual_panels" / "panel_b_coreference_scatter",
    )
    args = parser.parse_args()

    apply_nature_style(7)
    figure, axis = plt.subplots(figsize=(5.7, 3.8))
    figure.subplots_adjust(left=0.15, right=0.97, bottom=0.17, top=0.91)
    draw_coreference(axis, order_coreference(summarize_coreference(args.input)))
    save_publication_figure(figure, args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
