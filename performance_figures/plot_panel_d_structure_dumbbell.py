"""Export panel d: field-level precision-recall dumbbells with F1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from nature_style import apply_nature_style, project_paths, save_publication_figure
from performance_data import load_structure
from plot_combined_performance import draw_structure


def main() -> None:
    data_dir, output_dir = project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=data_dir / "structure.xlsx")
    parser.add_argument(
        "--output",
        type=Path,
        default=output_dir / "individual_panels" / "panel_d_structure_dumbbell",
    )
    args = parser.parse_args()

    apply_nature_style(7)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    figure.subplots_adjust(left=0.25, right=0.75, bottom=0.12, top=0.90)
    draw_structure(axis, load_structure(args.input))
    save_publication_figure(figure, args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
