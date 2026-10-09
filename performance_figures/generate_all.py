"""Generate the final combined figure and its four standalone panels."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "plot_combined_performance.py",
    "plot_panel_a_shot.py",
    "plot_panel_b_coreference_scatter.py",
    "plot_panel_c_paragraph.py",
    "plot_panel_d_structure_dumbbell.py",
]


def main() -> None:
    script_dir = Path(__file__).parent
    for name in SCRIPTS:
        subprocess.run([sys.executable, str(script_dir / name)], check=True)


if __name__ == "__main__":
    main()
