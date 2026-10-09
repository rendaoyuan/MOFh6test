"""Compose the four notebook std panels using the manuscript figure style."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mofhv2-std-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from nature_style import apply_nature_style, project_paths
from std_panels import draw_shot, draw_coreference, draw_paragraph, draw_structure

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})

DRAW = dict(shot=draw_shot, coreference=draw_coreference,
            paragraph=draw_paragraph, structure=draw_structure)
LETTERS = dict(zip(DRAW, 'abcd'))


def validate_inputs(paths):
    """Report missing source statistics before producing a publication figure."""
    errors = []
    for name, path in paths.items():
        try:
            if name == 'shot':
                frame = pd.read_excel(path)
                cols = {str(c).strip() for c in frame.columns}
                missing = {'model', 'Cosine', 'std'} - cols
                if missing:
                    raise ValueError(f'missing columns: {sorted(missing)}')
            else:
                xls = pd.ExcelFile(path)
                required = {'std'} if name == 'structure' else {'mean', 'std'}
                missing = required - set(xls.sheet_names)
                if missing:
                    raise ValueError(f'missing sheets: {sorted(missing)}')
                for sheet in required:
                    frame = pd.read_excel(xls, sheet_name=sheet)
                    if frame.empty:
                        raise ValueError(f'{sheet} sheet is empty')
                    cols = {str(c).strip() for c in frame.columns}
                    required_cols = ({'pub', 't_percent'} if name == 'coreference' else
                                     {'pub', '25-shot', '50-shot', '75-shot', '100-shot'}
                                     if name == 'paragraph' else set())
                    missing_cols = required_cols - cols
                    if missing_cols:
                        raise ValueError(f'{sheet} missing columns: {sorted(missing_cols)}')
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{name}: {path}: {exc}')
    return errors


def panel_label(ax, label):
    transform = ax.transAxes + ScaledTranslation(-18/72, 9/72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, f'({label})', transform=transform,
            fontsize=8, fontweight='bold', va='bottom', clip_on=False)


def export(fig, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(prefix.with_suffix('.pdf'), bbox_inches='tight')


def capture_panel(ax, name, path):
    """Record the actual mean/std values supplied to the errorbar artists."""
    rows = []
    original = ax.errorbar
    def errorbar(x, y, *args, **kwargs):
        result = original(x, y, *args, **kwargs)
        xx, yy = np.atleast_1d(x), np.atleast_1d(y)
        errors = np.broadcast_to(np.asarray(kwargs.get('yerr', 0)), yy.shape)
        for xpos, mean, std in zip(xx, yy, errors):
            rows.append(dict(panel=name, series=kwargs.get('label', ''),
                             x=str(xpos), mean=float(mean), std=float(std)))
        return result
    ax.errorbar = errorbar
    try:
        DRAW[name](ax, path)
    finally:
        ax.errorbar = original
    panel_label(ax, LETTERS[name])
    return rows


def main():
    data_dir, output_dir = project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shot', type=Path, default=data_dir/'shot.xlsx')
    parser.add_argument('--coreference', type=Path, default=data_dir/'cr.xlsx')
    parser.add_argument('--paragraph', type=Path, default=data_dir/'parag.xlsx')
    parser.add_argument('--structure', type=Path, default=data_dir/'structure.xlsx')
    parser.add_argument('--output-dir', type=Path, default=output_dir/'std')
    parser.add_argument('--panels', nargs='+', choices=list(DRAW), default=list(DRAW),
                        help='Select standalone panels; all four also create the combined figure.')
    args = parser.parse_args()
    names = list(dict.fromkeys(args.panels))
    paths = {name: getattr(args, name) for name in names}
    errors = validate_inputs(paths)
    if errors:
        parser.exit(2, 'Source data required:\n' + '\n'.join(errors) + '\n')
    apply_nature_style(6.5)
    rows = []
    for name in names:
        fig, ax = plt.subplots(figsize=(7.2, 3.2) if name=='structure' else (3.6, 2.5))
        rows.extend(capture_panel(ax, name, paths[name]))
        fig.subplots_adjust(left=0.16, right=0.98, top=0.82,
                            bottom=0.42 if name=='structure' else 0.24)
        export(fig, args.output_dir/f'panel_{LETTERS[name]}_{name}_std')
        plt.close(fig)
    if set(names) == set(DRAW):
        fig = plt.figure(figsize=(7.2, 8.0))
        grid = fig.add_gridspec(3, 2, height_ratios=[0.82, 0.90, 1.62],
                               width_ratios=[0.88, 1.52], hspace=0.85, wspace=0.40)
        axes = dict(shot=fig.add_subplot(grid[0,0]), coreference=fig.add_subplot(grid[0,1]),
                    paragraph=fig.add_subplot(grid[1,:]), structure=fig.add_subplot(grid[2,:]))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.23, top=0.94)
        for name, ax in axes.items():
            capture_panel(ax, name, paths[name])
        export(fig, args.output_dir/'combined_performance_std')
        plt.close(fig)
    pd.DataFrame(rows).to_csv(args.output_dir/'plotted_mean_std.csv', index=False)
    (args.output_dir/'input_manifest.json').write_text(json.dumps(
        {name: os.path.relpath(path, start=Path(__file__).parent) for name, path in paths.items()}, indent=2)+'\n')
    print(f'Exported panels: {", ".join(names)} to {args.output_dir}')


if __name__ == '__main__':
    main()
