"""Notebook std cells adapted only for axes injection, paths and publication styling.

Data preparation, order, offsets, mean/std pairing and y-limit calculations
are retained from the extracted cells. Structure uses Precision/Recall/F1.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

def draw_shot(ax, input_path):
    df = pd.read_excel(input_path)
    df.columns = [c.strip() for c in df.columns]
    df = df[['model', 'Cosine', 'std']].copy()
    df['Cosine'] = pd.to_numeric(df['Cosine'], errors='coerce')
    df['std'] = pd.to_numeric(df['std'], errors='coerce')
    df = df.dropna(subset=['Cosine', 'std'])
    df = df.sort_values('Cosine', ascending=True).reset_index(drop=True)
    labels = df['model'].tolist()
    means = df['Cosine'].to_numpy()
    stds = df['std'].to_numpy()
    colors = sns.color_palette('GnBu', n_colors=len(df))
    x = np.arange(len(df))
    for i in range(len(df)):
        ax.errorbar(x[i], means[i], yerr=stds[i], fmt='o', markersize=3.5, color=colors[i], ecolor=colors[i], elinewidth=0.8, capsize=2, capthick=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel('Cosine similarity')
    ax.set_xlabel('Model')
    ymin = max(0, float(np.min(means - stds)) - 0.01)
    ymax = min(1.0, float(np.max(means + stds)) + 0.01)
    ax.set_ylim(ymin, ymax)
    style_axes(ax)

def draw_coreference(ax, input_path):
    xlsx_path = input_path
    cmap = sns.color_palette('GnBu', as_cmap=True)
    df_mean = pd.read_excel(xlsx_path, sheet_name='mean')
    df_sted = pd.read_excel(xlsx_path, sheet_name='std')
    pub_order = df_mean['pub'].astype(str).str.strip()
    df_mean['pub'] = pub_order
    df_sted['pub'] = pub_order
    stats = pd.DataFrame({'pub': pub_order, 'mean': df_mean['t_percent'].astype(float), 'sted': df_sted['t_percent'].astype(float)})
    desired_order = ['ACS', 'Elsevier', 'RSC', 'Springer', 'Wiley', 'Total']
    stats['pub'] = stats['pub'].astype(str)
    stats = stats.set_index('pub').reindex(desired_order).reset_index()
    x = np.arange(len(stats))
    y = stats['mean'].to_numpy()
    e = stats['sted'].to_numpy()
    colors = cmap(np.linspace(0.25, 0.9, len(stats)))
    for i in range(len(stats)):
        ax.errorbar(x[i], y[i], yerr=e[i], fmt='o', markersize=3.5, elinewidth=0.8, capsize=2, capthick=0.8, color=colors[i], ecolor=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(stats['pub'], rotation=30, ha='right')
    ax.set_ylabel('Resolution rate (%)')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(axis='y', alpha=0.25)
    ax.grid(axis='x', visible=False)
    style_axes(ax)
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")

def draw_paragraph(ax, input_path):
    xlsx_path = input_path
    df_mean = pd.read_excel(xlsx_path, sheet_name='mean')
    df_std = pd.read_excel(xlsx_path, sheet_name='std')
    df_mean.columns = [c.strip() for c in df_mean.columns]
    df_std.columns = [c.strip() for c in df_std.columns]
    shot_columns = ['25-shot', '50-shot', '75-shot', '100-shot']
    mean_long = pd.melt(df_mean, id_vars=['pub'], value_vars=shot_columns, var_name='shot', value_name='value')
    std_long = pd.melt(df_std, id_vars=['pub'], value_vars=shot_columns, var_name='shot', value_name='std')
    df_long = mean_long.merge(std_long, on=['pub', 'shot'])
    df_long['shot'] = pd.Categorical(df_long['shot'], categories=shot_columns, ordered=True)
    palette = sns.color_palette('GnBu', n_colors=len(shot_columns))
    markers = ['o', 's', '^', 'D']
    for i, shot in enumerate(shot_columns):
        sub = df_long[df_long['shot'] == shot]
        ax.errorbar(sub['pub'], sub['value'], yerr=sub['std'], fmt=markers[i], linestyle='none', markersize=3.5, color=palette[i], ecolor=palette[i], elinewidth=0.8, capsize=2, capthick=0.8, label=shot)
    ax.set_ylabel('Cosine similarity')
    ax.set_xlabel('Publisher')
    ax.set_ylim(0.88, 1.0)
    style_axes(ax)
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))

def draw_structure(ax, input_path):
    xlsx_path = input_path
    std_sheet = 'std'
    mean_sheet = 'mean'
    metrics = ['Precision', 'Recall', 'F1']
    desired_cols = ['Metal Source', 'Organic Linkers Source', 'Modulator Source', 'Solvent Source', 'Quantity of Metal', 'Quantity of Organic Linkers', 'Quantity of Modulator', 'Quantity of Solvent', 'pH', 'Synthesis Temperature', 'Synthesis Time', 'Equipment', 'Crystal Morphology', 'Yield']
    palette = sns.color_palette('GnBu', n_colors=len(metrics))
    colors = dict(zip(metrics, palette))
    markers = {'Accuracy': 'o', 'Precision': 's', 'Recall': '^', 'F1': 'D'}
    xls = pd.ExcelFile(xlsx_path)
    std_df_raw = pd.read_excel(xls, sheet_name=std_sheet)
    std_df_raw.columns = [c.strip() if isinstance(c, str) else c for c in std_df_raw.columns]
    if 'Metric' in std_df_raw.columns:
        std_df_raw = std_df_raw.set_index('Metric')
    else:
        std_df_raw = std_df_raw.set_index(std_df_raw.columns[0])
    std_df = std_df_raw.reindex(index=metrics, columns=desired_cols)
    std_df = std_df.apply(pd.to_numeric, errors='coerce')
    has_mean = mean_sheet in xls.sheet_names
    mean_df = None
    if has_mean:
        mean_df_raw = pd.read_excel(xls, sheet_name=mean_sheet)
        mean_df_raw.columns = [c.strip() if isinstance(c, str) else c for c in mean_df_raw.columns]
        if 'Metric' in mean_df_raw.columns:
            mean_df_raw = mean_df_raw.set_index('Metric')
        else:
            mean_df_raw = mean_df_raw.set_index(mean_df_raw.columns[0])
        mean_df = mean_df_raw.reindex(index=metrics, columns=desired_cols)
        mean_df = mean_df.apply(pd.to_numeric, errors='coerce')
    all_cols = [c for c in desired_cols if c in std_df.columns]
    x = np.arange(len(all_cols))
    width = 0.18
    if has_mean:
        for i, m in enumerate(metrics):
            y = mean_df.loc[m, all_cols].values.astype(float)
            e = std_df.loc[m, all_cols].values.astype(float)
            ax.errorbar(x + (i - 1.5) * width, y, yerr=e, fmt=markers[m], color=colors[m], ecolor=colors[m], capsize=2, elinewidth=0.8, capthick=0.8, markersize=3.5, label=m, linestyle='none')
        ax.set_ylabel('Performance (mean ± std)')
    else:
        for i, m in enumerate(metrics):
            y = std_df.loc[m, all_cols].values.astype(float)
            ax.plot(x, y, marker=markers[m], markersize=3.5, linestyle='none', color=colors[m], label=m)
        ax.set_ylabel('Std (across runs)')
    ax.set_xticks(x)
    ax.set_xticklabels(all_cols, rotation=45, ha='right')
    y_all = []
    if has_mean:
        y_all = np.r_[mean_df.loc[:, all_cols].values.flatten(), (mean_df.loc[:, all_cols].values + std_df.loc[:, all_cols].values).flatten(), (mean_df.loc[:, all_cols].values - std_df.loc[:, all_cols].values).flatten()]
    else:
        y_all = std_df.loc[:, all_cols].values.flatten()
    y_all = y_all[np.isfinite(y_all)]
    if len(y_all) > 0:
        ymin = max(0.0, float(np.min(y_all)) - 0.02)
        ymax = min(1.0, float(np.max(y_all)) + 0.02)
        ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', alpha=0.25)
    style_axes(ax)
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    labels = [label.get_text().replace(" ", "\n", 1) for label in ax.get_xticklabels()]
    ax.set_xticks(x, labels, rotation=45, ha="right")

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for name in ('left', 'bottom'):
        ax.spines[name].set_linewidth(0.8)
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.4, alpha=0.5)
    ax.grid(axis='x', visible=False)
    ax.set_axisbelow(True)
