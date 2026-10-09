import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# =========================
# 1) Style（与你之前一致）
# =========================
sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "axes.facecolor": "white",
    "grid.color": "gray",
    "grid.alpha": 0.2,
    "grid.linestyle": "-",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "font.size": 12,
    "figure.facecolor": "white",
})
sns.set_context("notebook", font_scale=1.2)

# =========================
# 2) Load mean + std
# =========================
xlsx_path = Path(__file__).parents[3] / "performance_jmi_1" / "parag.xlsx"
df_mean = pd.read_excel(xlsx_path, sheet_name="mean")
df_std  = pd.read_excel(xlsx_path, sheet_name="std")

df_mean.columns = [c.strip() for c in df_mean.columns]
df_std.columns  = [c.strip() for c in df_std.columns]

shot_columns = ['25-shot', '50-shot', '75-shot', '100-shot']

# =========================
# 3) Long format（不画 Total，可保留也可删）
# =========================
mean_long = pd.melt(df_mean, id_vars=['pub'], value_vars=shot_columns,
                    var_name='shot', value_name='value')
std_long  = pd.melt(df_std,  id_vars=['pub'], value_vars=shot_columns,
                    var_name='shot', value_name='std')

df_long = mean_long.merge(std_long, on=['pub', 'shot'])

# 如不想画 Total，取消下面注释
# df_long = df_long[df_long['pub'] != 'Total']

# shot 顺序
df_long['shot'] = pd.Categorical(
    df_long['shot'],
    categories=shot_columns,
    ordered=True
)

# =========================
# 4) Plot: 单一框点图 + 误差棒
# =========================
plt.figure(figsize=(10, 5))

palette = sns.color_palette("GnBu", n_colors=len(shot_columns))
markers = ['o', 's', '^', 'D']

for i, shot in enumerate(shot_columns):
    sub = df_long[df_long['shot'] == shot]

    plt.errorbar(
        sub['pub'],
        sub['value'],
        yerr=sub['std'],
        fmt=markers[i],
        linestyle='none',
        markersize=7,
        color=palette[i],
        ecolor=palette[i],
        elinewidth=1.5,
        capsize=3,
        capthick=1.5,
        label=shot
    )

# =========================
# 5) Axis & legend
# =========================
plt.ylabel("Cosine similarity")
plt.xlabel("Publisher")

plt.ylim(0.88, 1.00)  # 根据你数据范围微调
plt.xticks(rotation=30, ha='right')

plt.legend(
    frameon=False,
    ncol=4,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15)
)

# 黑色边框
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1.0)

plt.tight_layout()
plt.savefig("parag_singlepanel_errorbar.pdf", dpi=300, bbox_inches="tight")
plt.show()
