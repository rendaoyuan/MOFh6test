import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================
# 1) Style
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
# 2) Load data
# =========================
df = pd.read_excel(Path(__file__).parents[3] / "performance_jmi_1" / "shot.xlsx")

# 兼容列名空格（可选但建议保留）
df.columns = [c.strip() for c in df.columns]

df = df[["model", "Cosine", "std"]].copy()
df["Cosine"] = pd.to_numeric(df["Cosine"], errors="coerce")
df["std"] = pd.to_numeric(df["std"], errors="coerce")
df = df.dropna(subset=["Cosine", "std"])

# 按 Cosine 排序（从低到高，阅读更直观）
df = df.sort_values("Cosine", ascending=True).reset_index(drop=True)

labels = df["model"].tolist()
means  = df["Cosine"].to_numpy()
stds   = df["std"].to_numpy()

# 颜色：GnBu 离散取色
colors = sns.color_palette("GnBu", n_colors=len(df))

# =========================
# 3) Dot plot (mean ± std)
# =========================
fig, ax = plt.subplots(figsize=(4.0, 3.6))

x = np.arange(len(df))

# 误差棒 + 点（每个点用不同颜色）
for i in range(len(df)):
    ax.errorbar(
        x[i], means[i], yerr=stds[i],
        fmt="o",
        markersize=7,
        color=colors[i],          # 点颜色
        ecolor=colors[i],         # 误差棒颜色 = 点颜色
        elinewidth=1.6,
        capsize=4,
        capthick=1.6
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)

ax.set_ylabel("Cosine similarity")
ax.set_xlabel("Model")

# 视情况收紧 y 轴范围（让误差棒更清晰）
ymin = max(0, float(np.min(means - stds)) - 0.01)
ymax = min(1.0, float(np.max(means + stds)) + 0.01)
ax.set_ylim(ymin, ymax)

plt.tight_layout()

# =========================
# 4) Save
# =========================
plt.savefig("cosine_dotplot_with_errorbars.png", dpi=300, bbox_inches="tight")
plt.savefig("cosine_dotplot_with_errorbars.pdf", bbox_inches="tight")

plt.show()
