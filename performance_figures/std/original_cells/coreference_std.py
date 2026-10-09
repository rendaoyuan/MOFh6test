import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from pathlib import Path

# ======================
# 尺寸与风格（完全沿用你的）
# ======================
FIGURE_WIDTH = 6
LEFT_MARGIN = 0.15
RIGHT_MARGIN = 0.75
TOP_MARGIN = 0.95
BOTTOM_MARGIN = 0.1

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

# ======================
# GnBu 调色板
# ======================
cmap = sns.color_palette("GnBu", as_cmap=True)

# ======================
# 1) Read Excel (mean & std)
# ======================
xlsx_path = Path(__file__).parents[3] / "performance_jmi_1" / "cr.xlsx"

df_mean = pd.read_excel(xlsx_path, sheet_name="mean")
df_sted = pd.read_excel(xlsx_path, sheet_name="std")

# 保证顺序一致
pub_order = df_mean["pub"].astype(str).str.strip()
df_mean["pub"] = pub_order
df_sted["pub"] = pub_order

# ======================
# 2) 合并 mean + sted
# ======================
stats = pd.DataFrame({
    "pub": pub_order,
    "mean": df_mean["t_percent"].astype(float),
    "sted": df_sted["t_percent"].astype(float)
})

# ======================
# 强制指定 x 轴顺序
# ======================
desired_order = ["ACS", "Elsevier", "RSC", "Springer", "Wiley", "Total"]

stats["pub"] = stats["pub"].astype(str)
stats = stats.set_index("pub").reindex(desired_order).reset_index()

# ======================
# 3) Plot：GnBu 点图 + 同色误差棒
# ======================
fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 5))

plt.subplots_adjust(
    left=LEFT_MARGIN,
    right=RIGHT_MARGIN,
    top=TOP_MARGIN,
    bottom=BOTTOM_MARGIN
)

x = np.arange(len(stats))
y = stats["mean"].to_numpy()
e = stats["sted"].to_numpy()

# 颜色从 GnBu 渐变中取（避开过浅）
colors = cmap(np.linspace(0.25, 0.9, len(stats)))

for i in range(len(stats)):
    ax.errorbar(
        x[i], y[i], yerr=e[i],
        fmt="o",
        markersize=7,
        elinewidth=1.6,
        capsize=4,
        capthick=1.6,
        color=colors[i],           # 点颜色
        ecolor=colors[i]  # 点边框黑色
    )

# ======================
# 4) 坐标轴与格式
# ======================
ax.set_xticks(x)
ax.set_xticklabels(stats["pub"], rotation=30, ha="right")
ax.set_ylabel("Resolution rate (%)")

ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

# 只保留 y 网格
ax.grid(axis="y", alpha=0.25)
ax.grid(axis="x", visible=False)

# 黑色边框
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1.5)

# ======================
# 5) 保存
# ======================
plt.tight_layout()
plt.savefig("cr_t_percent_mean_sted.pdf", dpi=300, bbox_inches="tight")
plt.show()
