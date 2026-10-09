import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ======================
# 0) Config
# ======================
xlsx_path = Path(__file__).parents[3] / "performance_jmi_1" / "structure.xlsx"
std_sheet = "std"
mean_sheet = "mean"   # 若不存在，会自动降级为只画 std

metrics = [ "Precision", "Recall", "F1"]

# 按你截图的顺序（请保持一致）
desired_cols = [
    "Metal Source",
    "Organic Linkers Source",
    "Modulator Source",
    "Solvent Source",
    "Quantity of Metal",
    "Quantity of Organic Linkers",
    "Quantity of Modulator",
    "Quantity of Solvent",
    "pH",
    "Synthesis Temperature",
    "Synthesis Time",
    "Equipment",
    "Crystal Morphology",
    "Yield",
]

# ======================
# 1) Style (你的风格)
# ======================
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

palette = sns.color_palette("GnBu", n_colors=len(metrics))
colors = dict(zip(metrics, palette))
markers = {"Accuracy": "o", "Precision": "s", "Recall": "^", "F1": "D"}

# ======================
# 2) Load sheets
# ======================
xls = pd.ExcelFile(xlsx_path)
std_df_raw = pd.read_excel(xls, sheet_name=std_sheet)

# 兼容列名空格
std_df_raw.columns = [c.strip() if isinstance(c, str) else c for c in std_df_raw.columns]

# 识别“指标列”作为行名（优先 Metric，否则用第一列）
if "Metric" in std_df_raw.columns:
    std_df_raw = std_df_raw.set_index("Metric")
else:
    std_df_raw = std_df_raw.set_index(std_df_raw.columns[0])

# 只保留你关心的指标与列顺序
std_df = std_df_raw.reindex(index=metrics, columns=desired_cols)
std_df = std_df.apply(pd.to_numeric, errors="coerce")

# 尝试读取 mean（如果存在）
has_mean = (mean_sheet in xls.sheet_names)
mean_df = None
if has_mean:
    mean_df_raw = pd.read_excel(xls, sheet_name=mean_sheet)
    mean_df_raw.columns = [c.strip() if isinstance(c, str) else c for c in mean_df_raw.columns]
    if "Metric" in mean_df_raw.columns:
        mean_df_raw = mean_df_raw.set_index("Metric")
    else:
        mean_df_raw = mean_df_raw.set_index(mean_df_raw.columns[0])

    mean_df = mean_df_raw.reindex(index=metrics, columns=desired_cols)
    mean_df = mean_df.apply(pd.to_numeric, errors="coerce")

# ======================
# 3) Plot
# ======================
all_cols = [c for c in desired_cols if c in std_df.columns]  # 按截图顺序过滤存在列
x = np.arange(len(all_cols))
width = 0.18

fig_w = max(10, len(all_cols) * 0.75)
fig, ax = plt.subplots(figsize=(fig_w, 5))

if has_mean:
    # mean ± std 误差棒图
    for i, m in enumerate(metrics):
        y = mean_df.loc[m, all_cols].values.astype(float)
        e = std_df.loc[m, all_cols].values.astype(float)

        ax.errorbar(
            x + (i - 1.5) * width,
            y,
            yerr=e,
            fmt=markers[m],
            color=colors[m],
            ecolor=colors[m],
            capsize=3,
            elinewidth=1.4,
            capthick=1.4,
            markersize=6,
            label=m,
            linestyle="none"
        )

    ax.set_ylabel("Performance (mean ± std)")
else:
    # 只有 std：画 std 的点图（没有误差棒）
    for i, m in enumerate(metrics):
        y = std_df.loc[m, all_cols].values.astype(float)
        ax.plot(
            x,
            y,
            marker=markers[m],
            markersize=6,
            linestyle="none",
            color=colors[m],
            label=m
        )

    ax.set_ylabel("Std (across runs)")

ax.set_xticks(x)
ax.set_xticklabels(all_cols, rotation=45, ha="right")

# y 轴范围：你可按数据改，这里给一个“自动 + 合理留白”
y_all = []
if has_mean:
    y_all = np.r_[mean_df.loc[:, all_cols].values.flatten(), (mean_df.loc[:, all_cols].values + std_df.loc[:, all_cols].values).flatten(),
                  (mean_df.loc[:, all_cols].values - std_df.loc[:, all_cols].values).flatten()]
else:
    y_all = std_df.loc[:, all_cols].values.flatten()

y_all = y_all[np.isfinite(y_all)]
if len(y_all) > 0:
    ymin = max(0.0, float(np.min(y_all)) - 0.02)
    ymax = min(1.0, float(np.max(y_all)) + 0.02)
    ax.set_ylim(ymin, ymax)
    
ax.grid(axis="y", alpha=0.25)

ax.legend(
    frameon=False,
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout()

# 保存
plt.savefig("structure_metrics_by_field.pdf", bbox_inches="tight")
plt.show()
