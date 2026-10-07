# -*- coding: utf-8 -*-
"""
axis1_2d.csv / axis1_3d.csv を1つの表(関節, 2D推定範囲, 2D正解範囲, 3D推定範囲, 3D正解範囲)にまとめる。
中身はaxis1_range_of_motion.csvと同じだが、単独の表画像として作り直す。

出力: zikkenn2/18/axis1_combined.csv/.png/.eps
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent

df_2d = pd.read_csv(OUT / "axis1_2d.csv")
df_3d = pd.read_csv(OUT / "axis1_3d.csv")
df = df_2d.merge(df_3d, on="関節")
df.to_csv(OUT / "axis1_combined.csv", index=False, encoding="utf-8-sig")
print("wrote axis1_combined.csv")

header = list(df.columns)
all_rows = [header] + df.values.tolist()
fig, ax = plt.subplots(figsize=(13, 0.55 * len(all_rows) + 1.3))
ax.axis("off")
col_widths = [0.16, 0.21, 0.21, 0.21, 0.21]
table = ax.table(cellText=all_rows, loc="center", cellLoc="center", colWidths=col_widths)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.9)
for (r, cidx), cell in table.get_celld().items():
    cell.set_edgecolor("#444444")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(weight="bold")
    if cidx == 0:
        cell.set_text_props(ha="left")
ax.set_title("① 可動域(Range of Motion) — 2D・3D 推定 vs 正解\nCamera4, frame270-386", fontsize=13, pad=12)
fig.tight_layout()
fig.savefig(OUT / "axis1_combined.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "axis1_combined.eps", bbox_inches="tight")
plt.close(fig)
print("wrote axis1_combined.png / .eps")
