# -*- coding: utf-8 -*-
"""
axis1_range_of_motion.csv(①可動域)を、2Dのみ(推定vs正解)・3Dのみ(推定vs正解)の
2つの表に分割して見やすくする。

出力: zikkenn2/18/
  axis1_2d.csv/.png/.eps
  axis1_3d.csv/.png/.eps
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent

df = pd.read_csv(OUT / "axis1_range_of_motion.csv")

df_2d = df[["関節", "2D推定範囲", "2D正解範囲"]]
df_3d = df[["関節", "3D推定範囲", "3D正解範囲"]]

df_2d.to_csv(OUT / "axis1_2d.csv", index=False, encoding="utf-8-sig")
df_3d.to_csv(OUT / "axis1_3d.csv", index=False, encoding="utf-8-sig")
print("wrote axis1_2d.csv / axis1_3d.csv")


def draw_table(df, title, path_stub):
    header = list(df.columns)
    all_rows = [header] + df.values.tolist()
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(all_rows) + 1))
    ax.axis("off")
    table = ax.table(cellText=all_rows, loc="center", cellLoc="center")
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
    ax.set_title(title, fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / f"{path_stub}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{path_stub}.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path_stub}.png / .eps")


draw_table(df_2d, "① 可動域(Range of Motion) — 2D: 推定 vs 正解\nCamera4, frame270-386", "axis1_2d")
draw_table(df_3d, "① 可動域(Range of Motion) — 3D: 推定 vs 正解\nCamera4, frame270-386", "axis1_3d")
