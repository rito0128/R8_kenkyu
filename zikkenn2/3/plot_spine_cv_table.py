# -*- coding: utf-8 -*-
"""
spine_cv_summary.csv(3.恒常性の評価)を表画像として書き出す。
縦方向: head-spine0 〜 spine4-spine5 の6区間
横方向: 2D推定結果 / 2D正解データ / 3D推定結果 / 3D正解データ
Camera1・Camera4それぞれ1枚ずつ作成する。

出力: spine_cv_table_camera1.png/.eps/.csv, spine_cv_table_camera4.png/.eps/.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent

df = pd.read_csv(OUT / "spine_cv_summary.csv")

ROW_ORDER = ["head-spine0", "spine0-spine1", "spine1-spine2", "spine2-spine3", "spine3-spine4", "spine4-spine5"]
COLS = [("区間", "区間"), ("2D推定CV", "2D推定結果"), ("2D正解CV", "2D正解データ"), ("3D推定CV", "3D推定結果"), ("3D正解CV", "3D正解データ")]

for cam in sorted(df["camera"].unique()):
    sub = df[df.camera == cam].set_index("区間").reindex(ROW_ORDER).reset_index()

    header = [disp for _, disp in COLS]
    cell_text = []
    for _, row in sub.iterrows():
        line = [row["区間"]]
        for col, _ in COLS[1:]:
            line.append(f"{row[col]:.4g}")
        cell_text.append(line)

    all_rows = [header] + cell_text

    pd.DataFrame(cell_text, columns=header).to_csv(
        OUT / f"spine_cv_table_camera{cam}.csv", index=False, encoding="utf-8-sig"
    )
    print(f"wrote spine_cv_table_camera{cam}.csv")

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(all_rows) + 0.8))
    ax.axis("off")
    table = ax.table(cellText=all_rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    for (r, c_), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        if c_ == 0:
            cell.set_text_props(ha="left")

    ax.set_title(f"恒常性の評価: 6区間のCV(標準偏差/平均) — Camera{cam}", fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(OUT / f"spine_cv_table_camera{cam}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"spine_cv_table_camera{cam}.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote spine_cv_table_camera{cam}.png / .eps")
