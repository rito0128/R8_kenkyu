# -*- coding: utf-8 -*-
"""
27. 拡張前(17点)モデルと拡張後(19点)モデルの3D推定結果について、Camera1のframe5・frame103
    における区間長と、その間の変化(変化量・変化率)を表にする。

    拡張前(17点, H36M17): 首の付け根(neck_nose, idx9)-背中(spine, idx7)-腰(hip, idx0)
                          の3点・2区間
    拡張後(19点): spine1-spine2-spine3-spine4-spine5 の4区間
                 (首側の1区間 spine0-spine1 は対象から除く、との指定に基づく)

    各モデルについて、区間ごとの長さ(frame5, frame103)・変化量(frame103-frame5)・
    変化率(%)に加え、区間の合計値の行も含める。GTは対象外、3D推定のみ。

出力: zikkenn2/27/
  frame5_103_change.csv   表の生データ
  frame5_103_change.png/.eps  表画像
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "27"
CAM = 1
TARGET_FRAMES = [5, 103]

pred17 = c.load_3d_pred("17pt", CAM)
pred19 = c.load_3d_pred("19pt", CAM)

mask17 = np.isin(pred17["frame"].values, TARGET_FRAMES)
mask19 = np.isin(pred19["frame"].values, TARGET_FRAMES)
frames17 = pred17["frame"].values[mask17]
frames19 = pred19["frame"].values[mask19]
assert list(frames17) == TARGET_FRAMES and list(frames19) == TARGET_FRAMES

MODEL17_LABEL = "拡張前(17点)"
MODEL19_LABEL = "拡張後(19点)"
SUM_LABEL_SUFFIX = " 合計"

# 拡張前(17点, H36M17): hip(0)-spine(7)-neck_nose(9)
CHAIN17 = [(9, 7, "首の付け根-背中"), (7, 0, "背中-腰")]
# 拡張後(19点): spine1(14)-spine2(15)-spine3(16)-spine4(17)-spine5(18)、首側の1区間(spine0-spine1)は除く
CHAIN19 = [(14, 15, "spine1-spine2"), (15, 16, "spine2-spine3"),
           (16, 17, "spine3-spine4"), (17, 18, "spine4-spine5")]


def build_rows(model_label, pred_df, mask, chain):
    rows = []
    sum_by_frame = {fr: 0.0 for fr in TARGET_FRAMES}
    for a, b, label in chain:
        lens = c.dist(c.pred_xyz(pred_df, a), c.pred_xyz(pred_df, b))[mask]
        for fr, v in zip(TARGET_FRAMES, lens):
            sum_by_frame[fr] += v
        f5, f103 = lens[0], lens[1]
        rows.append({
            "モデル": model_label, "区間": label,
            "frame5": f5, "frame103": f103,
            "変化量": f103 - f5, "変化率(%)": (f103 - f5) / f5 * 100,
        })
    f5s, f103s = sum_by_frame[TARGET_FRAMES[0]], sum_by_frame[TARGET_FRAMES[1]]
    rows.append({
        "モデル": model_label, "区間": model_label + SUM_LABEL_SUFFIX,
        "frame5": f5s, "frame103": f103s,
        "変化量": f103s - f5s, "変化率(%)": (f103s - f5s) / f5s * 100,
    })
    return rows


rows = build_rows(MODEL17_LABEL, pred17, mask17, CHAIN17) + build_rows(MODEL19_LABEL, pred19, mask19, CHAIN19)
df = pd.DataFrame(rows)

csv_path = OUT / "frame5_103_change.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"wrote {csv_path}")
print(df.to_string(index=False))

# ---------------- 表画像 ----------------
def fmt(v):
    return f"{v:.4g}"


header = ["モデル", "区間", "frame5", "frame103", "変化量", "変化率(%)"]
cell_text = []
for _, row in df.iterrows():
    cell_text.append([row["モデル"], row["区間"]] + [fmt(row[c_]) for c_ in header[2:]])
all_rows = [header] + cell_text

sum_row_idxs = {1 + i for i, label in enumerate(df["区間"]) if label.endswith(SUM_LABEL_SUFFIX)}

fig, ax = plt.subplots(figsize=(11, 0.5 * len(all_rows) + 1))
ax.axis("off")
col_widths = [0.18, 0.24, 0.145, 0.145, 0.145, 0.145]
table = ax.table(cellText=all_rows, loc="center", cellLoc="center", colWidths=col_widths)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

for (r, cidx), cell in table.get_celld().items():
    cell.set_edgecolor("#444444")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(weight="bold")
    if r in sum_row_idxs:
        cell.set_facecolor("#f7f0e0")
        cell.set_text_props(weight="bold")
    if cidx in (0, 1):
        cell.set_text_props(ha="left")

ax.set_title(f"Camera{CAM} 3D推定: frame5→frame103 の区間長変化", fontsize=13, pad=12)
fig.tight_layout()
fig.savefig(OUT / "frame5_103_change.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "frame5_103_change.eps", bbox_inches="tight")
plt.close(fig)
print("wrote frame5_103_change.png / .eps")
