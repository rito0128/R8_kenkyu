# -*- coding: utf-8 -*-
"""
17. Camera1のframe5→frame103について、各区間長の「フレーム間での変化の割合」
    (変化率(%) = (frame103の値 - frame5の値) / frame5の値 × 100)を表にする。
    3D推定値は無次元(正規化スケール)で絶対値の比較が難しいため、割合(%)にすることで
    2D・3Dを同じ尺度で比較できるようにした。16番のデータ定義を踏襲。

    ・19点モデル: spine0-spine1〜spine4-spine5の5区間+その合計の変化率
    ・17点モデル: 2D=肩-腰(shoulder-hip)の変化率、3D=首の付け根(Neck/Nose)-中間(Spine)-
      腰(Hip)の3点による2区間+その合計の変化率

出力: zikkenn2/17/
  change_rate_table.csv
  change_rate_table.png/.eps
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "17"
CAM = 1
TARGET_FRAMES = [5, 103]

pred2d = c.load_2d_pred("19pt", CAM)
gt2d = c.load_2d_gt(CAM)
pred3d = c.load_3d_pred("19pt", CAM)
gt3d = c.load_3d_gt(CAM)
k17_2d = c.load_2d_pred("17pt", CAM)
k17_3d = c.load_3d_pred("17pt", CAM)

mask = np.isin(pred2d["frame"].values, TARGET_FRAMES)
frames = pred2d["frame"].values[mask]  # [5, 103]の順

SHOULDER_HIP_LABEL = "shoulder-hip(17点,2Dのみ)"
SUM_LABEL = "spine0〜spine5 合計"
HIP_SPINE_LABEL = "腰-中間(17点,3Dのみ)"
SPINE_NECK_LABEL = "中間-首(17点,3Dのみ)"
HSN_SUM_LABEL = "腰-中間-首 合計(17点,3Dのみ)"
SPINE_ONLY_CHAIN = c.SPINE_CHAIN[1:]
SPINE_ONLY_LABELS = c.SPINE_CHAIN_LABELS[1:]
LABELS = (list(SPINE_ONLY_LABELS) + [SUM_LABEL, SHOULDER_HIP_LABEL]
          + [HIP_SPINE_LABEL, SPINE_NECK_LABEL, HSN_SUM_LABEL])
SUM_LABELS = {SUM_LABEL, HSN_SUM_LABEL}

# {区間: {"2D": [v_frame5, v_frame103], "3D": [...]}}
values = {}

sum2d = np.zeros(2)
sum3d = np.zeros(2)
for (a, b), label in zip(SPINE_ONLY_CHAIN, SPINE_ONLY_LABELS):
    l2p = c.dist(c.pred_xy(pred2d, a), c.pred_xy(pred2d, b))[mask]
    l3p = c.dist(c.pred_xyz(pred3d, a), c.pred_xyz(pred3d, b))[mask]
    values[label] = {"2D": l2p, "3D": l3p}
    sum2d += l2p
    sum3d += l3p
values[SUM_LABEL] = {"2D": sum2d, "3D": sum3d}

k17_shoulder_mid = (c.pred_xy(k17_2d, 5) + c.pred_xy(k17_2d, 6)) / 2
k17_hip_mid = (c.pred_xy(k17_2d, 11) + c.pred_xy(k17_2d, 12)) / 2
sh2p = c.dist(k17_shoulder_mid, k17_hip_mid)[mask]
values[SHOULDER_HIP_LABEL] = {"2D": sh2p, "3D": np.array([np.nan, np.nan])}

hip_p = c.pred_xyz(k17_3d, 0)
mid_p = c.pred_xyz(k17_3d, 7)
neck_p = c.pred_xyz(k17_3d, 9)
hip_spine_3dp = c.dist(hip_p, mid_p)[mask]
spine_neck_3dp = c.dist(mid_p, neck_p)[mask]
values[HIP_SPINE_LABEL] = {"2D": np.array([np.nan, np.nan]), "3D": hip_spine_3dp}
values[SPINE_NECK_LABEL] = {"2D": np.array([np.nan, np.nan]), "3D": spine_neck_3dp}
values[HSN_SUM_LABEL] = {"2D": np.array([np.nan, np.nan]), "3D": hip_spine_3dp + spine_neck_3dp}


def pct_change(v0, v1):
    return (v1 - v0) / v0 * 100


rows = []
for label in LABELS:
    v2 = values[label]["2D"]
    v3 = values[label]["3D"]
    rows.append({
        "区間": label,
        "2D_frame5": v2[0], "2D_frame103": v2[1], "2D変化率(%)": pct_change(v2[0], v2[1]),
        "3D_frame5": v3[0], "3D_frame103": v3[1], "3D変化率(%)": pct_change(v3[0], v3[1]),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "change_rate_table.csv", index=False, encoding="utf-8-sig")
print("wrote change_rate_table.csv")
print(df.to_string(index=False))

# ---------------- 表画像(変化率(%)のみを主表示、参考にframe5/103の値も併記) ----------------
def fmt(v):
    return "—" if pd.isna(v) else f"{v:.4g}"


header = ["区間", "2D_frame5", "2D_frame103", "2D変化率(%)", "3D_frame5", "3D_frame103", "3D変化率(%)"]
cell_text = []
for _, row in df.iterrows():
    cell_text.append([row["区間"]] + [fmt(row[c_]) for c_ in header[1:]])
all_rows = [header] + cell_text

fig, ax = plt.subplots(figsize=(15, 0.55 * len(LABELS) + 1.5))
ax.axis("off")
col_widths = [0.24, 0.125, 0.135, 0.14, 0.125, 0.135, 0.14]
table = ax.table(cellText=all_rows, loc="center", cellLoc="center", colWidths=col_widths)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
sum_row_idxs = {1 + i for i, label in enumerate(df["区間"]) if label in SUM_LABELS}
for (r, cidx), cell in table.get_celld().items():
    cell.set_edgecolor("#444444")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(weight="bold")
    if r in sum_row_idxs:
        cell.set_facecolor("#f7f0e0")
        cell.set_text_props(weight="bold")
    if cidx == 0:
        cell.set_text_props(ha="left")
    if cidx in (3, 6):  # 変化率(%)列を強調
        cell.set_facecolor(cell.get_facecolor() if r in sum_row_idxs else "#fafafa")

ax.set_title(f"Camera{CAM}: frame5→frame103 の変化率(%)", fontsize=13, pad=12)
fig.tight_layout()
fig.savefig(OUT / "change_rate_table.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "change_rate_table.eps", bbox_inches="tight")
plt.close(fig)
print("wrote change_rate_table.png / .eps")
