# -*- coding: utf-8 -*-
"""
16. Camera1のframe5・frame103について、19点モデルのspine0-spine1〜spine4-spine5の5区間長
    (head-spine0は除く)+その合計値を、2D推定・2D正解・3D推定・3D正解の4列で表にする
    (12番・15番と同じ定義)。

    17点モデルは、2Dは肩-腰(shoulder-hip、COCO17に脊柱点が無いための概算)のまま、
    3Dはユーザー指定により「腰(Hip)・中間の点(Spine)・首の付け根(Neck/Nose)」の3点
    (H36M17が実際に持つ関節点。Spineはhip-neck間のほぼ中点<50.5%>に位置することを確認済み)
    による2区間(腰-中間、中間-首)+その合計値に変更した。
    正解データ側は19点GTの spine5(J18)=腰に近い、spine0(J13)=首に近い、両者の中点=中間の点
    として対応させた(spine0が首寄り・spine5が腰寄りという、これまでの分析で確認済みの配置に基づく)。
    17点モデルの2D(COCO17)には脊柱点が無いため、この3D専用の3行は2D欄を「—」とした。
    同様にshoulder-hip(17点)行の3D欄は、この新しい3点分解に置き換わるため「—」とした。

    正解データ(2D正解・3D正解)は表から省き、推定結果(2D推定・3D推定)のみを示す。

出力: zikkenn2/16/
  frame5_103_table.csv   表の生データ(2D推定・3D推定のみ)
  frame5_103_table.png/.eps  表画像(frameごとに区切って表示)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "16"
CAM = 1
TARGET_FRAMES = [5, 103]

pred2d = c.load_2d_pred("19pt", CAM)
gt2d = c.load_2d_gt(CAM)
pred3d = c.load_3d_pred("19pt", CAM)
gt3d = c.load_3d_gt(CAM)
k17_2d = c.load_2d_pred("17pt", CAM)
k17_3d = c.load_3d_pred("17pt", CAM)

mask = np.isin(pred2d["frame"].values, TARGET_FRAMES)
frames = pred2d["frame"].values[mask]

SHOULDER_HIP_LABEL = "shoulder-hip(17点,2Dのみ)"
SUM_LABEL = "spine0〜spine5 合計"
HIP_SPINE_LABEL = "腰-中間(17点,3Dのみ)"
SPINE_NECK_LABEL = "中間-首(17点,3Dのみ)"
HSN_SUM_LABEL = "腰-中間-首 合計(17点,3Dのみ)"
# head-spine0を除いた、spine0〜spine5間の5区間
SPINE_ONLY_CHAIN = c.SPINE_CHAIN[1:]
SPINE_ONLY_LABELS = c.SPINE_CHAIN_LABELS[1:]
LABELS = (list(SPINE_ONLY_LABELS) + [SUM_LABEL, SHOULDER_HIP_LABEL]
          + [HIP_SPINE_LABEL, SPINE_NECK_LABEL, HSN_SUM_LABEL])
SUM_LABELS = {SUM_LABEL, HSN_SUM_LABEL}

rows = []
sums = {fr: {"2D推定": 0.0, "2D正解": 0.0, "3D推定": 0.0, "3D正解": 0.0} for fr in frames}
for (a, b), label in zip(SPINE_ONLY_CHAIN, SPINE_ONLY_LABELS):
    l2p = c.dist(c.pred_xy(pred2d, a), c.pred_xy(pred2d, b))[mask]
    l2g = c.dist(c.gt_xy(gt2d, a), c.gt_xy(gt2d, b))[mask]
    l3p = c.dist(c.pred_xyz(pred3d, a), c.pred_xyz(pred3d, b))[mask]
    l3g = c.dist(c.gt_xyz(gt3d, a), c.gt_xyz(gt3d, b))[mask]
    for fr, v2p, v2g, v3p, v3g in zip(frames, l2p, l2g, l3p, l3g):
        rows.append({"frame": fr, "区間": label, "2D推定": v2p, "2D正解": v2g, "3D推定": v3p, "3D正解": v3g})
        sums[fr]["2D推定"] += v2p
        sums[fr]["2D正解"] += v2g
        sums[fr]["3D推定"] += v3p
        sums[fr]["3D正解"] += v3g

for fr in frames:
    rows.append({"frame": fr, "区間": SUM_LABEL, **sums[fr]})

# 肩-腰(17点モデル、2Dのみ。正解は19点GTの肩・股関節点から算出)
k17_shoulder_mid = (c.pred_xy(k17_2d, 5) + c.pred_xy(k17_2d, 6)) / 2
k17_hip_mid = (c.pred_xy(k17_2d, 11) + c.pred_xy(k17_2d, 12)) / 2
sh2p = c.dist(k17_shoulder_mid, k17_hip_mid)[mask]
gt_shoulder_mid2d = (c.gt_xy(gt2d, 1) + c.gt_xy(gt2d, 2)) / 2
gt_hip_mid2d = (c.gt_xy(gt2d, 7) + c.gt_xy(gt2d, 8)) / 2
sh2g = c.dist(gt_shoulder_mid2d, gt_hip_mid2d)[mask]

for fr, v2p, v2g in zip(frames, sh2p, sh2g):
    rows.append({"frame": fr, "区間": SHOULDER_HIP_LABEL, "2D推定": v2p, "2D正解": v2g,
                 "3D推定": np.nan, "3D正解": np.nan})

# 腰(Hip)-中間(Spine)-首の付け根(Neck/Nose) の3点、3Dのみ(17点モデル)
# GT側は19点GTの spine5(腰寄り)-中点-spine0(首寄り) を対応させる
hip_p = c.pred_xyz(k17_3d, 0)
mid_p = c.pred_xyz(k17_3d, 7)
neck_p = c.pred_xyz(k17_3d, 9)
hip_spine_3dp = c.dist(hip_p, mid_p)[mask]
spine_neck_3dp = c.dist(mid_p, neck_p)[mask]

hip_g = c.gt_xyz(gt3d, 18)   # spine5(腰寄り)
neck_g = c.gt_xyz(gt3d, 13)  # spine0(首寄り)
mid_g = (hip_g + neck_g) / 2
hip_spine_3dg = c.dist(hip_g, mid_g)[mask]
spine_neck_3dg = c.dist(mid_g, neck_g)[mask]

for fr, v3p, v3g in zip(frames, hip_spine_3dp, hip_spine_3dg):
    rows.append({"frame": fr, "区間": HIP_SPINE_LABEL, "2D推定": np.nan, "2D正解": np.nan,
                 "3D推定": v3p, "3D正解": v3g})
for fr, v3p, v3g in zip(frames, spine_neck_3dp, spine_neck_3dg):
    rows.append({"frame": fr, "区間": SPINE_NECK_LABEL, "2D推定": np.nan, "2D正解": np.nan,
                 "3D推定": v3p, "3D正解": v3g})
for fr, v3p, v3g in zip(frames, hip_spine_3dp + spine_neck_3dp, hip_spine_3dg + spine_neck_3dg):
    rows.append({"frame": fr, "区間": HSN_SUM_LABEL, "2D推定": np.nan, "2D正解": np.nan,
                 "3D推定": v3p, "3D正解": v3g})

df = pd.DataFrame(rows)
df["_order"] = df["区間"].map({l: i for i, l in enumerate(LABELS)})
df = df.sort_values(["frame", "_order"]).drop(columns="_order").reset_index(drop=True)
df = df[["frame", "区間", "2D推定", "3D推定"]]
df.to_csv(OUT / "frame5_103_table.csv", index=False, encoding="utf-8-sig")
print("wrote frame5_103_table.csv")
print(df.to_string(index=False))

# ---------------- 表画像 ----------------
fig, axes = plt.subplots(len(TARGET_FRAMES), 1, figsize=(12, 0.55 * len(LABELS) * len(TARGET_FRAMES) + 2))
if len(TARGET_FRAMES) == 1:
    axes = [axes]

def fmt(v):
    return "—" if pd.isna(v) else f"{v:.4g}"


for ax, fr in zip(axes, TARGET_FRAMES):
    sub = df[df.frame == fr]
    header = ["区間", "2D推定", "3D推定"]
    cell_text = [[row["区間"]] + [fmt(row[col]) for col in header[1:]] for _, row in sub.iterrows()]
    all_rows = [header] + cell_text

    ax.axis("off")
    col_widths = [0.5, 0.25, 0.25]
    table = ax.table(cellText=all_rows, loc="center", cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    sum_row_idxs = {1 + i for i, label in enumerate(sub["区間"]) if label in SUM_LABELS}
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
    ax.set_title(f"Camera{CAM} frame{fr}", fontsize=12, pad=10)

fig.tight_layout()
fig.savefig(OUT / "frame5_103_table.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "frame5_103_table.eps", bbox_inches="tight")
plt.close(fig)
print("wrote frame5_103_table.png / .eps")
