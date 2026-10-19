# -*- coding: utf-8 -*-
"""
15. 12番・14番と同じ「19点6区間の長さ合計Δ vs 17点肩-腰の長さΔ」比較を、
    Camera1・frame1〜100(ユーザー指定の"0〜100")で作成する。
    (Camera1のこの範囲は2D推定が不安定な区間を含むことが判明済みだが、
     指定通りそのまま算出する)

出力: zikkenn2/15/
  lengths_camera1_0-100.csv            フレームごとの区間長生データ(2D推定/2D正解/3D推定/3D正解)
  displacement_comparison.csv          フレームごとの19点合計Δ・17点Δ(2D推定/2D正解/3D推定)
  displacement_comparison_2d.png/.eps  2D推定結果+正解データでのグラフ(1枚)
  displacement_comparison_3d.png/.eps  3D推定結果でのグラフ(1枚)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "15"
CAM = 1
FRAME_RANGES = [(0, 100)]


def in_ranges(frames):
    mask = np.zeros(len(frames), dtype=bool)
    for lo, hi in FRAME_RANGES:
        mask |= (frames >= lo) & (frames <= hi)
    return mask


RANGE_LABEL = ", ".join(f"{lo}-{hi}" for lo, hi in FRAME_RANGES)

# ---------------- データ読み込み・区間長の算出(12番と同じ定義) ----------------
pred2d = c.load_2d_pred("19pt", CAM)
gt2d = c.load_2d_gt(CAM)
pred3d = c.load_3d_pred("19pt", CAM)
gt3d = c.load_3d_gt(CAM)
k17_2d = c.load_2d_pred("17pt", CAM)
k17_3d = c.load_3d_pred("17pt", CAM)

frames_all = pred2d["frame"].values
mask = in_ranges(frames_all)
frames = frames_all[mask]

SHOULDER_HIP_LABEL = "shoulder-hip(17点)"
LENGTH_LABELS = list(c.SPINE_CHAIN_LABELS) + [SHOULDER_HIP_LABEL]

length_rows = []
for (a, b), label in zip(c.SPINE_CHAIN, c.SPINE_CHAIN_LABELS):
    l2p = c.dist(c.pred_xy(pred2d, a), c.pred_xy(pred2d, b))[mask]
    l2g = c.dist(c.gt_xy(gt2d, a), c.gt_xy(gt2d, b))[mask]
    l3p = c.dist(c.pred_xyz(pred3d, a), c.pred_xyz(pred3d, b))[mask]
    l3g = c.dist(c.gt_xyz(gt3d, a), c.gt_xyz(gt3d, b))[mask]
    for fr, v2p, v2g, v3p, v3g in zip(frames, l2p, l2g, l3p, l3g):
        length_rows.append({"camera": CAM, "frame": fr, "区間": label,
                             "2D推定": v2p, "2D正解": v2g, "3D推定": v3p, "3D正解": v3g})

# 肩-腰(17点モデル、正解は19点GTの肩・股関節点から算出)
k17_shoulder_mid = (c.pred_xy(k17_2d, 5) + c.pred_xy(k17_2d, 6)) / 2
k17_hip_mid = (c.pred_xy(k17_2d, 11) + c.pred_xy(k17_2d, 12)) / 2
sh2p = c.dist(k17_shoulder_mid, k17_hip_mid)[mask]
k17_shoulder_mid3d = (c.pred_xyz(k17_3d, 11) + c.pred_xyz(k17_3d, 14)) / 2
sh3p = c.dist(k17_shoulder_mid3d, c.pred_xyz(k17_3d, 0))[mask]
gt_shoulder_mid2d = (c.gt_xy(gt2d, 1) + c.gt_xy(gt2d, 2)) / 2
gt_hip_mid2d = (c.gt_xy(gt2d, 7) + c.gt_xy(gt2d, 8)) / 2
sh2g = c.dist(gt_shoulder_mid2d, gt_hip_mid2d)[mask]
gt_shoulder_mid3d = (c.gt_xyz(gt3d, 1) + c.gt_xyz(gt3d, 2)) / 2
gt_hip_mid3d = (c.gt_xyz(gt3d, 7) + c.gt_xyz(gt3d, 8)) / 2
sh3g = c.dist(gt_shoulder_mid3d, gt_hip_mid3d)[mask]

for fr, v2p, v2g, v3p, v3g in zip(frames, sh2p, sh2g, sh3p, sh3g):
    length_rows.append({"camera": CAM, "frame": fr, "区間": SHOULDER_HIP_LABEL,
                         "2D推定": v2p, "2D正解": v2g, "3D推定": v3p, "3D正解": v3g})

len_df = pd.DataFrame(length_rows)
len_df.to_csv(OUT / f"lengths_camera{CAM}_{FRAME_RANGES[0][0]}-{FRAME_RANGES[0][1]}.csv", index=False, encoding="utf-8-sig")
print("wrote lengths csv")

# ---------------- 変位(Δlength)の算出(14番と同じ定義) ----------------
pivot_2d = len_df.pivot(index="frame", columns="区間", values="2D推定").sort_index()
pivot_3d = len_df.pivot(index="frame", columns="区間", values="3D推定").sort_index()
pivot_2d_gt = len_df.pivot(index="frame", columns="区間", values="2D正解").sort_index()

total19_2d = pivot_2d[list(c.SPINE_CHAIN_LABELS)].sum(axis=1)
total19_3d = pivot_3d[list(c.SPINE_CHAIN_LABELS)].sum(axis=1)
total19_2d_gt = pivot_2d_gt[list(c.SPINE_CHAIN_LABELS)].sum(axis=1)
sh17_2d = pivot_2d[SHOULDER_HIP_LABEL]
sh17_3d = pivot_3d[SHOULDER_HIP_LABEL]
sh17_2d_gt = pivot_2d_gt[SHOULDER_HIP_LABEL]

out_df = pd.DataFrame({
    "frame": pivot_2d.index,
    "19点区間合計Δ_2D推定": total19_2d.diff().values,
    "17点肩腰Δ_2D推定": sh17_2d.diff().values,
    "19点区間合計Δ_2D正解": total19_2d_gt.diff().values,
    "17点肩腰Δ_2D正解": sh17_2d_gt.diff().values,
    "19点区間合計Δ_3D推定": total19_3d.diff().values,
    "17点肩腰Δ_3D推定": sh17_3d.diff().values,
}).dropna().reset_index(drop=True)
out_df.to_csv(OUT / "displacement_comparison.csv", index=False, encoding="utf-8-sig")
print("wrote displacement_comparison.csv")
print(out_df.describe().to_string())


def plot_delta(dim, series_specs, ylabel, title_suffix, unit_note):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for col, label, color, style in series_specs:
        ax.plot(out_df["frame"], out_df[col], color=color, linestyle=style, linewidth=1.4, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dim}{title_suffix}: フレーム間の変位(Δlength) 比較\nCamera{CAM}, frame{RANGE_LABEL}{unit_note}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"displacement_comparison_{dim.lower()}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"displacement_comparison_{dim.lower()}.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote displacement_comparison_{dim.lower()}.png / .eps")


plot_delta("2D", [
    ("19点区間合計Δ_2D推定", "19点: 6区間の長さ合計のΔ(推定)", "tab:red", "-"),
    ("17点肩腰Δ_2D推定", "17点: 肩-腰の長さのΔ(推定)", "tab:blue", "-"),
    ("19点区間合計Δ_2D正解", "19点: 6区間の長さ合計のΔ(正解)", "tab:red", "--"),
    ("17点肩腰Δ_2D正解", "17点: 肩-腰の長さのΔ(正解)", "tab:blue", "--"),
], "Δlength [px]", "推定結果+正解データ", "")

plot_delta("3D", [
    ("19点区間合計Δ_3D推定", "19点: 6区間の長さ合計のΔ(推定)", "tab:red", "-"),
    ("17点肩腰Δ_3D推定", "17点: 肩-腰の長さのΔ(推定)", "tab:blue", "-"),
], "Δlength [normalized]", "推定結果", "\n(3Dは推定側の正規化スケール)")
