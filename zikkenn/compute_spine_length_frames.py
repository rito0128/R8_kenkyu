# -*- coding: utf-8 -*-
"""
恒常性の評価: head-spine0-spine1-...-spine5 の6区間について、
フレーム(横軸)-長さ(縦軸)のグラフと、標準偏差・変動係数・フレーム間変化率の表を
複数パターンで書き出す。

対象: 2D(Camera1)・3D(Camera1)、それぞれ推定結果と正解データ。

出力:
  spine_lengths_frames_2d.png/.eps    2D: 6区間ごとのフレーム-長さグラフ(推定/正解)
  spine_lengths_frames_3d.png/.eps    3D: 6区間ごとのフレーム-長さグラフ(推定/正解)
  spine_lengths_frames_overlay.png/.eps  2D・3Dを1枚に重ねた版(区間ごとに2軸)
  spine_length_stats.csv              平均・標準偏差・CV・フレーム間変化率の表(生データ)
  spine_length_stats.png/.eps         同上を画像化した表
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HR_DIR = BASE_DIR / "for_test_HRNet"
MB_DIR = BASE_DIR / "for_test_motionbert" / "motionbert_20260827_motion_007"

KP_NAMES = ['head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
            'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            'spine0', 'spine1', 'spine2', 'spine3', 'spine4', 'spine5']

CHAIN = [(0, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18)]
LABELS = [f"{KP_NAMES[a]}-{KP_NAMES[b]}" for a, b in CHAIN]


def dist_xy(df, a, b):
    ax, ay = df[f"kp{a}_{KP_NAMES[a]}_x"], df[f"kp{a}_{KP_NAMES[a]}_y"]
    bx, by = df[f"kp{b}_{KP_NAMES[b]}_x"], df[f"kp{b}_{KP_NAMES[b]}_y"]
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def dist_xyz(df, a, b):
    ax, ay, az = df[f"J{a}_x"], df[f"J{a}_y"], df[f"J{a}_z"]
    bx, by, bz = df[f"J{b}_x"], df[f"J{b}_y"], df[f"J{b}_z"]
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


# ---------------- データ読み込み(すべてCamera1) ----------------
pred2d = pd.read_csv(HR_DIR / "keypoint_lengths.csv")
pred2d1 = pred2d[pred2d.camera == 1].sort_values("frame").reset_index(drop=True)
gt2d = pd.read_csv(HR_DIR / "correct_keypoint_lengths.csv")
gt2d1 = gt2d[gt2d.camera == 1].sort_values("frame").reset_index(drop=True)

pred3d = pd.read_csv(MB_DIR / "motionbert_keypoint_lengths.csv")
pred3d1 = pred3d[pred3d.camera == 1].sort_values("frame").reset_index(drop=True)

gt3d = pd.read_csv(MB_DIR / "correct_3d_anotation.csv")
gt3d1 = gt3d[gt3d["filename"].str.startswith("out_Camera1_")].copy()
gt3d1["frame"] = gt3d1["filename"].str.extract(r"_(\d+)$").astype(int)
gt3d1 = gt3d1.sort_values("frame").reset_index(drop=True)

frames = pred2d1["frame"].values  # 1..101、4系統共通

series = {}  # {区間ラベル: {系統名: np.array}}
for (a, b), label in zip(CHAIN, LABELS):
    series[label] = {
        "2D推定": dist_xy(pred2d1, a, b).values,
        "2D正解": dist_xy(gt2d1, a, b).values,
        "3D推定": pred3d1[f"len_{KP_NAMES[a]}_{KP_NAMES[b]}"].values,
        "3D正解": dist_xyz(gt3d1, a, b).values,
    }

# ---------------- 図1: 2D フレーム-長さ (6区間, 推定/正解) ----------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, label in zip(axes.flat, LABELS):
    ax.plot(frames, series[label]["2D推定"], color="tab:red", label="推定")
    ax.plot(frames, series[label]["2D正解"], color="black", linestyle="--", label="正解")
    ax.set_title(label)
    ax.set_xlabel("frame")
    ax.set_ylabel("length [px]")
    ax.grid(True, alpha=0.3)
axes.flat[0].legend(fontsize=9)
fig.suptitle("恒常性の評価: 2D(Camera1) フレーム-長さ", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(BASE_DIR / "spine_lengths_frames_2d.png", dpi=150, bbox_inches="tight")
fig.savefig(BASE_DIR / "spine_lengths_frames_2d.eps", bbox_inches="tight")
plt.close(fig)
print("wrote spine_lengths_frames_2d.png / .eps")

# ---------------- 図2: 3D フレーム-長さ (6区間, 推定/正解) ----------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, label in zip(axes.flat, LABELS):
    ax.plot(frames, series[label]["3D推定"], color="tab:blue", label="推定")
    ax.plot(frames, series[label]["3D正解"], color="black", linestyle="--", label="正解")
    ax.set_title(label)
    ax.set_xlabel("frame")
    ax.set_ylabel("length [normalized]")
    ax.grid(True, alpha=0.3)
axes.flat[0].legend(fontsize=9)
fig.suptitle("恒常性の評価: 3D(Camera1) フレーム-長さ", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(BASE_DIR / "spine_lengths_frames_3d.png", dpi=150, bbox_inches="tight")
fig.savefig(BASE_DIR / "spine_lengths_frames_3d.eps", bbox_inches="tight")
plt.close(fig)
print("wrote spine_lengths_frames_3d.png / .eps")

# ---------------- 図3: 2D・3D重ね書き (区間ごとに2軸) ----------------
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, label in zip(axes.flat, LABELS):
    ax2 = ax.twinx()
    l1, = ax.plot(frames, series[label]["2D推定"], color="tab:red", label="2D推定")
    l2, = ax.plot(frames, series[label]["2D正解"], color="tab:red", linestyle="--", alpha=0.6, label="2D正解")
    l3, = ax2.plot(frames, series[label]["3D推定"], color="tab:blue", label="3D推定")
    l4, = ax2.plot(frames, series[label]["3D正解"], color="tab:blue", linestyle="--", alpha=0.6, label="3D正解")
    ax.set_title(label)
    ax.set_xlabel("frame")
    ax.set_ylabel("2D length [px]", color="tab:red")
    ax2.set_ylabel("3D length [normalized]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, alpha=0.3)
    if label == LABELS[0]:
        ax.legend(handles=[l1, l2, l3, l4], fontsize=8, loc="lower left")
fig.suptitle("恒常性の評価: 2D・3D重ね書き(区間ごとに軸を分離)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(BASE_DIR / "spine_lengths_frames_overlay.png", dpi=150, bbox_inches="tight")
fig.savefig(BASE_DIR / "spine_lengths_frames_overlay.eps", bbox_inches="tight")
plt.close(fig)
print("wrote spine_lengths_frames_overlay.png / .eps")

# ---------------- 表: 平均・標準偏差・CV・フレーム間変化率 ----------------
rows = []
for label in LABELS:
    for src, arr in series[label].items():
        arr = np.asarray(arr, dtype=float)
        mean = arr.mean()
        std = arr.std()
        cv = std / mean
        frame_diff = np.abs(np.diff(arr))
        mean_frame_diff = frame_diff.mean()
        frame_diff_rate = mean_frame_diff / mean * 100
        rows.append({
            "区間": label, "系統": src, "平均": mean, "標準偏差": std, "CV(標準偏差/平均)": cv,
            "平均フレーム間変化量": mean_frame_diff, "平均フレーム間変化率(%)": frame_diff_rate,
        })

stats_df = pd.DataFrame(rows)
csv_path = BASE_DIR / "spine_length_stats.csv"
stats_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"wrote {csv_path}")

# ---------------- 表画像 ----------------
fmt = stats_df.copy()
for c in ["平均", "標準偏差", "CV(標準偏差/平均)", "平均フレーム間変化量", "平均フレーム間変化率(%)"]:
    fmt[c] = fmt[c].map(lambda v: f"{v:.4g}")

header = list(fmt.columns)
cell_text = fmt.values.tolist()
all_rows = [header] + cell_text

n_total_rows = len(all_rows)
n_total_cols = len(header)
fig_w = 1.7 * n_total_cols
fig_h = 0.45 * n_total_rows
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

table = ax.table(cellText=all_rows, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#444444")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(weight="bold")
    if c in (0, 1):
        cell.set_text_props(ha="left")

fig.tight_layout()
png_path = BASE_DIR / "spine_length_stats.png"
eps_path = BASE_DIR / "spine_length_stats.eps"
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(eps_path, bbox_inches="tight")
plt.close(fig)
print(f"wrote {png_path}")
print(f"wrote {eps_path}")
