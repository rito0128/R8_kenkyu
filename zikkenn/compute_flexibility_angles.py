# -*- coding: utf-8 -*-
"""
柔軟性評価(角度ベース): Camera1のデータ(2D/3D、推定/正解)について、
脊柱の内部関節(spine0〜spine4; 両端のhead・spine5を腕として使用)および
左右の肘・膝の3点関節角度をフレームごとに算出する。

出力:
  flexibility_angles_frames.csv    フレームごとの角度(ロング形式、生データ)
  flexibility_angles_summary.csv   関節ごとの可動範囲・相関・MAEの要約
  flexibility_angles_summary.png/.eps  上記の表画像
  flexibility_angles_2d_camera1.png/.eps  2D: フレーム-角度グラフ(脊柱/四肢)
  flexibility_angles_3d_camera1.png/.eps  3D: フレーム-角度グラフ(脊柱/四肢)
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

KP_NAMES = ['head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
            'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle',
            'spine0','spine1','spine2','spine3','spine4','spine5']

# joint: (joint_idx, neighbor_a_idx, neighbor_b_idx)
JOINTS = [
    ("spine0", 13, 0, 14),
    ("spine1", 14, 13, 15),
    ("spine2", 15, 14, 16),
    ("spine3", 16, 15, 17),
    ("spine4", 17, 16, 18),
    ("left_elbow", 3, 1, 5),
    ("right_elbow", 4, 2, 6),
    ("left_knee", 9, 7, 11),
    ("right_knee", 10, 8, 12),
]
SPINE_JOINTS = [j[0] for j in JOINTS if j[0].startswith("spine")]
LIMB_JOINTS = [j[0] for j in JOINTS if not j[0].startswith("spine")]


def angle_deg(p_joint, p_a, p_b):
    v1 = p_a - p_joint
    v2 = p_b - p_joint
    cos = (v1 * v2).sum(axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def pts_pred(df, idx, use_z):
    name = KP_NAMES[idx]
    cols = [f"kp{idx}_{name}_x", f"kp{idx}_{name}_y"]
    if use_z:
        cols.append(f"kp{idx}_{name}_z")
    return df[cols].values


def pts_gt2d(df, idx):
    name = KP_NAMES[idx]
    return df[[f"kp{idx}_{name}_x", f"kp{idx}_{name}_y"]].values


def pts_gt3d(df, idx):
    return df[[f"J{idx}_x", f"J{idx}_y", f"J{idx}_z"]].values


def compute_angles(df, point_fn):
    out = {"frame": df["frame"].values}
    for name, j, a, b in JOINTS:
        out[name] = angle_deg(point_fn(df, j), point_fn(df, a), point_fn(df, b))
    return pd.DataFrame(out)


# ---------------- データ読み込み ----------------
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

ang_2d_pred = compute_angles(pred2d1, lambda df, idx: pts_pred(df, idx, use_z=False))
ang_2d_gt = compute_angles(gt2d1, lambda df, idx: pts_gt2d(df, idx))
ang_3d_pred = compute_angles(pred3d1, lambda df, idx: pts_pred(df, idx, use_z=True))
ang_3d_gt = compute_angles(gt3d1, lambda df, idx: pts_gt3d(df, idx))

# ---------------- フレームごとの生データ(ロング形式) ----------------
frames_long = []
for source, kind, ang_df in [
    ("2D", "推定", ang_2d_pred), ("2D", "正解", ang_2d_gt),
    ("3D", "推定", ang_3d_pred), ("3D", "正解", ang_3d_gt),
]:
    for name, *_ in JOINTS:
        for frame, val in zip(ang_df["frame"], ang_df[name]):
            frames_long.append({"source": source, "kind": kind, "joint": name, "frame": frame, "angle_deg": val})

frames_df = pd.DataFrame(frames_long)
frames_csv = BASE_DIR / "flexibility_angles_frames.csv"
frames_df.to_csv(frames_csv, index=False)
print(f"wrote {frames_csv}")

# ---------------- 要約表 ----------------
def summarize(pred_df, gt_df):
    m = pred_df.merge(gt_df, on="frame", suffixes=("_pred", "_gt"))
    rows = {}
    for name, *_ in JOINTS:
        p = m[f"{name}_pred"]
        g = m[f"{name}_gt"]
        rows[name] = {
            "推定範囲": f"{p.min():.1f}°〜{p.max():.1f}°",
            "正解範囲": f"{g.min():.1f}°〜{g.max():.1f}°",
            "相関": round(p.corr(g), 2),
            "MAE": round((p - g).abs().mean(), 1),
        }
    return rows

sum_2d = summarize(ang_2d_pred, ang_2d_gt)
sum_3d = summarize(ang_3d_pred, ang_3d_gt)

joint_order = [j[0] for j in JOINTS]
summary_df = pd.DataFrame({
    "2D 推定範囲": [sum_2d[j]["推定範囲"] for j in joint_order],
    "2D 正解範囲": [sum_2d[j]["正解範囲"] for j in joint_order],
    "2D 相関": [sum_2d[j]["相関"] for j in joint_order],
    "2D MAE(°)": [sum_2d[j]["MAE"] for j in joint_order],
    "3D 推定範囲": [sum_3d[j]["推定範囲"] for j in joint_order],
    "3D 正解範囲": [sum_3d[j]["正解範囲"] for j in joint_order],
    "3D 相関": [sum_3d[j]["相関"] for j in joint_order],
    "3D MAE(°)": [sum_3d[j]["MAE"] for j in joint_order],
}, index=joint_order)
summary_df.index.name = "関節"

summary_csv = BASE_DIR / "flexibility_angles_summary.csv"
summary_df.to_csv(summary_csv)
print(f"wrote {summary_csv}")
print(summary_df)

# ---------------- 要約表 画像 ----------------
header = ["関節"] + list(summary_df.columns)
cell_text = [[j] + [str(v) for v in row] for j, row in zip(summary_df.index, summary_df.values)]
all_rows = [header] + cell_text

n_r, n_c = len(all_rows), len(header)
fig, ax = plt.subplots(figsize=(1.5 * n_c, 0.55 * n_r))
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
    if c == 0:
        cell.set_text_props(ha="left")
fig.tight_layout()
fig.savefig(BASE_DIR / "flexibility_angles_summary.png", dpi=200, bbox_inches="tight")
fig.savefig(BASE_DIR / "flexibility_angles_summary.eps", bbox_inches="tight")
plt.close(fig)
print("wrote flexibility_angles_summary.png / .eps")

# ---------------- グラフ: フレーム-角度(2Dと3Dそれぞれ、脊柱/四肢の2段) ----------------
COLORS = plt.cm.tab10.colors

def plot_group(ax, pred_df, gt_df, joints, title):
    for i, name in enumerate(joints):
        c = COLORS[i % len(COLORS)]
        ax.plot(pred_df["frame"], pred_df[name], color=c, linestyle="-", linewidth=1.3, label=f"{name} 推定")
        ax.plot(gt_df["frame"], gt_df[name], color=c, linestyle="--", linewidth=1.3, label=f"{name} 正解")
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel("angle [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def make_figure(pred_df, gt_df, out_stem, suptitle):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_group(axes[0], pred_df, gt_df, SPINE_JOINTS, "脊柱(spine0〜spine4)")
    plot_group(axes[1], pred_df, gt_df, LIMB_JOINTS, "四肢(肘・膝)")
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(BASE_DIR / f"{out_stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(BASE_DIR / f"{out_stem}.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_stem}.png / .eps")


make_figure(ang_2d_pred, ang_2d_gt, "flexibility_angles_2d_camera1", "2D (Camera1): フレーム-関節角度")
make_figure(ang_3d_pred, ang_3d_gt, "flexibility_angles_3d_camera1", "3D (Camera1): フレーム-関節角度")
