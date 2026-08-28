# -*- coding: utf-8 -*-
"""
「首から腰にかけての長さ」をフレームごとに計算し比較する。

対象(すべてCamera1):
  17点モデル(拡張前, for_test_keypoint_17_motion_007)
    - 2D: COCO17形式。首・腰のキーポイントが無いため、
          首 ≈ 左右肩の中点、腰 ≈ 左右股関節の中点 として概算
    - 3D: H36M17形式。首 = Neck/Nose(idx9)、腰 = Hip/root(idx0) をそのまま使用
  19点モデル(拡張後, for_test_HRNet / for_test_motionbert)
    - 首 ≈ spine5(idx18)、腰 ≈ spine0(idx13) の直線距離
  正解データ(correct_2d_annotation.csv / correct_3d_anotation.csv)
    - 同様に J13(spine0) - J18(spine5) の直線距離

出力:
  neck_waist_length.csv                 フレームごとの値(ロング形式)
  neck_waist_length_17kp.png/.eps       17点モデル単体のグラフ(2D/3D)
  neck_waist_length_comparison.png/.eps 17点/19点/正解 比較グラフ(2D/3D)
"""

import json
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
K17_DIR = BASE_DIR / "for_test_keypoint_17_motion_007"


def dist(p, q):
    return np.linalg.norm(np.asarray(p) - np.asarray(q))


# ---------------- 17点モデル(拡張前) ----------------
def load_17pt_2d():
    data = json.load(open(K17_DIR / "pose2d" / "predictions" / "motion_007_Camera1.json", encoding="utf-8"))
    rows = []
    for entry in data:
        kp = entry["instances"][0]["keypoints"]
        shoulder_mid = np.mean([kp[5], kp[6]], axis=0)
        hip_mid = np.mean([kp[11], kp[12]], axis=0)
        rows.append({"frame": entry["frame_id"] + 1, "length": dist(shoulder_mid, hip_mid)})
    return pd.DataFrame(rows)


def load_17pt_3d():
    data = json.load(open(K17_DIR / "predictions" / "motion_007_Camera1.json", encoding="utf-8"))
    rows = []
    for entry in data:
        kp = entry["instances"][0]["keypoints"]
        neck = kp[9]   # Neck/Nose
        hip = kp[0]    # Hip(root)
        rows.append({"frame": entry["frame_id"] + 1, "length": dist(neck, hip)})
    return pd.DataFrame(rows)


k17_2d = load_17pt_2d()
k17_3d = load_17pt_3d()

# ---------------- 19点モデル(拡張後)・正解データ ----------------
pred2d = pd.read_csv(HR_DIR / "keypoint_lengths.csv")
pred2d1 = pred2d[pred2d.camera == 1].sort_values("frame")
ext2d = pd.DataFrame({
    "frame": pred2d1["frame"].values,
    "length": np.linalg.norm(
        pred2d1[["kp13_spine0_x", "kp13_spine0_y"]].values - pred2d1[["kp18_spine5_x", "kp18_spine5_y"]].values,
        axis=1,
    ),
})

gt2d = pd.read_csv(HR_DIR / "correct_keypoint_lengths.csv")
gt2d1 = gt2d[gt2d.camera == 1].sort_values("frame")
gt_2d = pd.DataFrame({
    "frame": gt2d1["frame"].values,
    "length": np.linalg.norm(
        gt2d1[["kp13_spine0_x", "kp13_spine0_y"]].values - gt2d1[["kp18_spine5_x", "kp18_spine5_y"]].values,
        axis=1,
    ),
})

pred3d = pd.read_csv(MB_DIR / "motionbert_keypoint_lengths.csv")
pred3d1 = pred3d[pred3d.camera == 1].sort_values("frame")
ext3d = pd.DataFrame({
    "frame": pred3d1["frame"].values,
    "length": np.linalg.norm(
        pred3d1[["kp13_spine0_x", "kp13_spine0_y", "kp13_spine0_z"]].values
        - pred3d1[["kp18_spine5_x", "kp18_spine5_y", "kp18_spine5_z"]].values,
        axis=1,
    ),
})

gt3d = pd.read_csv(MB_DIR / "correct_3d_anotation.csv")
gt3d1 = gt3d[gt3d["filename"].str.startswith("out_Camera1_")].copy()
gt3d1["frame"] = gt3d1["filename"].str.extract(r"_(\d+)$").astype(int)
gt3d1 = gt3d1.sort_values("frame")
gt_3d = pd.DataFrame({
    "frame": gt3d1["frame"].values,
    "length": np.linalg.norm(
        gt3d1[["J13_x", "J13_y", "J13_z"]].values - gt3d1[["J18_x", "J18_y", "J18_z"]].values, axis=1
    ),
})

# ---------------- 生データ保存(ロング形式) ----------------
long_rows = []
for source, label, df in [
    ("2D", "17点モデル(拡張前)", k17_2d), ("2D", "19点モデル(拡張後)", ext2d), ("2D", "正解データ", gt_2d),
    ("3D", "17点モデル(拡張前)", k17_3d), ("3D", "19点モデル(拡張後)", ext3d), ("3D", "正解データ", gt_3d),
]:
    for frame, length in zip(df["frame"], df["length"]):
        long_rows.append({"source": source, "label": label, "frame": frame, "length": length})

long_df = pd.DataFrame(long_rows)
csv_path = BASE_DIR / "neck_waist_length.csv"
long_df.to_csv(csv_path, index=False)
print(f"wrote {csv_path}")

# ---------------- グラフ1: 17点モデル単体 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(k17_2d["frame"], k17_2d["length"], color="tab:blue")
axes[0].set_title("2D (肩中点-股関節中点で概算)")
axes[0].set_xlabel("frame")
axes[0].set_ylabel("length [px]")
axes[0].grid(True, alpha=0.3)

axes[1].plot(k17_3d["frame"], k17_3d["length"], color="tab:orange")
axes[1].set_title("3D (Neck/Nose - Hip)")
axes[1].set_xlabel("frame")
axes[1].set_ylabel("length [normalized]")
axes[1].grid(True, alpha=0.3)

fig.suptitle("17点モデル(拡張前, Camera1): 首-腰の長さ", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(BASE_DIR / "neck_waist_length_17kp.png", dpi=150, bbox_inches="tight")
fig.savefig(BASE_DIR / "neck_waist_length_17kp.eps", bbox_inches="tight")
plt.close(fig)
print("wrote neck_waist_length_17kp.png / .eps")

# ---------------- グラフ2: 17点/19点/正解 比較 ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k17_2d["frame"], k17_2d["length"], label="17点モデル(拡張前, 概算)", color="tab:blue")
axes[0].plot(ext2d["frame"], ext2d["length"], label="19点モデル(拡張後)", color="tab:red")
axes[0].plot(gt_2d["frame"], gt_2d["length"], label="正解データ", color="black", linestyle="--")
axes[0].set_title("2D (単位: px, 共通)")
axes[0].set_xlabel("frame")
axes[0].set_ylabel("length [px]")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=9)

axes[1].plot(k17_3d["frame"], k17_3d["length"], label="17点モデル(拡張前)", color="tab:blue")
axes[1].plot(ext3d["frame"], ext3d["length"], label="19点モデル(拡張後)", color="tab:red")
axes[1].plot(gt_3d["frame"], gt_3d["length"], label="正解データ", color="black", linestyle="--")
axes[1].set_title("3D (単位が異なるため参考値)")
axes[1].set_xlabel("frame")
axes[1].set_ylabel("length (正規化/実寸が混在)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9)

fig.suptitle("首-腰の長さ比較 (Camera1): 17点モデル vs 19点モデル vs 正解", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(BASE_DIR / "neck_waist_length_comparison.png", dpi=150, bbox_inches="tight")
fig.savefig(BASE_DIR / "neck_waist_length_comparison.eps", bbox_inches="tight")
plt.close(fig)
print("wrote neck_waist_length_comparison.png / .eps")
