# -*- coding: utf-8 -*-
"""
7. 17点(拡張前)モデル vs 19点(拡張後)モデル vs 正解データ: 首-腰間の長さをフレームごとに比較。
   17点モデルの2Dは肩中点-股関節中点(COCO17)、3DはH36M17のNeck/Nose-Hip(root)で近似。
   19点モデル・正解データはspine0(idx13)-spine5(idx18)間の直線距離。
   Camera1・Camera4。

出力: zikkenn2/7/
  neck_waist_length.csv               フレームごとの生データ(ロング形式)
  neck_waist_length_comparison_camera{1,4}.png/.eps  17点/19点/正解の比較グラフ(2D/3D)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "7"

# 17点モデルのインデックス
C17_2D_LSHOULDER, C17_2D_RSHOULDER = 5, 6
C17_2D_LHIP, C17_2D_RHIP = 11, 12
C17_3D_NECK, C17_3D_HIP = 9, 0

long_rows = []
for cam in c.CAMERAS:
    k17_2d = c.load_2d_pred("17pt", cam)
    k17_3d = c.load_3d_pred("17pt", cam)
    pred2d = c.load_2d_pred("19pt", cam)
    gt2d = c.load_2d_gt(cam)
    pred3d = c.load_3d_pred("19pt", cam)
    gt3d = c.load_3d_gt(cam)
    frames = k17_2d["frame"].values

    shoulder_mid = (c.pred_xy(k17_2d, C17_2D_LSHOULDER) + c.pred_xy(k17_2d, C17_2D_RSHOULDER)) / 2
    hip_mid = (c.pred_xy(k17_2d, C17_2D_LHIP) + c.pred_xy(k17_2d, C17_2D_RHIP)) / 2
    k17_2d_len = c.dist(shoulder_mid, hip_mid)

    k17_3d_len = c.dist(c.pred_xyz(k17_3d, C17_3D_NECK), c.pred_xyz(k17_3d, C17_3D_HIP))

    ext2d_len = c.dist(c.pred_xy(pred2d, 13), c.pred_xy(pred2d, 18))
    gt2d_len = c.dist(c.gt_xy(gt2d, 13), c.gt_xy(gt2d, 18))
    ext3d_len = c.dist(c.pred_xyz(pred3d, 13), c.pred_xyz(pred3d, 18))
    gt3d_len = c.dist(c.gt_xyz(gt3d, 13), c.gt_xyz(gt3d, 18))

    for source, label, arr in [
        ("2D", "17点モデル(拡張前)", k17_2d_len), ("2D", "19点モデル(拡張後)", ext2d_len), ("2D", "正解データ", gt2d_len),
        ("3D", "17点モデル(拡張前)", k17_3d_len), ("3D", "19点モデル(拡張後)", ext3d_len), ("3D", "正解データ", gt3d_len),
    ]:
        for fr, v in zip(frames, arr):
            long_rows.append({"camera": cam, "source": source, "label": label, "frame": fr, "length": v})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(frames, k17_2d_len, label="17点モデル(拡張前, 概算)", color="tab:blue")
    axes[0].plot(frames, ext2d_len, label="19点モデル(拡張後)", color="tab:red")
    axes[0].plot(frames, gt2d_len, label="正解データ", color="black", linestyle="--")
    axes[0].set_title("2D (単位: px, 共通)")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("length [px]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(frames, k17_3d_len, label="17点モデル(拡張前)", color="tab:blue")
    axes[1].plot(frames, ext3d_len, label="19点モデル(拡張後)", color="tab:red")
    axes[1].plot(frames, gt3d_len, label="正解データ", color="black", linestyle="--")
    axes[1].set_title("3D (単位が異なるため参考値)")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("length (正規化/実寸が混在)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle(f"首-腰の長さ比較 (Camera{cam}): 17点モデル vs 19点モデル vs 正解", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / f"neck_waist_length_comparison_camera{cam}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"neck_waist_length_comparison_camera{cam}.eps", bbox_inches="tight")
    plt.close(fig)

pd.DataFrame(long_rows).to_csv(OUT / "neck_waist_length.csv", index=False, encoding="utf-8-sig")
print("wrote neck_waist_length.csv / graphs")
