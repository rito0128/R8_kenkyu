# -*- coding: utf-8 -*-
"""
6. 柔軟性評価(拡張版): spine0〜spine4(脊柱側5関節)+左右肘・膝(四肢側4関節)、計9関節の
   3点角度を算出。19点拡張後モデル、Camera1・Camera4。
   spine0〜spine4は隣接2点から角度を計算(両端のhead・spine5は腕として使用)。

出力: zikkenn2/6/
  flexibility_9joint_summary.csv    関節×カメラごとの可動範囲・相関・MAE
  flexibility_9joint_frames.csv     フレームごとの角度生データ
  flexibility_angles_camera{1,4}.png/.eps  フレーム-角度グラフ(脊柱/四肢の2パネル)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "6"

# 3点角度の(始点, 中心, 終点)インデックス。spine系はKP19_NAMES順で隣接点を使用。
JOINTS = {
    "spine0": (0, 13, 14),
    "spine1": (13, 14, 15),
    "spine2": (14, 15, 16),
    "spine3": (15, 16, 17),
    "spine4": (16, 17, 18),
    "left_elbow": (1, 3, 5),
    "right_elbow": (2, 4, 6),
    "left_knee": (7, 9, 11),
    "right_knee": (8, 10, 12),
}
SPINE_JOINTS = ["spine0", "spine1", "spine2", "spine3", "spine4"]
LIMB_JOINTS = ["left_elbow", "right_elbow", "left_knee", "right_knee"]


def angle3pt(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos = (v1 * v2).sum(axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


summary_rows = []
frame_rows = []
for cam in c.CAMERAS:
    pred2d = c.load_2d_pred("19pt", cam)
    gt2d = c.load_2d_gt(cam)
    pred3d = c.load_3d_pred("19pt", cam)
    gt3d = c.load_3d_gt(cam)
    frames = pred2d["frame"].values

    angles = {}
    for name, (a, b, cidx) in JOINTS.items():
        ang_pred2d = angle3pt(c.pred_xy(pred2d, a), c.pred_xy(pred2d, b), c.pred_xy(pred2d, cidx))
        ang_gt2d = angle3pt(c.gt_xy(gt2d, a), c.gt_xy(gt2d, b), c.gt_xy(gt2d, cidx))
        ang_pred3d = angle3pt(c.pred_xyz(pred3d, a), c.pred_xyz(pred3d, b), c.pred_xyz(pred3d, cidx))
        ang_gt3d = angle3pt(c.gt_xyz(gt3d, a), c.gt_xyz(gt3d, b), c.gt_xyz(gt3d, cidx))
        angles[name] = (ang_pred2d, ang_gt2d, ang_pred3d, ang_gt3d)

        summary_rows.append({
            "camera": cam, "関節": name,
            "2D推定範囲": f"{ang_pred2d.min():.4g}°〜{ang_pred2d.max():.4g}°",
            "2D正解範囲": f"{ang_gt2d.min():.4g}°〜{ang_gt2d.max():.4g}°",
            "2D相関": c.sig4(np.corrcoef(ang_pred2d, ang_gt2d)[0, 1]),
            "2D_MAE": c.sig4(np.abs(ang_pred2d - ang_gt2d).mean()),
            "3D推定範囲": f"{ang_pred3d.min():.4g}°〜{ang_pred3d.max():.4g}°",
            "3D正解範囲": f"{ang_gt3d.min():.4g}°〜{ang_gt3d.max():.4g}°",
            "3D相関": c.sig4(np.corrcoef(ang_pred3d, ang_gt3d)[0, 1]),
            "3D_MAE": c.sig4(np.abs(ang_pred3d - ang_gt3d).mean()),
        })
        for fr, p2, g2, p3, g3 in zip(frames, ang_pred2d, ang_gt2d, ang_pred3d, ang_gt3d):
            frame_rows.append({"camera": cam, "frame": fr, "関節": name,
                                "2D推定": p2, "2D正解": g2, "3D推定": p3, "3D正解": g3})

    # ---- グラフ: 脊柱/四肢の2パネル(2D=実線, 3D=別図にするとcamera単位で見づらいため
    #      2D・3Dそれぞれ1枚ずつ、脊柱/四肢2パネル構成) ----
    for dim, pred_idx, gt_idx in [("2d", 0, 1), ("3d", 2, 3)]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        cmap = plt.get_cmap("tab10")
        for i, name in enumerate(SPINE_JOINTS):
            axes[0].plot(frames, angles[name][pred_idx], color=cmap(i), label=f"{name} 推定")
            axes[0].plot(frames, angles[name][gt_idx], color=cmap(i), linestyle="--", alpha=0.7, label=f"{name} 正解")
        axes[0].set_title("脊柱(spine0〜spine4)")
        axes[0].set_xlabel("frame")
        axes[0].set_ylabel("angle [deg]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2)

        for i, name in enumerate(LIMB_JOINTS):
            axes[1].plot(frames, angles[name][pred_idx], color=cmap(i), label=f"{name} 推定")
            axes[1].plot(frames, angles[name][gt_idx], color=cmap(i), linestyle="--", alpha=0.7, label=f"{name} 正解")
        axes[1].set_title("四肢(肘・膝)")
        axes[1].set_xlabel("frame")
        axes[1].set_ylabel("angle [deg]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=7, ncol=2)

        fig.suptitle(f"{dim.upper()} (Camera{cam}): フレーム-関節角度", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT / f"flexibility_angles_{dim}_camera{cam}.png", dpi=150, bbox_inches="tight")
        fig.savefig(OUT / f"flexibility_angles_{dim}_camera{cam}.eps", bbox_inches="tight")
        plt.close(fig)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "flexibility_9joint_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(frame_rows).to_csv(OUT / "flexibility_9joint_frames.csv", index=False, encoding="utf-8-sig")
print("wrote flexibility_9joint_summary.csv / flexibility_9joint_frames.csv / graphs")
print(summary_df.to_string(index=False))
