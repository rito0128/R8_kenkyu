# -*- coding: utf-8 -*-
"""
1. 2D推定(HRNet, 19点拡張後モデル)の精度評価
   Camera1・Camera4について、正解2D座標と推定結果を比較し、
   関節ごとの位置誤差(MPJPE-2D)・キーポイント間距離を算出する。

出力: zikkenn2/1/
  keypoint_errors.csv       フレーム・カメラ・キーポイントごとの位置誤差(px)
  mpjpe_summary.csv         カメラ別・キーポイント別MPJPE要約
  mpjpe_by_camera.png/.eps  カメラ別MPJPE棒グラフ
  mpjpe_by_keypoint.png/.eps キーポイント別MPJPE棒グラフ
  error_frames_2d.png/.eps  カメラごとのフレーム-誤差推移
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from pathlib import Path

import common as c

OUT = c.BASE_DIR / "1"

rows = []
for cam in c.CAMERAS:
    pred = c.load_2d_pred("19pt", cam)
    gt = c.load_2d_gt(cam)
    assert len(pred) == len(gt) == c.N_FRAMES
    for i, name in enumerate(c.KP19_NAMES):
        err = c.dist(c.pred_xy(pred, i), c.gt_xy(gt, i))
        for frame, e in zip(pred["frame"], err):
            rows.append({"camera": cam, "frame": frame, "keypoint": name, "kp_idx": i, "error_px": e})

err_df = pd.DataFrame(rows)
err_df.to_csv(OUT / "keypoint_errors.csv", index=False, encoding="utf-8-sig")
print("wrote keypoint_errors.csv")

summary_cam = err_df.groupby("camera")["error_px"].agg(mean="mean", median="median", std="std").reset_index()
summary_kp = err_df.groupby(["keypoint", "kp_idx"])["error_px"].agg(mean="mean", median="median", std="std").reset_index().sort_values("kp_idx")
overall_mean = err_df["error_px"].mean()

with open(OUT / "mpjpe_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    f.write(f"# 全体MPJPE(px): {overall_mean:.4g}\n\n")
    f.write("# カメラ別\n")
    summary_cam.to_csv(f, index=False)
    f.write("\n# キーポイント別\n")
    summary_kp.to_csv(f, index=False)
print("wrote mpjpe_summary.csv")
print(f"overall MPJPE-2D: {overall_mean:.4g} px")
print(summary_cam)

# ---- グラフ: カメラ別MPJPE ----
fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(summary_cam["camera"].astype(str), summary_cam["mean"], color="tab:red")
ax.set_xlabel("camera")
ax.set_ylabel("MPJPE-2D [px]")
ax.set_title("カメラ別 MPJPE-2D")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "mpjpe_by_camera.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "mpjpe_by_camera.eps", bbox_inches="tight")
plt.close(fig)

# ---- グラフ: キーポイント別MPJPE ----
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(summary_kp["keypoint"], summary_kp["mean"], color="tab:red")
ax.set_ylabel("MPJPE-2D [px]")
ax.set_title("キーポイント別 MPJPE-2D (Camera1・4平均)")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "mpjpe_by_keypoint.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "mpjpe_by_keypoint.eps", bbox_inches="tight")
plt.close(fig)

# ---- グラフ: フレーム-誤差(カメラ別、全キーポイント平均) ----
fig, ax = plt.subplots(figsize=(9, 4.5))
for cam, color in zip(c.CAMERAS, ["tab:red", "tab:blue"]):
    sub = err_df[err_df.camera == cam].groupby("frame")["error_px"].mean()
    ax.plot(sub.index, sub.values, label=f"Camera{cam}", color=color)
ax.set_xlabel("frame")
ax.set_ylabel("平均誤差 [px]")
ax.set_title("2D推定誤差のフレーム推移(全キーポイント平均)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "error_frames_2d.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "error_frames_2d.eps", bbox_inches="tight")
plt.close(fig)
print("wrote graphs")
