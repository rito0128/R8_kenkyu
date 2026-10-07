# -*- coding: utf-8 -*-
"""
19番の続き: spine0・spine3だけでなく、背中(脊柱)の全関節点(spine0〜spine4の5関節)について
散布図・フレーム-角度推移グラフを作成する。18番・19番と同じCamera4, frame270-386, 19点モデル。

出力: zikkenn2/19/
  scatter_all_spine.png/.eps      spine0〜spine4 全5関節の散布図(2D・3D、5行×2列)
  timeseries_all_spine.png/.eps   spine0〜spine4 全5関節のフレーム-角度推移(2D・3D、5行×2列)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

IN18 = c.BASE_DIR / "18"
OUT = c.BASE_DIR / "19"

frames_df = pd.read_csv(IN18 / "flexibility_axes_frames.csv")

SPINE_JOINTS = ["spine0", "spine1", "spine2", "spine3", "spine4"]

# ---------------- 散布図: spine0〜spine4 全関節 ----------------
fig, axes = plt.subplots(len(SPINE_JOINTS), 2, figsize=(11, 4.2 * len(SPINE_JOINTS)))
for row_i, joint in enumerate(SPINE_JOINTS):
    for col, dim in enumerate(["2D", "3D"]):
        ax = axes[row_i, col]
        sub = frames_df[frames_df["関節"] == joint]
        x = sub[f"{dim}正解"].values
        y = sub[f"{dim}推定"].values
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=16, alpha=0.6, color="tab:blue")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y=x(完全一致)")
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(lo, hi, 50)
        ax.plot(xs, np.polyval(coeffs, xs), color="tab:red", linewidth=1.5, label="回帰直線")
        ax.set_xlabel("正解角度 [deg]")
        ax.set_ylabel("推定角度 [deg]")
        sign = "正" if r >= 0 else "逆"
        ax.set_title(f"{joint} ({dim})  R={r:.3f} ({sign}相関)", fontsize=11)
        ax.grid(True, alpha=0.3)
        if row_i == 0 and col == 0:
            ax.legend(fontsize=8)
fig.suptitle("背中(脊柱)全関節: 推定角度 vs 正解角度の散布図\nCamera4, frame270-386, 19点モデル", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "scatter_all_spine.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "scatter_all_spine.eps", bbox_inches="tight")
plt.close(fig)
print("wrote scatter_all_spine.png / .eps")

# ---------------- フレーム-角度推移: spine0〜spine4 全関節 ----------------
fig, axes = plt.subplots(len(SPINE_JOINTS), 2, figsize=(14, 3.2 * len(SPINE_JOINTS)))
for row_i, joint in enumerate(SPINE_JOINTS):
    for col, dim in enumerate(["2D", "3D"]):
        ax = axes[row_i, col]
        sub = frames_df[frames_df["関節"] == joint].sort_values("frame")
        r = np.corrcoef(sub[f"{dim}推定"], sub[f"{dim}正解"])[0, 1]
        ax.plot(sub["frame"], sub[f"{dim}推定"], color="tab:red", label="推定")
        ax.plot(sub["frame"], sub[f"{dim}正解"], color="black", linestyle="--", label="正解")
        ax.set_xlabel("frame")
        ax.set_ylabel("angle [deg]")
        sign = "正" if r >= 0 else "逆"
        ax.set_title(f"{joint} ({dim})  相関 R={r:.3f} ({sign}相関)", fontsize=11)
        ax.grid(True, alpha=0.3)
        if row_i == 0 and col == 0:
            ax.legend(fontsize=8)
fig.suptitle("背中(脊柱)全関節: フレーム-角度の推移\nCamera4, frame270-386, 19点モデル", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "timeseries_all_spine.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "timeseries_all_spine.eps", bbox_inches="tight")
plt.close(fig)
print("wrote timeseries_all_spine.png / .eps")
