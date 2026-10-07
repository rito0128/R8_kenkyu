# -*- coding: utf-8 -*-
"""
11. 恒常性の評価の詳細可視化: head-spine0-...-spine5の6区間について、
    フレーム-長さグラフ(2D/3D/重ね書き)と、標準偏差・CV・フレーム間変化率の表を作成する。
    19点拡張後モデル、Camera1・Camera4。

出力: zikkenn2/11/
  spine_lengths_frames_2d_camera{1,4}.png/.eps
  spine_lengths_frames_3d_camera{1,4}.png/.eps
  spine_lengths_frames_overlay_camera{1,4}.png/.eps
  spine_length_stats.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "11"

stat_rows = []
for cam in c.CAMERAS:
    pred2d = c.load_2d_pred("19pt", cam)
    gt2d = c.load_2d_gt(cam)
    pred3d = c.load_3d_pred("19pt", cam)
    gt3d = c.load_3d_gt(cam)
    frames = pred2d["frame"].values

    series = {}
    for (a, b), label in zip(c.SPINE_CHAIN, c.SPINE_CHAIN_LABELS):
        series[label] = {
            "2D推定": c.dist(c.pred_xy(pred2d, a), c.pred_xy(pred2d, b)),
            "2D正解": c.dist(c.gt_xy(gt2d, a), c.gt_xy(gt2d, b)),
            "3D推定": c.dist(c.pred_xyz(pred3d, a), c.pred_xyz(pred3d, b)),
            "3D正解": c.dist(c.gt_xyz(gt3d, a), c.gt_xyz(gt3d, b)),
        }

    # 図1: 2D
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, label in zip(axes.flat, c.SPINE_CHAIN_LABELS):
        ax.plot(frames, series[label]["2D推定"], color="tab:red", label="推定")
        ax.plot(frames, series[label]["2D正解"], color="black", linestyle="--", label="正解")
        ax.set_title(label)
        ax.set_xlabel("frame")
        ax.set_ylabel("length [px]")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle(f"恒常性の評価: 2D(Camera{cam}) フレーム-長さ", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / f"spine_lengths_frames_2d_camera{cam}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"spine_lengths_frames_2d_camera{cam}.eps", bbox_inches="tight")
    plt.close(fig)

    # 図2: 3D
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, label in zip(axes.flat, c.SPINE_CHAIN_LABELS):
        ax.plot(frames, series[label]["3D推定"], color="tab:blue", label="推定")
        ax.plot(frames, series[label]["3D正解"], color="black", linestyle="--", label="正解")
        ax.set_title(label)
        ax.set_xlabel("frame")
        ax.set_ylabel("length [normalized]")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle(f"恒常性の評価: 3D(Camera{cam}) フレーム-長さ", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / f"spine_lengths_frames_3d_camera{cam}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"spine_lengths_frames_3d_camera{cam}.eps", bbox_inches="tight")
    plt.close(fig)

    # 図3: 重ね書き(区間ごとに2軸)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, label in zip(axes.flat, c.SPINE_CHAIN_LABELS):
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
        if label == c.SPINE_CHAIN_LABELS[0]:
            ax.legend(handles=[l1, l2, l3, l4], fontsize=8, loc="lower left")
    fig.suptitle(f"恒常性の評価: 2D・3D重ね書き Camera{cam}(区間ごとに軸を分離)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / f"spine_lengths_frames_overlay_camera{cam}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"spine_lengths_frames_overlay_camera{cam}.eps", bbox_inches="tight")
    plt.close(fig)

    # 統計表
    for label in c.SPINE_CHAIN_LABELS:
        for src, arr in series[label].items():
            arr = np.asarray(arr, dtype=float)
            mean = arr.mean()
            std = arr.std()
            frame_diff = np.abs(np.diff(arr))
            stat_rows.append({
                "camera": cam, "区間": label, "系統": src,
                "平均": mean, "標準偏差": std, "CV(標準偏差/平均)": std / mean,
                "平均フレーム間変化量": frame_diff.mean(),
                "平均フレーム間変化率(%)": frame_diff.mean() / mean * 100,
            })

stats_df = pd.DataFrame(stat_rows)
stats_df.to_csv(OUT / "spine_length_stats.csv", index=False, encoding="utf-8-sig")
print("wrote spine_length_stats.csv / graphs")
print(stats_df.head(12).to_string(index=False))
