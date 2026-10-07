# -*- coding: utf-8 -*-
"""
23. 17点モデル vs 19点モデル: 左右の肩・肘・手首について、frame33〜82の各フレームで
    推定値と正解データの距離(位置誤差)を算出する。Camera1・Camera4、2D・3Dそれぞれ。

    対象キーポイント(6点): left_shoulder, right_shoulder, left_elbow, right_elbow,
    left_wrist, right_wrist (17点・19点モデルの両方に共通して存在する点)

出力: zikkenn2/23/
  keypoint_distance_{17pt,19pt}_camera{1,4}_{2d,3d}.csv   (計8ファイル)
    frame, keypoint, 推定x, 推定y[, 推定z], 正解x, 正解y[, 正解z], distance
  distance_2d.png/.eps   2D: モデル×カメラの4パネル(各パネルに6キーポイント)
  distance_3d.png/.eps   3D: 同上
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

import common as c

OUT = c.BASE_DIR / "23"
FRAME_RANGE = (33, 82)

# 対象6点: 19点(idx) -> 17点2D(idx), 17点3D(idx)
TARGETS = {
    "left_shoulder": (1, 5, 11),
    "right_shoulder": (2, 6, 14),
    "left_elbow": (3, 7, 12),
    "right_elbow": (4, 8, 15),
    "left_wrist": (5, 9, 13),
    "right_wrist": (6, 10, 16),
}

all_series = {}  # (model, cam, dim) -> {keypoint: (frames, distances)}

for cam in c.CAMERAS:
    pred19_2d = c.load_2d_pred("19pt", cam)
    pred17_2d = c.load_2d_pred("17pt", cam)
    gt2d = c.load_2d_gt(cam)
    pred19_3d = c.load_3d_pred("19pt", cam)
    pred17_3d = c.load_3d_pred("17pt", cam)
    gt3d = c.load_3d_gt(cam)

    frames_all = pred19_2d["frame"].values
    mask = (frames_all >= FRAME_RANGE[0]) & (frames_all <= FRAME_RANGE[1])
    frames = frames_all[mask]

    for model in ["17pt", "19pt"]:
        for dim in ["2d", "3d"]:
            rows = []
            series = {}
            for name, (i19, i17_2d, i17_3d) in TARGETS.items():
                if dim == "2d":
                    if model == "19pt":
                        p = c.pred_xy(pred19_2d, i19)[mask]
                    else:
                        p = c.pred_xy(pred17_2d, i17_2d)[mask]
                    g = c.gt_xy(gt2d, i19)[mask]
                else:
                    if model == "19pt":
                        p = c.pred_xyz(pred19_3d, i19)[mask]
                    else:
                        p = c.pred_xyz(pred17_3d, i17_3d)[mask]
                    g = c.gt_xyz(gt3d, i19)[mask]

                dist = c.dist(p, g)
                series[name] = (frames, dist)

                for j, fr in enumerate(frames):
                    row = {"frame": fr, "keypoint": name}
                    if dim == "2d":
                        row["推定x"] = p[j, 0]; row["推定y"] = p[j, 1]
                        row["正解x"] = g[j, 0]; row["正解y"] = g[j, 1]
                    else:
                        row["推定x"] = p[j, 0]; row["推定y"] = p[j, 1]; row["推定z"] = p[j, 2]
                        row["正解x"] = g[j, 0]; row["正解y"] = g[j, 1]; row["正解z"] = g[j, 2]
                    row["distance"] = dist[j]
                    rows.append(row)

            df = pd.DataFrame(rows)
            out_path = OUT / f"keypoint_distance_{model}_camera{cam}_{dim}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"wrote {out_path.name}")

            all_series[(model, cam, dim)] = series

# ---------------- グラフ(次元ごとに1枚、モデル×カメラの4パネル) ----------------
MODELS = ["17pt", "19pt"]
CAMS = c.CAMERAS
cmap = plt.get_cmap("tab10")

for dim, ylabel, unit in [("2d", "distance [px]", "2D"), ("3d", "distance [normalized]", "3D")]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for r, model in enumerate(MODELS):
        for cidx, cam in enumerate(CAMS):
            ax = axes[r, cidx]
            series = all_series[(model, cam, dim)]
            for i, (name, (frames, dist)) in enumerate(series.items()):
                ax.plot(frames, dist, color=cmap(i), label=name, linewidth=1.4)
            ax.set_title(f"{model} / Camera{cam}", fontsize=12)
            ax.set_xlabel("frame")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if r == 0 and cidx == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"{unit}: 推定→正解の距離 (frame{FRAME_RANGE[0]}-{FRAME_RANGE[1]})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / f"distance_{dim}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"distance_{dim}.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote distance_{dim}.png / .eps")
