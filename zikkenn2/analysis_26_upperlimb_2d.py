# -*- coding: utf-8 -*-
"""
26. 24番と同じ観点(方法A・方法B)・同じフレーム範囲(33-82)・同じ対象(左右肩・肘・手首)を、
    2D推定結果で行う。17点モデル vs 19点モデル。

    2Dはpx単位で両モデル共通のため、24番(3D)のような正規化スケールの違いという問題は
    本来無いが、比較の枠組みを揃えるため同じ2手法を適用する。
    方法A: 骨長基準の相対誤差 (誤差 / 肩-腰長、2D・同一モデル・同一フレーム内)
    方法B: スケール補正(最小二乗)。12点(肩・肘・手首・股関節・膝・足首)から
           最適スケールsを求め、推定にsを掛けてから誤差を算出。

出力: zikkenn2/26/
  method_a_relative_error.csv
  method_b_scaled_distance.csv
  comparison_method_a.png/.eps
  comparison_method_b.png/.eps
  favorable_frame_check.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

import common as c

OUT = c.BASE_DIR / "26"
FRAME_RANGE = (33, 82)

TARGETS = {
    "left_shoulder": (1, 5), "right_shoulder": (2, 6),
    "left_elbow": (3, 7), "right_elbow": (4, 8),
    "left_wrist": (5, 9), "right_wrist": (6, 10),
}
# 19pt idx -> 17pt(COCO17) idx、12点共通対応
COMMON12 = {1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11, 8: 12, 9: 13, 10: 14, 11: 15, 12: 16}

a_rows, b_rows = [], []

for cam in c.CAMERAS:
    pred19 = c.load_2d_pred("19pt", cam)
    pred17 = c.load_2d_pred("17pt", cam)
    gt = c.load_2d_gt(cam)

    frames_all = pred19["frame"].values
    mask = (frames_all >= FRAME_RANGE[0]) & (frames_all <= FRAME_RANGE[1])
    frames = frames_all[mask]

    for model, pred, idx_map in [("19pt", pred19, {k: k for k in COMMON12}), ("17pt", pred17, COMMON12)]:
        P = np.stack([c.pred_xy(pred, idx_map[i19])[mask] for i19 in COMMON12], axis=1)
        G = np.stack([c.gt_xy(gt, i19)[mask] for i19 in COMMON12], axis=1)
        s = (P * G).sum(axis=(1, 2)) / (P * P).sum(axis=(1, 2))

        if model == "19pt":
            shoulder_mid = (c.pred_xy(pred, 1)[mask] + c.pred_xy(pred, 2)[mask]) / 2
            hip_mid = (c.pred_xy(pred, 7)[mask] + c.pred_xy(pred, 8)[mask]) / 2
        else:
            shoulder_mid = (c.pred_xy(pred, 5)[mask] + c.pred_xy(pred, 6)[mask]) / 2
            hip_mid = (c.pred_xy(pred, 11)[mask] + c.pred_xy(pred, 12)[mask]) / 2
        ref_length = c.dist(shoulder_mid, hip_mid)

        for name, (i19, i17) in TARGETS.items():
            idx = i19 if model == "19pt" else i17
            p = c.pred_xy(pred, idx)[mask]
            g = c.gt_xy(gt, i19)[mask]
            raw_dist = c.dist(p, g)
            rel_err = raw_dist / ref_length
            scaled_dist = c.dist(p * s[:, None], g)

            for j, fr in enumerate(frames):
                a_rows.append({"frame": fr, "model": model, "camera": cam, "keypoint": name,
                                "raw_distance": raw_dist[j], "reference_length": ref_length[j],
                                "relative_error": rel_err[j]})
                b_rows.append({"frame": fr, "model": model, "camera": cam, "keypoint": name,
                                "scale": s[j], "raw_distance": raw_dist[j], "scaled_distance": scaled_dist[j]})

a_df = pd.DataFrame(a_rows)
b_df = pd.DataFrame(b_rows)
a_df.to_csv(OUT / "method_a_relative_error.csv", index=False, encoding="utf-8-sig")
b_df.to_csv(OUT / "method_b_scaled_distance.csv", index=False, encoding="utf-8-sig")
print("wrote method_a_relative_error.csv / method_b_scaled_distance.csv")


def plot_comparison(df, value_col, ylabel, title, out_stub):
    agg = df.groupby(["frame", "model", "camera"])[value_col].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, cam in zip(axes, c.CAMERAS):
        for model, color in [("17pt", "tab:blue"), ("19pt", "tab:red")]:
            sub = agg[(agg.camera == cam) & (agg.model == model)].sort_values("frame")
            ax.plot(sub["frame"], sub[value_col], label=model, color=color, linewidth=1.6)
        ax.set_title(f"Camera{cam}")
        ax.set_xlabel("frame")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / f"{out_stub}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{out_stub}.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_stub}.png / .eps")


plot_comparison(a_df, "relative_error", "相対誤差(誤差/肩-腰長)",
                 "2D 方法A: 骨長基準の相対誤差(上半身6点平均) frame33-82", "comparison_method_a")
plot_comparison(b_df, "scaled_distance", "スケール補正後の距離 [px]",
                 "2D 方法B: スケール補正後の距離(上半身6点平均) frame33-82", "comparison_method_b")

summary_a = a_df.groupby(["model", "camera"])["relative_error"].agg(["mean", "std"]).reset_index()
summary_b = b_df.groupby(["model", "camera"])["scaled_distance"].agg(["mean", "std"]).reset_index()
print("\n--- 方法A 要約 ---")
print(summary_a.to_string(index=False))
print("\n--- 方法B 要約 ---")
print(summary_b.to_string(index=False))

check_rows = []
for name, df, col in [("A", a_df, "relative_error"), ("B", b_df, "scaled_distance")]:
    agg = df.groupby(["frame", "model", "camera"])[col].mean().reset_index()
    piv = agg.pivot_table(index=["frame", "camera"], columns="model", values=col).reset_index()
    piv["19pt_better"] = piv["19pt"] < piv["17pt"]
    for cam in c.CAMERAS:
        sub = piv[piv.camera == cam]
        n_better = int(sub["19pt_better"].sum())
        frames_better = sub[sub["19pt_better"]]["frame"].tolist()
        check_rows.append({"method": name, "camera": cam, "n_total": len(sub), "n_19pt_better": n_better,
                            "pct_19pt_better": n_better / len(sub) * 100,
                            "frames_19pt_better": str(frames_better)})
        print(f"method {name} camera{cam}: 19pt better in {n_better}/{len(sub)} frames")

pd.DataFrame(check_rows).to_csv(OUT / "favorable_frame_check.csv", index=False, encoding="utf-8-sig")
print("wrote favorable_frame_check.csv")
