# -*- coding: utf-8 -*-
"""
29. 28番(Camera1, frame1-222, 両腕6点, 2D/3D×17点/19点の距離)について、3Dのみ
    24番と同じ2つのスケール非依存な方法で補正する。2Dはpx単位で両モデル共通のため
    そのまま(28番のデータを流用)。

    方法A: 骨長基準の相対誤差 (距離 / 肩-腰長、同一モデル・同一フレーム内)
    方法B: スケール補正(N-MPJPE方式)。12点(肩・肘・手首・股関節・膝・足首)から
           最小二乗で最適スケールsを求め、推定にsを掛けてから距離を算出。

出力: zikkenn2/29/
  method_a_3d.csv, method_b_3d.csv   3Dの補正後データ
  summary_table.png/.eps             2D(28番流用)・3D方法A・3D方法Bを1つにまとめた表
  arms_grid_scalefixed.png/.eps      3行(2D/3D方法A/3D方法B)×2列(17点/19点)のグラフ
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

import common as c

IN28 = c.BASE_DIR / "28"
OUT = c.BASE_DIR / "29"
CAM = 1
FRAME_RANGE = (1, 222)

TARGETS = {
    "left_shoulder": (1, 11), "right_shoulder": (2, 14),
    "left_elbow": (3, 12), "right_elbow": (4, 15),
    "left_wrist": (5, 13), "right_wrist": (6, 16),
}
COMMON12 = {1: 11, 2: 14, 3: 12, 4: 15, 5: 13, 6: 16, 7: 4, 8: 1, 9: 5, 10: 2, 11: 6, 12: 3}

pred19 = c.load_3d_pred("19pt", CAM)
pred17 = c.load_3d_pred("17pt", CAM)
gt = c.load_3d_gt(CAM)

frames_all = pred19["frame"].values
mask = (frames_all >= FRAME_RANGE[0]) & (frames_all <= FRAME_RANGE[1])
frames = frames_all[mask]

a_rows, b_rows = [], []
series_a, series_b = {}, {}

for model, pred, idx_map in [("19pt", pred19, {k: k for k in COMMON12}), ("17pt", pred17, COMMON12)]:
    P = np.stack([c.pred_xyz(pred, idx_map[i19])[mask] for i19 in COMMON12], axis=1)
    G = np.stack([c.gt_xyz(gt, i19)[mask] for i19 in COMMON12], axis=1)
    s = (P * G).sum(axis=(1, 2)) / (P * P).sum(axis=(1, 2))

    if model == "19pt":
        shoulder_mid = (c.pred_xyz(pred, 1)[mask] + c.pred_xyz(pred, 2)[mask]) / 2
        hip_mid = (c.pred_xyz(pred, 7)[mask] + c.pred_xyz(pred, 8)[mask]) / 2
    else:
        shoulder_mid = (c.pred_xyz(pred, 11)[mask] + c.pred_xyz(pred, 14)[mask]) / 2
        hip_mid = c.pred_xyz(pred, 0)[mask]
    ref_length = c.dist(shoulder_mid, hip_mid)

    sa, sb = {}, {}
    for name, (i19, i17) in TARGETS.items():
        idx = i19 if model == "19pt" else i17
        p = c.pred_xyz(pred, idx)[mask]
        g = c.gt_xyz(gt, i19)[mask]
        raw_dist = c.dist(p, g)
        rel_err = raw_dist / ref_length
        scaled_dist = c.dist(p * s[:, None], g)
        sa[name] = (frames, rel_err)
        sb[name] = (frames, scaled_dist)

        for j, fr in enumerate(frames):
            a_rows.append({"frame": fr, "model": model, "keypoint": name,
                            "raw_distance": raw_dist[j], "reference_length": ref_length[j],
                            "relative_error": rel_err[j]})
            b_rows.append({"frame": fr, "model": model, "keypoint": name,
                            "scale": s[j], "raw_distance": raw_dist[j], "scaled_distance": scaled_dist[j]})
    series_a[model] = sa
    series_b[model] = sb

a_df = pd.DataFrame(a_rows)
b_df = pd.DataFrame(b_rows)
a_df.to_csv(OUT / "method_a_3d.csv", index=False, encoding="utf-8-sig")
b_df.to_csv(OUT / "method_b_3d.csv", index=False, encoding="utf-8-sig")
print("wrote method_a_3d.csv / method_b_3d.csv")

# ---------------- 2D(28番流用) ----------------
d2_17 = pd.read_csv(IN28 / "distance_17pt_2d.csv")
d2_19 = pd.read_csv(IN28 / "distance_19pt_2d.csv")
series_2d = {
    "17pt": {name: (g["frame"].values, g["distance"].values) for name, g in d2_17.groupby("keypoint")},
    "19pt": {name: (g["frame"].values, g["distance"].values) for name, g in d2_19.groupby("keypoint")},
}

# ---------------- 要約表 ----------------
summary_rows = []
for name in TARGETS:
    row = {"keypoint": name}
    for model in ["17pt", "19pt"]:
        row[f"2D_{model}(px)"] = np.mean(series_2d[model][name][1])
        row[f"3D方法A_{model}"] = np.mean(series_a[model][name][1])
        row[f"3D方法B_{model}"] = np.mean(series_b[model][name][1])
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "summary_table.csv", index=False, encoding="utf-8-sig")
print(summary.to_string(index=False))

header = list(summary.columns)
cell_text = [[row["keypoint"]] + [f"{row[c_]:.4g}" for c_ in header[1:]] for _, row in summary.iterrows()]
all_rows = [header] + cell_text
fig, ax = plt.subplots(figsize=(15, 0.55 * len(all_rows) + 1.3))
ax.axis("off")
table = ax.table(cellText=all_rows, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 1.9)
for (r, cidx), cell in table.get_celld().items():
    cell.set_edgecolor("#444444")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(weight="bold")
    if cidx == 0:
        cell.set_text_props(ha="left")
ax.set_title(f"両腕 距離の平均(3Dはスケール補正後) Camera{CAM}, frame{FRAME_RANGE[0]}-{FRAME_RANGE[1]}", fontsize=12, pad=10)
fig.tight_layout()
fig.savefig(OUT / "summary_table.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "summary_table.eps", bbox_inches="tight")
plt.close(fig)
print("wrote summary_table.png / .eps")

# ---------------- グラフ: 3行(2D/3D方法A/3D方法B) x 2列(17点/19点) ----------------
cmap = plt.get_cmap("tab10")
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
row_defs = [
    ("2D(px)", series_2d, "distance [px]"),
    ("3D 方法A(誤差/肩-腰長)", series_a, "relative error"),
    ("3D 方法B(スケール補正)", series_b, "scaled distance"),
]
for row_i, (row_label, data, ylabel) in enumerate(row_defs):
    for col_i, model in enumerate(["17pt", "19pt"]):
        ax = axes[row_i, col_i]
        for i, (name, (fr, val)) in enumerate(data[model].items()):
            ax.plot(fr, val, color=cmap(i), label=name, linewidth=1.2)
        ax.set_title(f"{row_label} / {model}", fontsize=11)
        ax.set_xlabel("frame")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=8)
fig.suptitle(f"両腕: 推定→正解の距離(3Dはスケール補正後) Camera{CAM}, frame{FRAME_RANGE[0]}-{FRAME_RANGE[1]}", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "arms_grid_scalefixed.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "arms_grid_scalefixed.eps", bbox_inches="tight")
plt.close(fig)
print("wrote arms_grid_scalefixed.png / .eps")
