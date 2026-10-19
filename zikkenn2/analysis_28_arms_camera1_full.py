# -*- coding: utf-8 -*-
"""
28. Camera1, frame1〜222について、両腕(左右の肩・肘・手首、6点)の推定値と正解データの
    距離(位置誤差)を、2D・3D、拡張前(17点)・拡張後(19点)モデルそれぞれで算出する。

出力: zikkenn2/28/
  distance_{17pt,19pt}_{2d,3d}.csv   (計4ファイル) frame, keypoint, 推定x/y(/z), 正解x/y(/z), distance
  distance_summary.csv               キーポイント×モデル×次元の平均・標準偏差の表
  distance_summary_table.png/.eps    上記の表画像
  arms_distance_grid.png/.eps        2D/3D(行)×17点/19点(列)の4パネルグラフ(各6キーポイント)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

import common as c

OUT = c.BASE_DIR / "28"
CAM = 1
FRAME_RANGE = (1, 222)

# 19pt idx -> 17pt 2D(COCO17) idx, 17pt 3D(H36M17) idx
TARGETS = {
    "left_shoulder": (1, 5, 11), "right_shoulder": (2, 6, 14),
    "left_elbow": (3, 7, 12), "right_elbow": (4, 8, 15),
    "left_wrist": (5, 9, 13), "right_wrist": (6, 10, 16),
}

pred19_2d = c.load_2d_pred("19pt", CAM)
pred17_2d = c.load_2d_pred("17pt", CAM)
gt2d = c.load_2d_gt(CAM)
pred19_3d = c.load_3d_pred("19pt", CAM)
pred17_3d = c.load_3d_pred("17pt", CAM)
gt3d = c.load_3d_gt(CAM)

frames_all = pred19_2d["frame"].values
mask = (frames_all >= FRAME_RANGE[0]) & (frames_all <= FRAME_RANGE[1])
frames = frames_all[mask]

series = {}  # (model, dim) -> {keypoint: (frames, distance)}
summary_rows = []

for model in ["17pt", "19pt"]:
    for dim in ["2d", "3d"]:
        rows = []
        s = {}
        for name, (i19, i17_2d, i17_3d) in TARGETS.items():
            if dim == "2d":
                p = c.pred_xy(pred19_2d if model == "19pt" else pred17_2d,
                               i19 if model == "19pt" else i17_2d)[mask]
                g = c.gt_xy(gt2d, i19)[mask]
            else:
                p = c.pred_xyz(pred19_3d if model == "19pt" else pred17_3d,
                                i19 if model == "19pt" else i17_3d)[mask]
                g = c.gt_xyz(gt3d, i19)[mask]

            dist = c.dist(p, g)
            s[name] = (frames, dist)
            summary_rows.append({"model": model, "dim": dim, "keypoint": name,
                                  "mean": dist.mean(), "std": dist.std()})

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
        out_path = OUT / f"distance_{model}_{dim}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"wrote {out_path.name}")
        series[(model, dim)] = s

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "distance_summary.csv", index=False, encoding="utf-8-sig")
print("\n--- summary ---")
print(summary.to_string(index=False))

# ---------------- 表画像 ----------------
piv = summary.pivot_table(index="keypoint", columns=["dim", "model"], values="mean")
piv = piv.reindex(list(TARGETS.keys()))
piv = piv[[("2d", "17pt"), ("2d", "19pt"), ("3d", "17pt"), ("3d", "19pt")]]
piv.columns = ["2D_17点(px)", "2D_19点(px)", "3D_17点(norm)", "3D_19点(norm)"]
piv = piv.reset_index()

header = list(piv.columns)
cell_text = [[row["keypoint"]] + [f"{row[col]:.4g}" for col in header[1:]] for _, row in piv.iterrows()]
all_rows = [header] + cell_text
fig, ax = plt.subplots(figsize=(11, 0.55 * len(all_rows) + 1))
ax.axis("off")
table = ax.table(cellText=all_rows, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.9)
for (r, cidx), cell in table.get_celld().items():
    cell.set_edgecolor("#444444")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(weight="bold")
    if cidx == 0:
        cell.set_text_props(ha="left")
ax.set_title(f"両腕の推定→正解 平均距離 Camera{CAM}, frame{FRAME_RANGE[0]}-{FRAME_RANGE[1]}", fontsize=12, pad=10)
fig.tight_layout()
fig.savefig(OUT / "distance_summary_table.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "distance_summary_table.eps", bbox_inches="tight")
plt.close(fig)
print("wrote distance_summary_table.png / .eps")

# ---------------- グラフ: 2x2(行=2D/3D, 列=17点/19点) ----------------
cmap = plt.get_cmap("tab10")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for row_i, dim in enumerate(["2d", "3d"]):
    for col_i, model in enumerate(["17pt", "19pt"]):
        ax = axes[row_i, col_i]
        s = series[(model, dim)]
        for i, (name, (fr, dist)) in enumerate(s.items()):
            ax.plot(fr, dist, color=cmap(i), label=name, linewidth=1.2)
        ax.set_title(f"{'2D' if dim == '2d' else '3D'} / {model}", fontsize=12)
        ax.set_xlabel("frame")
        ax.set_ylabel("distance [px]" if dim == "2d" else "distance [normalized]")
        ax.grid(True, alpha=0.3)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=8)
fig.suptitle(f"両腕: 推定→正解の距離 Camera{CAM}, frame{FRAME_RANGE[0]}-{FRAME_RANGE[1]}", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "arms_distance_grid.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "arms_distance_grid.eps", bbox_inches="tight")
plt.close(fig)
print("wrote arms_distance_grid.png / .eps")
