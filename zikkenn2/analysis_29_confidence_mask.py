# -*- coding: utf-8 -*-
"""
29. 2D推定のキーポイント信頼度(keypoint_scores)を、アバター領域内/領域外で層別して
    比較する。generated_demo_img/motion_00x_test/ のマスク画像
    (out_Camera{N}_XXXX_mask.png: 0=背景, 255=アバター, 境界はアンチエイリアス)を使い、
    各キーポイントの推定座標がアバター領域内か領域外かを判定する。

    対象: 17点・19点モデル、Camera1・Camera4、全386フレーム、2D推定のみ。

出力: zikkenn2/29/
  confidence_by_region_frames.csv   フレーム・キーポイントごとの信頼度と領域判定
  confidence_by_region_summary.csv  モデル×カメラ×領域(内/外)の信頼度 最大・最小・平均
  confidence_by_region_summary.png/.eps  表画像
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

import common as c

OUT = c.BASE_DIR / "29"
MASK_DIR = Path(r"C:\Users\sinki\R8_kenkyu\generated_demo_img\motion_00x_test")
MASK_THRESHOLD = 127

rows = []
for cam in c.CAMERAS:
    for model in ["17pt", "19pt"]:
        pred_dir = c.BASE_DIR / f"{model}_2d" / f"camera{cam}" / "predictions"
        n_kp = 17 if model == "17pt" else 19
        for frame in range(1, c.N_FRAMES + 1):
            pred_path = pred_dir / f"out_Camera{cam}_{frame:04d}.json"
            mask_path = MASK_DIR / f"out_Camera{cam}_{frame:04d}_mask.png"
            data = json.load(open(pred_path, encoding="utf-8"))
            kp = np.asarray(data[0]["keypoints"])
            scores = np.asarray(data[0]["keypoint_scores"])
            mask = np.asarray(Image.open(mask_path))
            h, w = mask.shape[:2]

            for i in range(n_kp):
                x, y = kp[i]
                xi = int(np.clip(round(x), 0, w - 1))
                yi = int(np.clip(round(y), 0, h - 1))
                in_avatar = bool(mask[yi, xi] > MASK_THRESHOLD)
                rows.append({
                    "model": model, "camera": cam, "frame": frame, "kp_idx": i,
                    "score": float(scores[i]), "in_avatar": in_avatar,
                })

df = pd.DataFrame(rows)
df.to_csv(OUT / "confidence_by_region_frames.csv", index=False, encoding="utf-8-sig")
print(f"wrote confidence_by_region_frames.csv ({len(df)} rows)")

df["region"] = df["in_avatar"].map({True: "領域内(アバター)", False: "領域外(背景)"})
summary = df.groupby(["model", "camera", "region"])["score"].agg(
    n="count", 最大値="max", 最小値="min", 平均値="mean"
).reset_index()
summary.to_csv(OUT / "confidence_by_region_summary.csv", index=False, encoding="utf-8-sig")
print("wrote confidence_by_region_summary.csv")
print(summary.to_string(index=False))

# ---------------- 表画像 ----------------
fmt = summary.copy()
for col in ["最大値", "最小値", "平均値"]:
    fmt[col] = fmt[col].map(lambda v: f"{v:.4g}")
header = list(fmt.columns)
all_rows = [header] + fmt.values.tolist()

fig, ax = plt.subplots(figsize=(10, 0.5 * len(all_rows) + 1.2))
ax.axis("off")
table = ax.table(cellText=all_rows, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.7)
for (r, cidx), cell in table.get_celld().items():
    cell.set_edgecolor("#444444")
    cell.set_linewidth(0.5)
    if r == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(weight="bold")
    if cidx in (0, 2):
        cell.set_text_props(ha="left")
ax.set_title("2D信頼度: アバター領域内/領域外での比較", fontsize=13, pad=10)
fig.tight_layout()
fig.savefig(OUT / "confidence_by_region_summary.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "confidence_by_region_summary.eps", bbox_inches="tight")
plt.close(fig)
print("wrote confidence_by_region_summary.png / .eps")
