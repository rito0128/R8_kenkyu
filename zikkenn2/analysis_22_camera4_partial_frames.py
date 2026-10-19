# -*- coding: utf-8 -*-
"""
19pt_2d3d/camera4/frames/out_Camera4_0232.png の左側(2Dオーバーレイ)パネルに、
実際の推定座標(19pt_2d, Camera4, frame232)を使ってキーポイントを大きな点で描画する。
番号は付けず、すべての点を同じ色の大きなマーカーで示す。

2種類作成する:
  ・赤色バージョン
  ・背景(暗赤色のカーテン・ベージュの床・白い肌・紺のジーンズ)に対して
    最も視認性が高くなる色(明るいライム系グリーン)のバージョン

出力: zikkenn2/22/out_Camera4_0232_dots_red.png
      zikkenn2/22/out_Camera4_0232_dots_visible.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
from PIL import Image

import common as c

OUT = c.BASE_DIR / "22"
OUT.mkdir(exist_ok=True)
CAM = 4
FRAME = 232

img_path = c.BASE_DIR / "19pt_2d3d" / f"camera{CAM}" / "frames" / f"out_Camera{CAM}_{FRAME:04d}.png"
full_img = Image.open(img_path)
left_panel = full_img.crop((0, 0, 1000, 1000))  # 左半分(2Dオーバーレイ)のみ切り出し

pred = c.load_2d_pred("19pt", CAM)
row = pred[pred.frame == FRAME].iloc[0]

xs = [row[f"kp{i}_x"] for i in range(19)]
ys = [row[f"kp{i}_y"] for i in range(19)]


def plot_dots(color, edgecolor, title_suffix, out_name):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(left_panel)
    ax.axis("off")
    ax.scatter(xs, ys, s=260, c=color, edgecolors=edgecolor, linewidths=1.8, zorder=3)
    ax.set_title(f"Camera{CAM} frame{FRAME}: キーポイント{title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / out_name}")


plot_dots("red", "white", "(赤)", f"out_Camera{CAM}_{FRAME:04d}_dots_red.png")
plot_dots("#39FF14", "black", "(視認性重視: ライムグリーン)", f"out_Camera{CAM}_{FRAME:04d}_dots_visible.png")
