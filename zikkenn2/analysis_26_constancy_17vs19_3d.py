# -*- coding: utf-8 -*-
"""
26. 恒常性の評価(3Dのみ): 頭〜背5にあたる区間の長さを算出し、フレーム間の変動係数
    (CV=標準偏差/平均値)を推定結果と正解データで比較する。3Dの拡張前(17点)・
    拡張後(19点)モデルの両方で実施し、比較する。Camera1・Camera4、全386フレーム。

    19点モデル: head-spine0-spine1-spine2-spine3-spine4-spine5 の6区間(既存の定義)。

    17点モデル: spine0〜spine5に相当する点が無いため、H36M17が実際に持つ体幹の
    チェーン Head_top(10)-Neck/Nose(9)-Thorax(8)-Spine(7)-Hip(0) の4区間で代用する。
    正解データとの対応(GT側に無い点は近似で対応させる):
      head_top  ≒ GT head(J0)
      neck_nose ≒ GT spine0(J13)   [これまでの分析で首寄りと確認済み]
      thorax    ≒ GT spine1(J14)   [近似: 首寄り]
      spine     ≒ GT spine3(J16)   [近似: 中間]
      hip       ≒ GT spine5(J18)   [これまでの分析で腰寄りと確認済み]
    区間数(17点=4, 19点=6)が異なる点、および17点側の一部対応が近似である点に注意。

出力: zikkenn2/26/
  cv_comparison_17vs19_3d.csv        区間×モデル×カメラのCV表(生データ)
  cv_comparison_17vs19_3d.png/.eps   棒グラフ(モデル×カメラ、推定/正解)
  cv_comparison_table.png/.eps       表画像
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

# 19点モデルのチェーン(idx): 既存のc.SPINE_CHAIN / c.SPINE_CHAIN_LABELSを使用

# 17点モデル(H36M idx)のチェーンと、対応するGT(19点)idx
CHAIN_17 = [(10, 9), (9, 8), (8, 7), (7, 0)]
LABELS_17 = ["head_top-neck_nose", "neck_nose-thorax", "thorax-spine", "spine-hip"]
GT_PROXY_17 = {10: 0, 9: 13, 8: 14, 7: 16, 0: 18}  # 17pt idx -> GT(19pt) idx


def cv(arr):
    arr = np.asarray(arr, dtype=float)
    return arr.std() / arr.mean()


rows = []
for cam in c.CAMERAS:
    pred19 = c.load_3d_pred("19pt", cam)
    pred17 = c.load_3d_pred("17pt", cam)
    gt = c.load_3d_gt(cam)

    # 19点モデル: 6区間
    for (a, b), label in zip(c.SPINE_CHAIN, c.SPINE_CHAIN_LABELS):
        len_pred = c.dist(c.pred_xyz(pred19, a), c.pred_xyz(pred19, b))
        len_gt = c.dist(c.gt_xyz(gt, a), c.gt_xyz(gt, b))
        rows.append({"camera": cam, "model": "19点(拡張後)", "区間": label,
                      "推定CV": cv(len_pred), "正解CV": cv(len_gt)})

    # 17点モデル: 4区間(GT近似対応)
    for (a, b), label in zip(CHAIN_17, LABELS_17):
        len_pred = c.dist(c.pred_xyz(pred17, a), c.pred_xyz(pred17, b))
        ga, gb = GT_PROXY_17[a], GT_PROXY_17[b]
        len_gt = c.dist(c.gt_xyz(gt, ga), c.gt_xyz(gt, gb))
        rows.append({"camera": cam, "model": "17点(拡張前)", "区間": label,
                      "推定CV": cv(len_pred), "正解CV": cv(len_gt)})

df = pd.DataFrame(rows)
df.to_csv(OUT / "cv_comparison_17vs19_3d.csv", index=False, encoding="utf-8-sig")
print("wrote cv_comparison_17vs19_3d.csv")
print(df.to_string(index=False))

# ---------------- グラフ: モデル×カメラの4パネル、区間ごとに推定/正解を並べた棒グラフ ----------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for r, model in enumerate(["17点(拡張前)", "19点(拡張後)"]):
    for cidx, cam in enumerate(c.CAMERAS):
        ax = axes[r, cidx]
        sub = df[(df.model == model) & (df.camera == cam)]
        x = np.arange(len(sub))
        width = 0.38
        ax.bar(x - width / 2, sub["推定CV"], width, label="推定", color="tab:red")
        ax.bar(x + width / 2, sub["正解CV"], width, label="正解", color="black", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["区間"], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{model} / Camera{cam}", fontsize=11)
        ax.set_ylabel("CV")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
fig.suptitle("恒常性の評価(3D): 17点 vs 19点モデルのCV比較", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "cv_comparison_17vs19_3d.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "cv_comparison_17vs19_3d.eps", bbox_inches="tight")
plt.close(fig)
print("wrote cv_comparison_17vs19_3d.png / .eps")

# ---------------- 表画像 ----------------
fmt_df = df.copy()
fmt_df["推定CV"] = fmt_df["推定CV"].map(lambda v: f"{v:.4g}")
fmt_df["正解CV"] = fmt_df["正解CV"].map(lambda v: f"{v:.4g}")
header = list(fmt_df.columns)
all_rows = [header] + fmt_df.values.tolist()

fig, ax = plt.subplots(figsize=(10, 0.5 * len(all_rows) + 1))
ax.axis("off")
table = ax.table(cellText=all_rows, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
for (r, cidx), cell in table.get_celld().items():
    cell.set_edgecolor("#444444")
    cell.set_linewidth(0.5)
    if r == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(weight="bold")
    if cidx in (1, 2):
        cell.set_text_props(ha="left")
ax.set_title("恒常性の評価(3D): 17点 vs 19点モデルのCV比較", fontsize=12, pad=10)
fig.tight_layout()
fig.savefig(OUT / "cv_comparison_table.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "cv_comparison_table.eps", bbox_inches="tight")
plt.close(fig)
print("wrote cv_comparison_table.png / .eps")
