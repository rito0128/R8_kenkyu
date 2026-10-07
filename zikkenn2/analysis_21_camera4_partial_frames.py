# -*- coding: utf-8 -*-
"""
21. Camera4, frame270-386について、推定データの各キーポイントから正解データの
    対応するキーポイントへの距離(位置誤差)を、17点モデル・19点モデルそれぞれの
    2D・3Dで計算する。

    19点モデル: 19点すべて(head, 体幹・四肢12点, spine0〜5)についてGTと直接比較。
    17点モデル: 17点モデルと19点GTで1対1対応が明確な12点(肩・肘・手首・股関節・膝・足首)
                のみ算出。COCO17の鼻・目・耳(2D)やH36M17のSpine/Thorax等(3D)は
                19点GTに対応する点が無いため対象外(20番の遮蔽評価と同じ方針)。

出力: zikkenn2/21/
  keypoint_distance_frames.csv   フレーム・キーポイントごとの距離生データ
  keypoint_distance_summary.csv  キーポイントごとの平均・標準偏差
  keypoint_distance_2d.png/.eps  2D: 17点/19点 比較棒グラフ
  keypoint_distance_3d.png/.eps  3D: 17点/19点 比較棒グラフ
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

import common as c

OUT = c.BASE_DIR / "21"
CAM = 4
FRAME_RANGE = (270, 386)

# 19点(idx)→17点(idx) の対応。共通12点(肩・肘・手首・股関節・膝・足首)のみ。
# 2Dは17pt=COCO17順、3Dは17pt=H36M17順で、インデックスの意味が異なる点に注意。
KP19_TO_KP17_2D = {1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11, 8: 12, 9: 13, 10: 14, 11: 15, 12: 16}
KP19_TO_KP17_3D = {1: 11, 2: 14, 3: 12, 4: 15, 5: 13, 6: 16, 7: 4, 8: 1, 9: 5, 10: 2, 11: 6, 12: 3}

pred19_2d = c.load_2d_pred("19pt", CAM)
pred17_2d = c.load_2d_pred("17pt", CAM)
gt2d = c.load_2d_gt(CAM)
pred19_3d = c.load_3d_pred("19pt", CAM)
pred17_3d = c.load_3d_pred("17pt", CAM)
gt3d = c.load_3d_gt(CAM)

frames_all = pred19_2d["frame"].values
mask = (frames_all >= FRAME_RANGE[0]) & (frames_all <= FRAME_RANGE[1])
frames = frames_all[mask]

rows = []
# 19点(2D・3D、19点すべて)
for i in range(19):
    name = c.KP19_NAMES[i]
    err2d = c.dist(c.pred_xy(pred19_2d, i), c.gt_xy(gt2d, i))[mask]
    err3d = c.dist(c.pred_xyz(pred19_3d, i), c.gt_xyz(gt3d, i))[mask]
    for fr, e2, e3 in zip(frames, err2d, err3d):
        rows.append({"モデル": "19点", "キーポイント": name, "frame": fr, "2D距離": e2, "3D距離": e3})

# 17点(共通12点のみ)
for i19, i17_2d in KP19_TO_KP17_2D.items():
    name = c.KP19_NAMES[i19]
    i17_3d = KP19_TO_KP17_3D[i19]
    err2d = c.dist(c.pred_xy(pred17_2d, i17_2d), c.gt_xy(gt2d, i19))[mask]
    err3d = c.dist(c.pred_xyz(pred17_3d, i17_3d), c.gt_xyz(gt3d, i19))[mask]
    for fr, e2, e3 in zip(frames, err2d, err3d):
        rows.append({"モデル": "17点", "キーポイント": name, "frame": fr, "2D距離": e2, "3D距離": e3})

df = pd.DataFrame(rows)
df.to_csv(OUT / "keypoint_distance_frames.csv", index=False, encoding="utf-8-sig")
print("wrote keypoint_distance_frames.csv")

summary = df.groupby(["モデル", "キーポイント"]).agg(
    n=("frame", "count"),
    dist2d_mean=("2D距離", "mean"), dist2d_std=("2D距離", "std"),
    dist3d_mean=("3D距離", "mean"), dist3d_std=("3D距離", "std"),
).reset_index()
summary.to_csv(OUT / "keypoint_distance_summary.csv", index=False, encoding="utf-8-sig")
print("wrote keypoint_distance_summary.csv")
print(summary.to_string(index=False))

KP_ORDER = c.KP19_NAMES


def plot_bar(dim_col, ylabel, title, out_stub):
    piv = summary.pivot(index="キーポイント", columns="モデル", values=dim_col)
    piv = piv.reindex(KP_ORDER)

    x = np.arange(len(piv.index))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.5))
    if "17点" in piv.columns:
        ax.bar(x - width / 2, piv["17点"], width, label="17点モデル", color="tab:blue")
    ax.bar(x + width / 2, piv["19点"], width, label="19点モデル", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{out_stub}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{out_stub}.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_stub}.png / .eps")


plot_bar("dist2d_mean", "平均距離(位置誤差) [px]",
          f"2D: 推定→正解の距離(キーポイント別) Camera{CAM}, frame{FRAME_RANGE[0]}-{FRAME_RANGE[1]}",
          "keypoint_distance_2d")
plot_bar("dist3d_mean", "平均距離(位置誤差) [normalized]",
          f"3D: 推定→正解の距離(キーポイント別) Camera{CAM}, frame{FRAME_RANGE[0]}-{FRAME_RANGE[1]}",
          "keypoint_distance_3d")
