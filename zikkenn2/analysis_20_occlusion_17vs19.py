# -*- coding: utf-8 -*-
"""
20. セルフオクルージョンに対する推定精度: 17点モデル vs 19点モデル

    Camera1・Camera4は同一モーションを別視点から撮影しているため、同じフレーム
    (=同じ実際の姿勢)でも視点によって遮蔽の有無が異なる。これを利用し、
    「Camera1のみ遮蔽 / Camera4のみ遮蔽」というペアに絞って
      誤差の差 = 遮蔽側カメラの誤差 − 可視側カメラの誤差
    を計算する。姿勢(動作量)が完全に同一なペアで比較するため、従来の
    「遮蔽時/可視時で単純にフレームを分けて比較する」方法(5番)にあった
    動作量との交絡を排除できる。

    対象キーポイント: 17点・19点の両モデルに共通する12点(肩・肘・手首・股関節・膝・足首)。
    脊柱系(spine0〜5)は17点モデルに存在しないため対象外。
    可視性(occluded/visible)は19点正解データのJ{n}_visibilityを、モデルによらない
    共通の遮蔽ラベルとして使用する(遮蔽は実世界の幾何で決まり、モデルに依存しないため)。
    誤差は2D(px)で統一(2Dは単位が共通で直接比較できるため)。

出力: zikkenn2/20/
  occlusion_pair_errors.csv         フレーム・キーポイントごとの誤差とペア分類
  occlusion_paired_diff_summary.csv 「片方だけ遮蔽」ペアでの誤差増分(17点/19点比較)
  occlusion_stratified_summary.csv  (参考)遮蔽時/可視時で単純に分けた場合のMAE
  occlusion_paired_diff.png/.eps    誤差増分の比較棒グラフ
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

import common as c

OUT = c.BASE_DIR / "20"

# 19点(idx)→17点COCO(idx) の対応。共通12点(肩・肘・手首・股関節・膝・足首)のみ。
KP19_TO_KP17 = {
    1: 5, 2: 6,     # shoulders
    3: 7, 4: 8,     # elbows
    5: 9, 6: 10,    # wrists
    7: 11, 8: 12,   # hips
    9: 13, 10: 14,  # knees
    11: 15, 12: 16, # ankles
}

rows = []
for cam in c.CAMERAS:
    pred19 = c.load_2d_pred("19pt", cam)
    pred17 = c.load_2d_pred("17pt", cam)
    gt = c.load_2d_gt(cam)
    frames = pred19["frame"].values

    for i19, i17 in KP19_TO_KP17.items():
        name = c.KP19_NAMES[i19]
        err19 = c.dist(c.pred_xy(pred19, i19), c.gt_xy(gt, i19))
        err17 = c.dist(c.pred_xy(pred17, i17), c.gt_xy(gt, i19))
        vis = gt[f"J{i19}_visibility"].values
        for fr, e19, e17, v in zip(frames, err19, err17, vis):
            rows.append({"camera": cam, "frame": fr, "keypoint": name, "err_19pt": e19, "err_17pt": e17,
                         "visible": bool(v >= 2)})

df = pd.DataFrame(rows)
df.to_csv(OUT / "occlusion_pair_errors.csv", index=False, encoding="utf-8-sig")
print("wrote occlusion_pair_errors.csv")

# ---------------- (参考)単純な層別: 遮蔽時/可視時のMAE ----------------
strat_rows = []
for model, col in [("17pt", "err_17pt"), ("19pt", "err_19pt")]:
    for vis_flag, label in [(True, "可視時"), (False, "遮蔽時")]:
        sub = df[df.visible == vis_flag]
        strat_rows.append({"モデル": model, "状態": label, "MAE(px)": sub[col].mean(), "n": len(sub)})
strat_df = pd.DataFrame(strat_rows)
strat_df.to_csv(OUT / "occlusion_stratified_summary.csv", index=False, encoding="utf-8-sig")
print("\n--- (参考)単純な層別比較 ---")
print(strat_df.to_string(index=False))

# ---------------- 本題: Camera1/Camera4ペアでの「片方だけ遮蔽」比較 ----------------
piv19 = df.pivot_table(index=["frame", "keypoint"], columns="camera", values="err_19pt")
piv17 = df.pivot_table(index=["frame", "keypoint"], columns="camera", values="err_17pt")
visp = df.pivot_table(index=["frame", "keypoint"], columns="camera", values="visible", aggfunc="first")

paired = pd.DataFrame({
    "err19_cam1": piv19[1], "err19_cam4": piv19[4],
    "err17_cam1": piv17[1], "err17_cam4": piv17[4],
    "vis_cam1": visp[1], "vis_cam4": visp[4],
}).dropna().reset_index()


def classify(row):
    if row.vis_cam1 and row.vis_cam4:
        return "両方可視"
    if (not row.vis_cam1) and (not row.vis_cam4):
        return "両方遮蔽"
    if (not row.vis_cam1) and row.vis_cam4:
        return "Camera1のみ遮蔽"
    return "Camera4のみ遮蔽"


paired["区分"] = paired.apply(classify, axis=1)

# 「片方だけ遮蔽」の誤差増分 = 遮蔽側の誤差 - 可視側の誤差 (同一フレーム・同一キーポイント)
diff_rows = []
for model, cam1_col, cam4_col in [("17pt", "err17_cam1", "err17_cam4"), ("19pt", "err19_cam1", "err19_cam4")]:
    m1 = paired[paired["区分"] == "Camera1のみ遮蔽"]
    d1 = m1[cam1_col] - m1[cam4_col]  # 遮蔽側(cam1) - 可視側(cam4)
    m4 = paired[paired["区分"] == "Camera4のみ遮蔽"]
    d4 = m4[cam4_col] - m4[cam1_col]  # 遮蔽側(cam4) - 可視側(cam1)
    diffs = pd.concat([d1, d4])
    diff_rows.append({
        "モデル": model, "n(ペア数)": len(diffs),
        "誤差増分 平均(px)": diffs.mean(), "誤差増分 中央値(px)": diffs.median(),
        "誤差増分>0の割合(%)": (diffs > 0).mean() * 100,
    })

diff_summary = pd.DataFrame(diff_rows)
diff_summary.to_csv(OUT / "occlusion_paired_diff_summary.csv", index=False, encoding="utf-8-sig")
print("\n--- 本題: 「片方だけ遮蔽」ペアでの誤差増分(遮蔽側-可視側、同一姿勢) ---")
print(diff_summary.to_string(index=False))

print("\n--- 区分ごとのペア数 ---")
print(paired["区分"].value_counts())

# ---------------- グラフ ----------------
fig, ax = plt.subplots(figsize=(6, 5))
colors = ["tab:blue", "tab:red"]
bars = ax.bar(diff_summary["モデル"], diff_summary["誤差増分 平均(px)"], color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("誤差増分の平均 [px]\n(遮蔽側カメラの誤差 − 可視側カメラの誤差)")
ax.set_title("セルフオクルージョンによる誤差増分: 17点 vs 19点\n(同一姿勢ペア比較、Camera1・4)")
for b, v in zip(bars, diff_summary["誤差増分 平均(px)"]):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "occlusion_paired_diff.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "occlusion_paired_diff.eps", bbox_inches="tight")
plt.close(fig)
print("\nwrote occlusion_paired_diff.png / .eps")
