# -*- coding: utf-8 -*-
"""
18. 柔軟性の評価: Camera4, frame270-386, 19点拡張後モデル(2D・3D)を対象に、
    恒常性の評価とは区別した4つの評価軸で関節角度データをまとめる。

    対象関節(9関節): spine0〜spine4(脊柱)、left/right_elbow、left/right_knee(四肢)
    ※ head・spine5はチェーン両端のため角度算出対象外(これまでと同じ制約)

    恒常性(長さ)との違い:
      恒常性 = 「本来一定であるはずの長さ」がどれだけ変動してしまうか(変動=誤差)
      柔軟性 = 「本来動作に応じて変化するべき角度」を、動きのパターンごと正しく
               捉えられているか(変動そのものではなく、変動の中身・タイミングを評価)

    評価軸:
      ① 可動域(Range of Motion): 推定・正解それぞれの角度の最小〜最大
      ② フレーム間相関: 推定角度と正解角度のピアソン相関係数(動きのタイミングの一致度)
      ③ MAE(平均絶対誤差): |推定-正解|の平均(値そのものの精度)
      ④ フレーム間変化量・変化率: Δ角度(t)=角度(t)-角度(t-1) の平均絶対値と、
         平均角度に対する比率(%)。推定・正解それぞれで算出し、動きの速さの再現度を見る。

出力: zikkenn2/18/
  flexibility_axes_frames.csv       フレームごとの角度生データ(2D/3D 推定/正解)
  axis1_range_of_motion.csv
  axis2_correlation.csv
  axis3_mae.csv
  axis4_frame_change.csv
  flexibility_axes_summary.csv      ①〜④を1つにまとめた表
  flexibility_axes_summary.png/.eps 上記の画像版
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "18"
CAM = 4
FRAME_RANGE = (270, 386)

JOINTS = {
    "spine0": (0, 13, 14),
    "spine1": (13, 14, 15),
    "spine2": (14, 15, 16),
    "spine3": (15, 16, 17),
    "spine4": (16, 17, 18),
    "left_elbow": (1, 3, 5),
    "right_elbow": (2, 4, 6),
    "left_knee": (7, 9, 11),
    "right_knee": (8, 10, 12),
}
SPINE_JOINTS = ["spine0", "spine1", "spine2", "spine3", "spine4"]
LIMB_JOINTS = ["left_elbow", "right_elbow", "left_knee", "right_knee"]
JOINT_ORDER = SPINE_JOINTS + LIMB_JOINTS


def angle3pt(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos = (v1 * v2).sum(axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


pred2d = c.load_2d_pred("19pt", CAM)
gt2d = c.load_2d_gt(CAM)
pred3d = c.load_3d_pred("19pt", CAM)
gt3d = c.load_3d_gt(CAM)

frames_all = pred2d["frame"].values
mask = (frames_all >= FRAME_RANGE[0]) & (frames_all <= FRAME_RANGE[1])
frames = frames_all[mask]

angles = {}  # joint -> {"2D推定":..., "2D正解":..., "3D推定":..., "3D正解":...}
frame_rows = []
for name, (a, b, cidx) in JOINTS.items():
    a2p = angle3pt(c.pred_xy(pred2d, a), c.pred_xy(pred2d, b), c.pred_xy(pred2d, cidx))[mask]
    a2g = angle3pt(c.gt_xy(gt2d, a), c.gt_xy(gt2d, b), c.gt_xy(gt2d, cidx))[mask]
    a3p = angle3pt(c.pred_xyz(pred3d, a), c.pred_xyz(pred3d, b), c.pred_xyz(pred3d, cidx))[mask]
    a3g = angle3pt(c.gt_xyz(gt3d, a), c.gt_xyz(gt3d, b), c.gt_xyz(gt3d, cidx))[mask]
    angles[name] = {"2D推定": a2p, "2D正解": a2g, "3D推定": a3p, "3D正解": a3g}
    for fr, v2p, v2g, v3p, v3g in zip(frames, a2p, a2g, a3p, a3g):
        frame_rows.append({"関節": name, "frame": fr, "2D推定": v2p, "2D正解": v2g, "3D推定": v3p, "3D正解": v3g})

pd.DataFrame(frame_rows).to_csv(OUT / "flexibility_axes_frames.csv", index=False, encoding="utf-8-sig")
print("wrote flexibility_axes_frames.csv")

# ---------------- ①可動域 ----------------
rows1 = []
for name in JOINT_ORDER:
    s = angles[name]
    rows1.append({
        "関節": name,
        "2D推定範囲": f"{s['2D推定'].min():.4g}°〜{s['2D推定'].max():.4g}°",
        "2D正解範囲": f"{s['2D正解'].min():.4g}°〜{s['2D正解'].max():.4g}°",
        "3D推定範囲": f"{s['3D推定'].min():.4g}°〜{s['3D推定'].max():.4g}°",
        "3D正解範囲": f"{s['3D正解'].min():.4g}°〜{s['3D正解'].max():.4g}°",
    })
axis1 = pd.DataFrame(rows1)
axis1.to_csv(OUT / "axis1_range_of_motion.csv", index=False, encoding="utf-8-sig")

# ---------------- ②フレーム間相関 ----------------
rows2 = []
for name in JOINT_ORDER:
    s = angles[name]
    rows2.append({
        "関節": name,
        "2D相関": np.corrcoef(s["2D推定"], s["2D正解"])[0, 1],
        "3D相関": np.corrcoef(s["3D推定"], s["3D正解"])[0, 1],
    })
axis2 = pd.DataFrame(rows2)
for col in ["2D相関", "3D相関"]:
    axis2[col] = axis2[col].map(c.sig4)
axis2.to_csv(OUT / "axis2_correlation.csv", index=False, encoding="utf-8-sig")

# ---------------- ③MAE ----------------
rows3 = []
for name in JOINT_ORDER:
    s = angles[name]
    rows3.append({
        "関節": name,
        "2D_MAE(°)": np.abs(s["2D推定"] - s["2D正解"]).mean(),
        "3D_MAE(°)": np.abs(s["3D推定"] - s["3D正解"]).mean(),
    })
axis3 = pd.DataFrame(rows3)
for col in ["2D_MAE(°)", "3D_MAE(°)"]:
    axis3[col] = axis3[col].map(c.sig4)
axis3.to_csv(OUT / "axis3_mae.csv", index=False, encoding="utf-8-sig")

# ---------------- ④フレーム間変化量・変化率 ----------------
rows4 = []
for name in JOINT_ORDER:
    s = angles[name]
    d2p = np.abs(np.diff(s["2D推定"])); d2g = np.abs(np.diff(s["2D正解"]))
    d3p = np.abs(np.diff(s["3D推定"])); d3g = np.abs(np.diff(s["3D正解"]))
    rows4.append({
        "関節": name,
        "2D推定 変化量(°)": d2p.mean(), "2D推定 変化率(%)": d2p.mean() / s["2D推定"].mean() * 100,
        "2D正解 変化量(°)": d2g.mean(), "2D正解 変化率(%)": d2g.mean() / s["2D正解"].mean() * 100,
        "3D推定 変化量(°)": d3p.mean(), "3D推定 変化率(%)": d3p.mean() / s["3D推定"].mean() * 100,
        "3D正解 変化量(°)": d3g.mean(), "3D正解 変化率(%)": d3g.mean() / s["3D正解"].mean() * 100,
    })
axis4 = pd.DataFrame(rows4)
for col in axis4.columns[1:]:
    axis4[col] = axis4[col].map(c.sig4)
axis4.to_csv(OUT / "axis4_frame_change.csv", index=False, encoding="utf-8-sig")

print("wrote axis1〜axis4 csv")
print("\n--- axis1 可動域 ---")
print(axis1.to_string(index=False))
print("\n--- axis2 相関 ---")
print(axis2.to_string(index=False))
print("\n--- axis3 MAE ---")
print(axis3.to_string(index=False))
print("\n--- axis4 変化量・変化率 ---")
print(axis4.to_string(index=False))

# ---------------- 統合表 ----------------
summary = axis1.merge(axis2, on="関節").merge(axis3, on="関節").merge(axis4, on="関節")
summary.to_csv(OUT / "flexibility_axes_summary.csv", index=False, encoding="utf-8-sig")
print("wrote flexibility_axes_summary.csv")

# ---------------- 画像版(見やすいよう軸ごとに区切って表示) ----------------
def draw_table(ax, df, title):
    header = list(df.columns)
    cell_text = df.values.tolist()
    all_rows = [header] + cell_text
    ax.axis("off")
    table = ax.table(cellText=all_rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    for (r, cidx), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        if cidx == 0:
            cell.set_text_props(ha="left")
    ax.set_title(title, fontsize=12, pad=8, loc="left")


fig, axes = plt.subplots(4, 1, figsize=(13, 22))
draw_table(axes[0], axis1, "① 可動域(Range of Motion)")
draw_table(axes[1], axis2, "② フレーム間相関(動きのタイミングの一致度)")
draw_table(axes[2], axis3, "③ MAE(平均絶対誤差)")
draw_table(axes[3], axis4, "④ フレーム間変化量・変化率(動きの速さの再現度)")
fig.suptitle(f"柔軟性の評価: Camera{CAM}, frame{FRAME_RANGE[0]}-{FRAME_RANGE[1]}, 19点モデル", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "flexibility_axes_summary.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "flexibility_axes_summary.eps", bbox_inches="tight")
plt.close(fig)
print("wrote flexibility_axes_summary.png / .eps")
