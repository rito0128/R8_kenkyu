"""
head-spine0-spine1-spine2-spine3-spine4-spine5 の6区間について、
フレーム間の変動係数(CV=std/mean)を以下の条件でまとめる:
  - 2D: 全カメラ(12台プール) / Camera1のみ  (推定・正解)
  - 3D: Camera1のみ                          (推定・正解)

出力:
  spine_cv_summary.csv
  spine_cv_summary.png
  spine_cv_summary.eps
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

KP_NAMES = ['head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
            'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle',
            'spine0','spine1','spine2','spine3','spine4','spine5']

CHAIN = [(0, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18)]
ROW_LABELS = [f"{KP_NAMES[a]}-{KP_NAMES[b]}" for a, b in CHAIN]


def cv(series):
    return series.std() / series.mean()


def dist_xy(df, a, b):
    ax, ay = df[f"kp{a}_{KP_NAMES[a]}_x"], df[f"kp{a}_{KP_NAMES[a]}_y"]
    bx, by = df[f"kp{b}_{KP_NAMES[b]}_x"], df[f"kp{b}_{KP_NAMES[b]}_y"]
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def dist_xyz(df, a, b):
    ax, ay, az = df[f"J{a}_x"], df[f"J{a}_y"], df[f"J{a}_z"]
    bx, by, bz = df[f"J{b}_x"], df[f"J{b}_y"], df[f"J{b}_z"]
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def cv_series_2d(df):
    return [cv(dist_xy(df, a, b)) for a, b in CHAIN]


# ---- 2D ----
pred2d = pd.read_csv(BASE_DIR / "for_test_HRNet" / "keypoint_lengths.csv")
gt2d = pd.read_csv(BASE_DIR / "for_test_HRNet" / "correct_keypoint_lengths.csv")

col_2d_pred_all = cv_series_2d(pred2d)
col_2d_gt_all = cv_series_2d(gt2d)
col_2d_pred_c1 = cv_series_2d(pred2d[pred2d.camera == 1])
col_2d_gt_c1 = cv_series_2d(gt2d[gt2d.camera == 1])

# ---- 3D (Camera1) ----
pred3d = pd.read_csv(
    BASE_DIR / "for_test_motionbert" / "motionbert_20260827_motion_007" / "motionbert_keypoint_lengths.csv"
)
pred3d1 = pred3d[pred3d.camera == 1]
chain_cols = [f"len_{KP_NAMES[a]}_{KP_NAMES[b]}" for a, b in CHAIN]
col_3d_pred_c1 = [cv(pred3d1[c]) for c in chain_cols]

gt3d = pd.read_csv(
    BASE_DIR / "for_test_motionbert" / "motionbert_20260827_motion_007" / "correct_3d_anotation.csv"
)
gt3d1 = gt3d[gt3d["filename"].str.startswith("out_Camera1_")]
col_3d_gt_c1 = [cv(dist_xyz(gt3d1, a, b)) for a, b in CHAIN]

# ---- まとめ ----
columns = {
    ("2D 全カメラ", "推定"): col_2d_pred_all,
    ("2D 全カメラ", "正解"): col_2d_gt_all,
    ("2D Camera1", "推定"): col_2d_pred_c1,
    ("2D Camera1", "正解"): col_2d_gt_c1,
    ("3D Camera1", "推定"): col_3d_pred_c1,
    ("3D Camera1", "正解"): col_3d_gt_c1,
}

df_out = pd.DataFrame(columns, index=ROW_LABELS)
df_out.columns = pd.MultiIndex.from_tuples(df_out.columns)
df_out.index.name = "区間"

csv_path = BASE_DIR / "spine_cv_summary.csv"
df_out.to_csv(csv_path)
print(f"wrote {csv_path}")
print(df_out.round(3))

# ---- 表画像の作成 ----
fmt = df_out.copy()
for c in fmt.columns:
    fmt[c] = fmt[c].map(lambda v: f"{v:.3f}")

header = ["区間"] + [f"{g}\n{s}" for g, s in fmt.columns]
cell_text = [[label] + list(row.values) for label, row in fmt.iterrows()]
all_rows = [header] + cell_text

n_total_rows = len(all_rows)
n_total_cols = len(header)

fig_w = 1.35 * n_total_cols
fig_h = 0.6 * n_total_rows
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

table = ax.table(cellText=all_rows, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.0)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#444444")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(weight="bold")
    if c == 0:
        cell.set_text_props(ha="left")

fig.tight_layout()

png_path = BASE_DIR / "spine_cv_summary.png"
eps_path = BASE_DIR / "spine_cv_summary.eps"
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(eps_path, bbox_inches="tight")
print(f"wrote {png_path}")
print(f"wrote {eps_path}")
