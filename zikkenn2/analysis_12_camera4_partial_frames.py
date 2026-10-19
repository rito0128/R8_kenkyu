# -*- coding: utf-8 -*-
"""
12. Camera4限定・フレーム範囲限定(270〜386)の解析
    19点拡張後モデル。
    ※ 当初は34〜100, 270〜386の2区間で実施したが、34〜100は2D推定の破綻(frame55〜65付近で
       複数区間・角度が急激に振動)が大きく評価に適さないため、270〜386のみに絞って再算出した。

① 区間の長さ: head-spine0, spine0-spine1, spine1-spine2, spine2-spine3, spine3-spine4, spine4-spine5
   の6区間長(19点拡張後モデル)に加え、17点(拡張前)モデルの推定結果から
   「肩-腰」の長さ(2D=左右肩中点-左右股関節中点、3D=H36M17のLShoulder/RShoulder中点-Hip)を
   算出し追加した(正解データは19点GTの肩・股関節点から同様に算出)。
   ※ ユーザー指定ではspine0-spine1が抜けていたが、これまでの恒常性評価と同じ6区間の
      チェーン(head-spine0-spine1-...-spine5)として揃えるため、spine0-spine1も含めて算出した。
   ※ 17点モデルにはspine0〜spine5に相当する点が無いため、6区間そのものは計算できない。
      「肩-腰」はその制約下で計算できる、19点モデル側と直接比較可能な唯一に近い体幹長。

② 関節角度: spine0〜spine4(隣接2点から3点角度を計算できる5関節)を各フレームで算出
   (2D推定・2D正解・3D推定・3D正解)。
   ※ head・spine5はチェーンの両端で隣接点が片側にしかないため、単独では3点角度を
      計算できない(これまでの柔軟性評価と同じ制約。headはspine0の、spine5はspine4の
      角度計算の"腕"としてのみ使用される)。そのため角度算出の対象は spine0〜spine4 とした。

出力: zikkenn2/12/
  lengths_camera4_partial.csv       ①のフレームごとの生データ
  lengths_camera4_partial.png/.eps  ①の全区間まとめグラフ(1枚、左軸=2D[px]、右軸=3D[normalized])
  lengths_by_segment/{区間名}.png/.eps  ①の区間ごとの個別グラフ(2D推定/2D正解/3D推定/3D正解を1枚に)
  angles_camera4_partial.csv        ②のフレームごとの生データ
  angles_camera4_partial.png/.eps   ②の全関節まとめグラフ(1枚、左軸=2D[deg]、右軸=3D[deg])
  angles_by_joint/{関節名}.png/.eps     ②の関節ごとの個別グラフ(2D推定/2D正解/3D推定/3D正解を1枚に)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "12"
CAM = 4
FRAME_RANGES = [(270, 386)]


def in_ranges(frames):
    mask = np.zeros(len(frames), dtype=bool)
    for lo, hi in FRAME_RANGES:
        mask |= (frames >= lo) & (frames <= hi)
    return mask


def angle3pt(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos = (v1 * v2).sum(axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


RANGE_LABEL = ", ".join(f"{lo}-{hi}" for lo, hi in FRAME_RANGES)


def plot_ranges(ax, frames, values, **kwargs):
    """34-100と270-386の間を線でつながないよう、範囲ごとに分けてプロットする"""
    label = kwargs.pop("label", None)
    for i, (lo, hi) in enumerate(FRAME_RANGES):
        m = (frames >= lo) & (frames <= hi)
        ax.plot(frames[m], values[m], label=(label if i == 0 else None), **kwargs)


def plot_single(name, series, ylabel2d, ylabel3d, out_dir, title):
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    plot_ranges(ax, frames, series["2D推定"], color="tab:red", linestyle="-", linewidth=1.6, label="2D推定")
    plot_ranges(ax, frames, series["2D正解"], color="tab:red", linestyle="--", linewidth=1.4, alpha=0.8, label="2D正解")
    plot_ranges(ax2, frames, series["3D推定"], color="tab:blue", linestyle="-", linewidth=1.6, label="3D推定")
    plot_ranges(ax2, frames, series["3D正解"], color="tab:blue", linestyle="--", linewidth=1.4, alpha=0.8, label="3D正解")
    ax.set_xlabel("frame")
    ax.set_ylabel(ylabel2d, color="tab:red")
    ax2.set_ylabel(ylabel3d, color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    h1, l1_ = ax.get_legend_handles_labels()
    h2, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1_ + l2_, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.eps", bbox_inches="tight")
    plt.close(fig)


pred2d = c.load_2d_pred("19pt", CAM)
gt2d = c.load_2d_gt(CAM)
pred3d = c.load_3d_pred("19pt", CAM)
gt3d = c.load_3d_gt(CAM)
k17_2d = c.load_2d_pred("17pt", CAM)
k17_3d = c.load_3d_pred("17pt", CAM)

frames_all = pred2d["frame"].values
mask = in_ranges(frames_all)
frames = frames_all[mask]

# ---------------- ① 区間長 ----------------
SHOULDER_HIP_LABEL = "shoulder-hip(17点)"
LENGTH_LABELS = list(c.SPINE_CHAIN_LABELS) + [SHOULDER_HIP_LABEL]

length_rows = []
length_series = {}
for (a, b), label in zip(c.SPINE_CHAIN, c.SPINE_CHAIN_LABELS):
    l2p = c.dist(c.pred_xy(pred2d, a), c.pred_xy(pred2d, b))[mask]
    l2g = c.dist(c.gt_xy(gt2d, a), c.gt_xy(gt2d, b))[mask]
    l3p = c.dist(c.pred_xyz(pred3d, a), c.pred_xyz(pred3d, b))[mask]
    l3g = c.dist(c.gt_xyz(gt3d, a), c.gt_xyz(gt3d, b))[mask]
    length_series[label] = {"2D推定": l2p, "2D正解": l2g, "3D推定": l3p, "3D正解": l3g}
    for fr, v2p, v2g, v3p, v3g in zip(frames, l2p, l2g, l3p, l3g):
        length_rows.append({"camera": CAM, "frame": fr, "区間": label,
                             "2D推定": v2p, "2D正解": v2g, "3D推定": v3p, "3D正解": v3g})

# ---- 肩-腰(17点モデル、正解は19点GTの肩・股関節点から算出) ----
# 17点2D(COCO17): 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
k17_shoulder_mid = (c.pred_xy(k17_2d, 5) + c.pred_xy(k17_2d, 6)) / 2
k17_hip_mid = (c.pred_xy(k17_2d, 11) + c.pred_xy(k17_2d, 12)) / 2
sh2p = c.dist(k17_shoulder_mid, k17_hip_mid)[mask]
# 17点3D(H36M17): 11=LShoulder, 14=RShoulder, 0=Hip(root)
k17_shoulder_mid3d = (c.pred_xyz(k17_3d, 11) + c.pred_xyz(k17_3d, 14)) / 2
sh3p = c.dist(k17_shoulder_mid3d, c.pred_xyz(k17_3d, 0))[mask]
# 正解(19点GT): 1=left_shoulder, 2=right_shoulder, 7=left_hip, 8=right_hip
gt_shoulder_mid2d = (c.gt_xy(gt2d, 1) + c.gt_xy(gt2d, 2)) / 2
gt_hip_mid2d = (c.gt_xy(gt2d, 7) + c.gt_xy(gt2d, 8)) / 2
sh2g = c.dist(gt_shoulder_mid2d, gt_hip_mid2d)[mask]
gt_shoulder_mid3d = (c.gt_xyz(gt3d, 1) + c.gt_xyz(gt3d, 2)) / 2
gt_hip_mid3d = (c.gt_xyz(gt3d, 7) + c.gt_xyz(gt3d, 8)) / 2
sh3g = c.dist(gt_shoulder_mid3d, gt_hip_mid3d)[mask]

length_series[SHOULDER_HIP_LABEL] = {"2D推定": sh2p, "2D正解": sh2g, "3D推定": sh3p, "3D正解": sh3g}
for fr, v2p, v2g, v3p, v3g in zip(frames, sh2p, sh2g, sh3p, sh3g):
    length_rows.append({"camera": CAM, "frame": fr, "区間": SHOULDER_HIP_LABEL,
                         "2D推定": v2p, "2D正解": v2g, "3D推定": v3p, "3D正解": v3g})

pd.DataFrame(length_rows).to_csv(OUT / "lengths_camera4_partial.csv", index=False, encoding="utf-8-sig")
print("wrote lengths_camera4_partial.csv")

fig, ax = plt.subplots(figsize=(14, 7))
ax2 = ax.twinx()
cmap = plt.get_cmap("tab10")
for i, label in enumerate(LENGTH_LABELS):
    color = cmap(i)
    s = length_series[label]
    plot_ranges(ax, frames, s["2D推定"], color=color, linestyle="-", linewidth=1.6, label=f"{label} 2D推定")
    plot_ranges(ax, frames, s["2D正解"], color=color, linestyle="--", linewidth=1.2, alpha=0.7, label=f"{label} 2D正解")
    plot_ranges(ax2, frames, s["3D推定"], color=color, linestyle=":", linewidth=1.6, label=f"{label} 3D推定")
    plot_ranges(ax2, frames, s["3D正解"], color=color, linestyle="-.", linewidth=1.2, alpha=0.7, label=f"{label} 3D正解")

ax.set_xlabel("frame")
ax.set_ylabel("2D length [px]")
ax2.set_ylabel("3D length [normalized]")
ax.set_title(f"① 区間長のフレーム推移 Camera{CAM} (frame {RANGE_LABEL})\n(shoulder-hipのみ17点モデル、他は19点モデル)")
ax.grid(True, alpha=0.3)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(OUT / "lengths_camera4_partial.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "lengths_camera4_partial.eps", bbox_inches="tight")
plt.close(fig)
print("wrote lengths_camera4_partial.png / .eps")

# ---- ①区間ごとの個別グラフ ----
lengths_dir = OUT / "lengths_by_segment"
for label in LENGTH_LABELS:
    plot_single(
        label, length_series[label], "2D length [px]", "3D length [normalized]", lengths_dir,
        f"① 区間長 Camera{CAM} (frame {RANGE_LABEL}): {label}",
    )
print(f"wrote lengths_by_segment/*.png (.eps) x{len(LENGTH_LABELS)}")

# ---------------- ② 関節角度(spine0〜spine4) ----------------
ANGLE_JOINTS = {
    "spine0": (0, 13, 14),
    "spine1": (13, 14, 15),
    "spine2": (14, 15, 16),
    "spine3": (15, 16, 17),
    "spine4": (16, 17, 18),
}

angle_rows = []
angle_series = {}
for name, (a, b, cidx) in ANGLE_JOINTS.items():
    a2p = angle3pt(c.pred_xy(pred2d, a), c.pred_xy(pred2d, b), c.pred_xy(pred2d, cidx))[mask]
    a2g = angle3pt(c.gt_xy(gt2d, a), c.gt_xy(gt2d, b), c.gt_xy(gt2d, cidx))[mask]
    a3p = angle3pt(c.pred_xyz(pred3d, a), c.pred_xyz(pred3d, b), c.pred_xyz(pred3d, cidx))[mask]
    a3g = angle3pt(c.gt_xyz(gt3d, a), c.gt_xyz(gt3d, b), c.gt_xyz(gt3d, cidx))[mask]
    angle_series[name] = {"2D推定": a2p, "2D正解": a2g, "3D推定": a3p, "3D正解": a3g}
    for fr, v2p, v2g, v3p, v3g in zip(frames, a2p, a2g, a3p, a3g):
        angle_rows.append({"camera": CAM, "frame": fr, "関節": name,
                            "2D推定": v2p, "2D正解": v2g, "3D推定": v3p, "3D正解": v3g})

pd.DataFrame(angle_rows).to_csv(OUT / "angles_camera4_partial.csv", index=False, encoding="utf-8-sig")
print("wrote angles_camera4_partial.csv")

fig, ax = plt.subplots(figsize=(14, 7))
ax2 = ax.twinx()
for i, name in enumerate(ANGLE_JOINTS.keys()):
    color = cmap(i)
    s = angle_series[name]
    plot_ranges(ax, frames, s["2D推定"], color=color, linestyle="-", linewidth=1.6, label=f"{name} 2D推定")
    plot_ranges(ax, frames, s["2D正解"], color=color, linestyle="--", linewidth=1.2, alpha=0.7, label=f"{name} 2D正解")
    plot_ranges(ax2, frames, s["3D推定"], color=color, linestyle=":", linewidth=1.6, label=f"{name} 3D推定")
    plot_ranges(ax2, frames, s["3D正解"], color=color, linestyle="-.", linewidth=1.2, alpha=0.7, label=f"{name} 3D正解")

ax.set_xlabel("frame")
ax.set_ylabel("2D angle [deg]")
ax2.set_ylabel("3D angle [deg]")
ax.set_title(f"② 関節角度のフレーム推移 Camera{CAM} (frame {RANGE_LABEL})\n(head・spine5はチェーン両端のため角度算出対象外)")
ax.grid(True, alpha=0.3)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(OUT / "angles_camera4_partial.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "angles_camera4_partial.eps", bbox_inches="tight")
plt.close(fig)
print("wrote angles_camera4_partial.png / .eps")

# ---- ②関節ごとの個別グラフ ----
angles_dir = OUT / "angles_by_joint"
for name in ANGLE_JOINTS.keys():
    plot_single(
        name, angle_series[name], "2D angle [deg]", "3D angle [deg]", angles_dir,
        f"② 関節角度 Camera{CAM} (frame {RANGE_LABEL}): {name}",
    )
print(f"wrote angles_by_joint/*.png (.eps) x{len(ANGLE_JOINTS)}")
