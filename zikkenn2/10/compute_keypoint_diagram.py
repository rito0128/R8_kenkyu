# -*- coding: utf-8 -*-
"""
19点キーポイントの「インデックス ⇔ 名称」対応を人体図として可視化する。

キーポイント名(0〜18)は for_test_HRNet/keypoint_common.py と
for_test_motionbert/.../keypoint_common.py で共通(19点: 体幹12点+spine0〜5)。
ただしKEYPOINT_PAIRS(接続定義)は2D/3Dで異なり、3D側の方が体幹の閉じた
骨格(肩-肩、腰-腰、肩-腰)を含み完全なので、本図の骨格線は3D側の定義を採用した。

座標は実データではなく、対応関係を見やすくするための模式的な人体配置。

出力: keypoint_diagram.png / .eps
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

KEYPOINT_NAMES = [
    "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle", "spine0", "spine1", "spine2",
    "spine3", "spine4", "spine5",
]

# 模式的な座標(x: 左右, y: 上下)。実データの座標ではない。
POS = {
    0: (0.0, 10.0),    # head
    1: (-1.5, 8.7),    # left_shoulder
    2: (1.5, 8.7),     # right_shoulder
    3: (-2.3, 7.0),    # left_elbow
    4: (2.3, 7.0),     # right_elbow
    5: (-2.6, 5.3),    # left_wrist
    6: (2.6, 5.3),     # right_wrist
    7: (-1.0, 4.8),    # left_hip
    8: (1.0, 4.8),     # right_hip
    9: (-1.1, 2.6),    # left_knee
    10: (1.1, 2.6),    # right_knee
    11: (-1.2, 0.3),   # left_ankle
    12: (1.2, 0.3),    # right_ankle
    13: (0.0, 8.3),    # spine0 (首寄り)
    14: (0.0, 7.6),    # spine1
    15: (0.0, 6.9),    # spine2
    16: (0.0, 6.2),    # spine3
    17: (0.0, 5.5),    # spine4
    18: (0.0, 4.9),    # spine5 (腰寄り)
}

# 3D(MotionBert)側 keypoint_common.py の KEYPOINT_PAIRS
EDGES = [
    (11, 9), (9, 7), (12, 10), (10, 8), (7, 8),
    (1, 7), (2, 8), (1, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
    (0, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
]

LIMB_IDX = set(range(0, 13))
SPINE_IDX = set(range(13, 19))

fig, ax = plt.subplots(figsize=(8, 10))

for a, b in EDGES:
    xa, ya = POS[a]
    xb, yb = POS[b]
    ax.plot([xa, xb], [ya, yb], color="gray", linewidth=2, zorder=1)

xs_limb = [POS[i][0] for i in LIMB_IDX]
ys_limb = [POS[i][1] for i in LIMB_IDX]
ax.scatter(xs_limb, ys_limb, s=260, color="tab:blue", zorder=2, edgecolor="white", linewidth=1)

xs_spine = [POS[i][0] for i in SPINE_IDX]
ys_spine = [POS[i][1] for i in SPINE_IDX]
ax.scatter(xs_spine, ys_spine, s=260, color="tab:orange", zorder=2, edgecolor="white", linewidth=1)

for idx, name in enumerate(KEYPOINT_NAMES):
    x, y = POS[idx]
    ax.annotate(str(idx), (x, y), ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=3)
    dx = -1 if x < 0 else (1 if x > 0 else 0)
    ha = "right" if dx < 0 else ("left" if dx > 0 else "center")
    offset_x = dx * 0.35 if dx != 0 else 0.0
    offset_y = 0.28 if idx == 0 else 0.0
    ax.annotate(f"{idx}: {name}", (x + offset_x, y + offset_y), ha=ha if dx != 0 else "center",
                va="bottom" if idx == 0 else "center", fontsize=10)

ax.scatter([], [], s=200, color="tab:blue", label="体幹・四肢 (0〜12)")
ax.scatter([], [], s=200, color="tab:orange", label="脊柱 spine0〜spine5 (13〜18)")
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

ax.set_title("キーポイント インデックス ⇔ 名称 対応図 (19点)\n(骨格線は3D側KEYPOINT_PAIRSの定義に基づく模式図)", fontsize=13)
ax.set_xlim(-4.2, 4.8)
ax.set_ylim(-0.8, 11.0)
ax.set_aspect("equal")
ax.axis("off")

fig.tight_layout()
fig.savefig("keypoint_diagram.png", dpi=150, bbox_inches="tight")
fig.savefig("keypoint_diagram.eps", bbox_inches="tight")
plt.close(fig)
print("wrote keypoint_diagram.png / .eps")
