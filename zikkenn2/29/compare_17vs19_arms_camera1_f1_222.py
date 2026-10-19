# -*- coding: utf-8 -*-
"""
confidence_by_region_frames.csv(29番)から、Camera1・frame1〜222・両腕(肩・肘・手首、
計6点)のみを抽出し、17点(拡張前)・19点(拡張後)モデルの信頼度を領域内/領域外で比較する。

出力: zikkenn2/29/
  compare_17vs19_arms_camera1_f1-222.csv
  compare_17vs19_arms_camera1_f1-222.png/.eps
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
CAM = 1
FRAME_RANGE = (1, 222)

ARM_IDX = {"19pt": [1, 2, 3, 4, 5, 6], "17pt": [5, 6, 7, 8, 9, 10]}

df = pd.read_csv(OUT / "confidence_by_region_frames.csv")
df = df[(df.camera == CAM) & (df.frame >= FRAME_RANGE[0]) & (df.frame <= FRAME_RANGE[1])]
mask = df.apply(lambda r: r["kp_idx"] in ARM_IDX[r["model"]], axis=1)
arms_df = df[mask].copy()
arms_df["region"] = arms_df["in_avatar"].map({True: "領域内(アバター)", False: "領域外(背景)"})

summary = arms_df.groupby(["model", "region"])["score"].agg(
    n="count", 最大値="max", 最小値="min", 平均値="mean"
).reset_index()
print(summary.to_string(index=False))

piv = summary.pivot_table(index=["region"], columns="model", values=["最大値", "最小値", "平均値", "n"])
piv.columns = [f"{stat}_{model}" for stat, model in piv.columns]
piv = piv.reset_index()
for stat in ["最大値", "最小値", "平均値"]:
    piv[f"{stat}_差分(19pt-17pt)"] = piv[f"{stat}_19pt"] - piv[f"{stat}_17pt"]

col_order = ["region", "n_17pt", "n_19pt",
             "最大値_17pt", "最大値_19pt", "最大値_差分(19pt-17pt)",
             "最小値_17pt", "最小値_19pt", "最小値_差分(19pt-17pt)",
             "平均値_17pt", "平均値_19pt", "平均値_差分(19pt-17pt)"]
piv = piv[col_order]
out_csv = OUT / "compare_17vs19_arms_camera1_f1-222.csv"
piv.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nwrote {out_csv.name}")
print(piv.to_string(index=False))

# ---------------- グラフ ----------------
labels = list(piv["region"])
x = range(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 6))
ax.bar([i - width / 2 for i in x], piv["平均値_17pt"], width, label="17点(拡張前)", color="tab:blue")
ax.bar([i + width / 2 for i in x], piv["平均値_19pt"], width, label="19点(拡張後)", color="tab:red")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("信頼度 平均値")
ax.set_title(f"2D信頼度(両腕のみ): 17点 vs 19点\nCamera{CAM}, frame{FRAME_RANGE[0]}-{FRAME_RANGE[1]}")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
for i in x:
    v17 = piv["平均値_17pt"].iloc[i]
    v19 = piv["平均値_19pt"].iloc[i]
    ax.text(i - width / 2, v17, f"{v17:.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + width / 2, v19, f"{v19:.3f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
out_png = OUT / "compare_17vs19_arms_camera1_f1-222.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(OUT / "compare_17vs19_arms_camera1_f1-222.eps", bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_png.name} / .eps")
