# -*- coding: utf-8 -*-
"""
confidence_by_region_frames.csv(29番)から両腕(肩・肘・手首、計6点)のみを抽出し、
17点(拡張前)・19点(拡張後)モデルの信頼度を領域内/領域外・カメラ別に比較する。

19点のkp_idx: left_shoulder=1, right_shoulder=2, left_elbow=3, right_elbow=4,
              left_wrist=5, right_wrist=6
17点(COCO)のkp_idx: left_shoulder=5, right_shoulder=6, left_elbow=7, right_elbow=8,
              left_wrist=9, right_wrist=10

出力: zikkenn2/29/
  compare_17vs19_arms.csv
  compare_17vs19_arms.png/.eps
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent

ARM_IDX = {"19pt": [1, 2, 3, 4, 5, 6], "17pt": [5, 6, 7, 8, 9, 10]}

df = pd.read_csv(OUT / "confidence_by_region_frames.csv")
mask = df.apply(lambda r: r["kp_idx"] in ARM_IDX[r["model"]], axis=1)
arms_df = df[mask].copy()
arms_df["region"] = arms_df["in_avatar"].map({True: "領域内(アバター)", False: "領域外(背景)"})

summary = arms_df.groupby(["model", "camera", "region"])["score"].agg(
    n="count", 最大値="max", 最小値="min", 平均値="mean"
).reset_index()
print(summary.to_string(index=False))

piv = summary.pivot_table(index=["camera", "region"], columns="model", values=["最大値", "最小値", "平均値", "n"])
piv.columns = [f"{stat}_{model}" for stat, model in piv.columns]
piv = piv.reset_index()
for stat in ["最大値", "最小値", "平均値"]:
    piv[f"{stat}_差分(19pt-17pt)"] = piv[f"{stat}_19pt"] - piv[f"{stat}_17pt"]

col_order = ["camera", "region", "n_17pt", "n_19pt",
             "最大値_17pt", "最大値_19pt", "最大値_差分(19pt-17pt)",
             "最小値_17pt", "最小値_19pt", "最小値_差分(19pt-17pt)",
             "平均値_17pt", "平均値_19pt", "平均値_差分(19pt-17pt)"]
piv = piv[col_order]
piv.to_csv(OUT / "compare_17vs19_arms.csv", index=False, encoding="utf-8-sig")
print("\nwrote compare_17vs19_arms.csv")
print(piv.to_string(index=False))

# ---------------- グラフ ----------------
labels = [f"Camera{r.camera}\n{r.region}" for r in piv.itertuples()]
x = range(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([i - width / 2 for i in x], piv["平均値_17pt"], width, label="17点(拡張前)", color="tab:blue")
ax.bar([i + width / 2 for i in x], piv["平均値_19pt"], width, label="19点(拡張後)", color="tab:red")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("信頼度 平均値")
ax.set_title("2D信頼度の平均値比較(両腕のみ): 17点(拡張前) vs 19点(拡張後)")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
for i in x:
    v17 = piv["平均値_17pt"].iloc[i]
    v19 = piv["平均値_19pt"].iloc[i]
    ax.text(i - width / 2, v17, f"{v17:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + width / 2, v19, f"{v19:.3f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "compare_17vs19_arms.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "compare_17vs19_arms.eps", bbox_inches="tight")
plt.close(fig)
print("wrote compare_17vs19_arms.png / .eps")
