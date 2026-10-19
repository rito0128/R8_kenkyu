# -*- coding: utf-8 -*-
"""
confidence_by_region_summary.csv(29番)を使い、拡張前(17点)・拡張後(19点)モデルの
信頼度を領域内/領域外・カメラ別に直接比較する。

出力: zikkenn2/29/
  compare_17vs19.csv    17点・19点を横に並べた比較表(差分付き)
  compare_17vs19.png/.eps  平均値の比較棒グラフ
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent

df = pd.read_csv(OUT / "confidence_by_region_summary.csv")

piv = df.pivot_table(index=["camera", "region"], columns="model", values=["最大値", "最小値", "平均値", "n"])
piv.columns = [f"{stat}_{model}" for stat, model in piv.columns]
piv = piv.reset_index()
for stat in ["最大値", "最小値", "平均値"]:
    piv[f"{stat}_差分(19pt-17pt)"] = piv[f"{stat}_19pt"] - piv[f"{stat}_17pt"]

col_order = ["camera", "region", "n_17pt", "n_19pt",
             "最大値_17pt", "最大値_19pt", "最大値_差分(19pt-17pt)",
             "最小値_17pt", "最小値_19pt", "最小値_差分(19pt-17pt)",
             "平均値_17pt", "平均値_19pt", "平均値_差分(19pt-17pt)"]
piv = piv[col_order]
piv.to_csv(OUT / "compare_17vs19.csv", index=False, encoding="utf-8-sig")
print("wrote compare_17vs19.csv")
print(piv.to_string(index=False))

# ---------------- グラフ: 平均値の比較棒グラフ ----------------
labels = [f"Camera{r.camera}\n{r.region}" for r in piv.itertuples()]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width / 2, piv["平均値_17pt"], width, label="17点(拡張前)", color="tab:blue")
ax.bar(x + width / 2, piv["平均値_19pt"], width, label="19点(拡張後)", color="tab:red")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("信頼度 平均値")
ax.set_title("2D信頼度の平均値比較: 17点(拡張前) vs 19点(拡張後)")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
for i in range(len(piv)):
    v17 = piv["平均値_17pt"].iloc[i]
    v19 = piv["平均値_19pt"].iloc[i]
    ax.text(i - width / 2, v17, f"{v17:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + width / 2, v19, f"{v19:.3f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "compare_17vs19.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "compare_17vs19.eps", bbox_inches="tight")
plt.close(fig)
print("wrote compare_17vs19.png / .eps")
