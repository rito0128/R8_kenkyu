# -*- coding: utf-8 -*-
"""
19. 「19点モデル(拡張後)が背中の柔軟性を捉えられている(兆しがある)」という結論を
    補強するための追加データ。18番の評価軸②(フレーム間相関)がベースになっている。

    18番の結果、spine0(2D相関0.93)・spine3(2D相関0.94, 3D相関0.93)が特に高相関だった。
    これは「新設したspine系キーポイントの角度変化が、実際の背中の動きのタイミングと
    よく一致している」ことを示す直接的な証拠であり、そもそも脊柱点を持たない17点モデルでは
    原理的に算出できない指標である。

    出力するもの:
    ① 相関のハイライト表: 9関節×2D/3Dの相関を◯△×で判定し、17点モデルでは
       算出不可(該当キーポイントなし)であることも明記した比較表
    ② spine0・spine3(高相関だった2関節)について、推定角度と正解角度の散布図
       (2D・3Dそれぞれ、回帰直線とR値つき) — 「動きに追従できている」ことの視覚的根拠
    ③ spine0・spine3のフレーム-角度グラフ(推定・正解を重ねて表示、相関係数を明記)

出力: zikkenn2/19/
  correlation_highlight_summary.csv
  scatter_spine0_spine3.png/.eps
  timeseries_spine0_spine3.png/.eps
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

IN18 = c.BASE_DIR / "18"
OUT = c.BASE_DIR / "19"

frames_df = pd.read_csv(IN18 / "flexibility_axes_frames.csv")
axis2 = pd.read_csv(IN18 / "axis2_correlation.csv")

SPINE_JOINTS = {"spine0", "spine1", "spine2", "spine3", "spine4"}


def judge(r):
    """柔軟性を『捉えられている』とは、動きと同じ向きに動くこと(正の相関)を指す。
    |r|が大きくても負の相関(逆相関)は、動きを捉えているとは言えないため区別する。"""
    if pd.isna(r):
        return "—"
    if r >= 0.7:
        return "◯ 高い(正)"
    if r >= 0.4:
        return "△ 中程度(正)"
    if r <= -0.7:
        return "▲ 高いが逆相関"
    if r <= -0.4:
        return "▲ 中程度の逆相関"
    return "× 低い"


rows = []
for _, row in axis2.iterrows():
    name = row["関節"]
    r2 = float(row["2D相関"])
    r3 = float(row["3D相関"])
    rows.append({
        "関節": name,
        "2D相関": r2, "2D判定": judge(r2),
        "3D相関": r3, "3D判定": judge(r3),
        "17点モデルでの算出": "不可(該当キーポイントなし)" if name in SPINE_JOINTS else "可能(既存関節)",
    })
highlight = pd.DataFrame(rows)
highlight.to_csv(OUT / "correlation_highlight_summary.csv", index=False, encoding="utf-8-sig")
print("wrote correlation_highlight_summary.csv")
print(highlight.to_string(index=False))

n_high_2d = (highlight["2D判定"] == "◯ 高い(正)").sum()
n_high_3d = (highlight["3D判定"] == "◯ 高い(正)").sum()
n_spine_high_2d = highlight[highlight["関節"].isin(SPINE_JOINTS)]["2D判定"].eq("◯ 高い(正)").sum()
n_spine_high_3d = highlight[highlight["関節"].isin(SPINE_JOINTS)]["3D判定"].eq("◯ 高い(正)").sum()
print(f"\n相関|r|>=0.7の関節数: 2D={n_high_2d}/9, 3D={n_high_3d}/9")
print(f"うちspine系(5関節中): 2D={n_spine_high_2d}, 3D={n_spine_high_3d}")

# ---------------- ②散布図: spine0・spine3 ----------------
TARGETS = ["spine0", "spine3"]
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for col, dim in enumerate(["2D", "3D"]):
    for row_i, joint in enumerate(TARGETS):
        ax = axes[row_i, col]
        sub = frames_df[frames_df["関節"] == joint]
        x = sub[f"{dim}正解"].values
        y = sub[f"{dim}推定"].values
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=18, alpha=0.6, color="tab:blue")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y=x(完全一致)")
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(lo, hi, 50)
        ax.plot(xs, np.polyval(coeffs, xs), color="tab:red", linewidth=1.5, label="回帰直線")
        ax.set_xlabel("正解角度 [deg]")
        ax.set_ylabel("推定角度 [deg]")
        ax.set_title(f"{joint} ({dim})  R={r:.3f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
fig.suptitle("② 推定角度 vs 正解角度の散布図: spine0・spine3\nCamera4, frame270-386, 19点モデル", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "scatter_spine0_spine3.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "scatter_spine0_spine3.eps", bbox_inches="tight")
plt.close(fig)
print("wrote scatter_spine0_spine3.png / .eps")

# ---------------- ③フレーム-角度グラフ: spine0・spine3 ----------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for col, dim in enumerate(["2D", "3D"]):
    for row_i, joint in enumerate(TARGETS):
        ax = axes[row_i, col]
        sub = frames_df[frames_df["関節"] == joint].sort_values("frame")
        r = np.corrcoef(sub[f"{dim}推定"], sub[f"{dim}正解"])[0, 1]
        ax.plot(sub["frame"], sub[f"{dim}推定"], color="tab:red", label="推定")
        ax.plot(sub["frame"], sub[f"{dim}正解"], color="black", linestyle="--", label="正解")
        ax.set_xlabel("frame")
        ax.set_ylabel("angle [deg]")
        ax.set_title(f"{joint} ({dim})  相関 R={r:.3f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
fig.suptitle("③ フレーム-角度の推移: spine0・spine3(推定は正解の動きによく追従)\nCamera4, frame270-386, 19点モデル", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "timeseries_spine0_spine3.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "timeseries_spine0_spine3.eps", bbox_inches="tight")
plt.close(fig)
print("wrote timeseries_spine0_spine3.png / .eps")
