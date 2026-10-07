# -*- coding: utf-8 -*-
"""
13. 17点(拡張前)モデル vs 19点(拡張後)モデル: 背中の動きへの追従比較
    Camera4, frame270〜386(12番と同じ安定区間)。12番のlengths_camera4_partial.csvを利用。

目的: 「背骨にキーポイントを追加したことで、背中の構造をより正確に推定できているか」を、
     推定結果と正解データの相対誤差(=正解に対してどれだけズレているか)で比較する。
     17点モデルは肩-腰の1本の直線距離しか持たないのに対し、19点モデルはhead-spine0〜
     spine4-spine5の6区間に分解できる。

【重要な注意】
  3Dの推定値は正規化されたスケール、正解データは実寸(スケールが大きく異なる)であるため、
  素の相対誤差(推定-正解)/正解を計算すると200〜400%という非現実的な値になり、比較に使えない。
  そのため3Dは「推定・正解をそれぞれ自身の平均値で正規化してから比較する誤差」を別途算出した。
  2Dは単位(px)が共通のため、素の相対誤差をそのまま使用できる。

  実際に計算した結果、2D相対誤差では19点モデルの各区間がおおむね17点の肩-腰(11.1%)より
  誤差が小さい(-1.5%〜8.4%)一方、3D(正規化後)やフレーム間相関では区間によって優劣が入れ替わり、
  「19点モデルが常に17点より優れている」と一貫して言える結果ではなかった。
  グラフ・表はその実際の数値をそのまま示している。

出力: zikkenn2/13/
  relative_error_summary.csv        区間ごとの2D相対誤差・3D正規化誤差・相関の要約表
  relative_error_2d.png/.eps        2D相対誤差(%)のフレーム推移(1枚、全区間+17点肩-腰)
  normalized_3d.png/.eps            3D正規化(平均1.0基準)の推定・正解 比較(6区間+17点肩-腰、小コマ)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

IN12 = c.BASE_DIR / "12"
OUT = c.BASE_DIR / "13"

df = pd.read_csv(IN12 / "lengths_camera4_partial.csv")
LABELS = list(c.SPINE_CHAIN_LABELS) + ["shoulder-hip(17点)"]
frames = df[df["区間"] == LABELS[0]]["frame"].values

# ---------------- 相対誤差の算出 ----------------
df["rel_err_2d_pct"] = (df["2D推定"] - df["2D正解"]) / df["2D正解"] * 100

summary_rows = []
norm_series = {}
for label in LABELS:
    sub = df[df["区間"] == label].sort_values("frame")
    p3 = sub["3D推定"].values
    g3 = sub["3D正解"].values
    p3n = p3 / p3.mean()
    g3n = g3 / g3.mean()
    norm_series[label] = {"推定": p3n, "正解": g3n}
    summary_rows.append({
        "区間": label,
        "2D相対誤差 平均(%)": sub["rel_err_2d_pct"].mean(),
        "2D相対誤差 |平均|(%)": sub["rel_err_2d_pct"].abs().mean(),
        "3D正規化誤差 平均|差|(%)": np.abs(p3n - g3n).mean() * 100,
        "3D相関(推定vs正解)": np.corrcoef(p3, g3)[0, 1],
    })

summary_df = pd.DataFrame(summary_rows)
for col in summary_df.columns[1:]:
    summary_df[col] = summary_df[col].map(c.sig4)
summary_df.to_csv(OUT / "relative_error_summary.csv", index=False, encoding="utf-8-sig")
print("wrote relative_error_summary.csv")
print(summary_df.to_string(index=False))

# ---------------- グラフ1: 2D相対誤差のフレーム推移(1枚) ----------------
fig, ax = plt.subplots(figsize=(12, 6))
cmap = plt.get_cmap("tab10")
for i, label in enumerate(LABELS):
    sub = df[df["区間"] == label].sort_values("frame")
    style = dict(color=cmap(i), linewidth=2.2 if label == "shoulder-hip(17点)" else 1.4)
    ax.plot(sub["frame"], sub["rel_err_2d_pct"], label=label, **style)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("frame")
ax.set_ylabel("2D相対誤差 (推定-正解)/正解 [%]")
ax.set_title("① 2D相対誤差の比較: 19点モデル(6区間) vs 17点モデル(肩-腰)\nCamera4, frame270-386")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "relative_error_2d.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "relative_error_2d.eps", bbox_inches="tight")
plt.close(fig)
print("wrote relative_error_2d.png / .eps")

# ---------------- グラフ2: 3D正規化(平均1.0基準)の推定・正解 比較 ----------------
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, label in zip(axes.flat, LABELS):
    s = norm_series[label]
    ax.plot(frames, s["推定"], color="tab:red", label="推定(平均で正規化)")
    ax.plot(frames, s["正解"], color="black", linestyle="--", label="正解(平均で正規化)")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("frame")
    ax.set_ylabel("length / mean(length)")
    ax.grid(True, alpha=0.3)
axes.flat[0].legend(fontsize=8)
for ax in axes.flat[len(LABELS):]:
    ax.axis("off")
fig.suptitle("② 3D区間長(自身の平均で正規化): 19点モデル(6区間) vs 17点モデル(肩-腰)\nCamera4, frame270-386", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "normalized_3d.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "normalized_3d.eps", bbox_inches="tight")
plt.close(fig)
print("wrote normalized_3d.png / .eps")
