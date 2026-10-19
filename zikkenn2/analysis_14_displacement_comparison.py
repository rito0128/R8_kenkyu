# -*- coding: utf-8 -*-
"""
14. 19点モデルの6区間の「フレーム間の差」を合計した値 と、17点モデルの肩-腰の長さの
    フレーム間の変位を比較する。Camera4, frame270〜386, 推定結果のみ(2D・3Dそれぞれ)。
    12番のlengths_camera4_partial.csvを利用。

    19点側: 各区間(head-spine0〜spine4-spine5)のΔlength(t)=length(t)-length(t-1)を
            6区間ぶん合計した値。これは「6区間の長さの合計(=首から腰までの経路長)」の
            フレーム間差分と数学的に同じ値になる(Σ_区間 Δlen_区間 = Δ(Σ_区間 len_区間))。
    17点側: 肩-腰の1本の直線距離のΔlength(t)。

    2Dのグラフのみ、比較の基準として正解データ(2D正解)側の同じ量(19点区間合計Δ・17点肩腰Δ)も
    追加でプロットする(3Dは推定側とスケールが異なる正規化値のため、当初の方針通り推定結果のみ)。

出力: zikkenn2/14/
  displacement_comparison.csv        フレームごとの19点合計Δ・17点Δ(2D推定/2D正解/3D推定)
  displacement_comparison_2d.png/.eps  2D推定結果+正解データでのグラフ(1枚)
  displacement_comparison_3d.png/.eps  3D推定結果でのグラフ(1枚)
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
OUT = c.BASE_DIR / "14"

df = pd.read_csv(IN12 / "lengths_camera4_partial.csv")

pivot_2d = df.pivot(index="frame", columns="区間", values="2D推定").sort_index()
pivot_3d = df.pivot(index="frame", columns="区間", values="3D推定").sort_index()
pivot_2d_gt = df.pivot(index="frame", columns="区間", values="2D正解").sort_index()

total19_2d = pivot_2d[list(c.SPINE_CHAIN_LABELS)].sum(axis=1)
total19_3d = pivot_3d[list(c.SPINE_CHAIN_LABELS)].sum(axis=1)
total19_2d_gt = pivot_2d_gt[list(c.SPINE_CHAIN_LABELS)].sum(axis=1)
sh17_2d = pivot_2d["shoulder-hip(17点)"]
sh17_3d = pivot_3d["shoulder-hip(17点)"]
sh17_2d_gt = pivot_2d_gt["shoulder-hip(17点)"]

delta19_2d = total19_2d.diff()
delta19_3d = total19_3d.diff()
delta17_2d = sh17_2d.diff()
delta17_3d = sh17_3d.diff()
delta19_2d_gt = total19_2d_gt.diff()
delta17_2d_gt = sh17_2d_gt.diff()

out_df = pd.DataFrame({
    "frame": pivot_2d.index,
    "19点区間合計Δ_2D推定": delta19_2d.values,
    "17点肩腰Δ_2D推定": delta17_2d.values,
    "19点区間合計Δ_2D正解": delta19_2d_gt.values,
    "17点肩腰Δ_2D正解": delta17_2d_gt.values,
    "19点区間合計Δ_3D推定": delta19_3d.values,
    "17点肩腰Δ_3D推定": delta17_3d.values,
}).dropna().reset_index(drop=True)
out_df.to_csv(OUT / "displacement_comparison.csv", index=False, encoding="utf-8-sig")
print("wrote displacement_comparison.csv")
print(out_df.describe().to_string())


def plot_delta(dim, series_specs, ylabel, title_suffix, unit_note):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for col, label, color, style in series_specs:
        ax.plot(out_df["frame"], out_df[col], color=color, linestyle=style, linewidth=1.4, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dim}{title_suffix}: フレーム間の変位(Δlength) 比較\nCamera4, frame270-386{unit_note}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"displacement_comparison_{dim.lower()}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"displacement_comparison_{dim.lower()}.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote displacement_comparison_{dim.lower()}.png / .eps")


plot_delta("2D", [
    ("19点区間合計Δ_2D推定", "19点: 6区間の長さ合計のΔ(推定)", "tab:red", "-"),
    ("17点肩腰Δ_2D推定", "17点: 肩-腰の長さのΔ(推定)", "tab:blue", "-"),
    ("19点区間合計Δ_2D正解", "19点: 6区間の長さ合計のΔ(正解)", "tab:red", "--"),
    ("17点肩腰Δ_2D正解", "17点: 肩-腰の長さのΔ(正解)", "tab:blue", "--"),
], "Δlength [px]", "推定結果+正解データ", "")

plot_delta("3D", [
    ("19点区間合計Δ_3D推定", "19点: 6区間の長さ合計のΔ(推定)", "tab:red", "-"),
    ("17点肩腰Δ_3D推定", "17点: 肩-腰の長さのΔ(推定)", "tab:blue", "-"),
], "Δlength [normalized]", "推定結果", "\n(3Dは推定側の正規化スケール)")
