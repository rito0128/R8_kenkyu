# -*- coding: utf-8 -*-
"""
28. 背筋が伸びているフレーム群(1〜30)と、背筋が丸まっているフレーム群(100〜150)の
    全組み合わせについて、拡張前(17点)・拡張後(19点)モデルの3D推定(Camera1)で
    「首の付け根〜腰」の合計長の変化率(%)を計算し、
    拡張後の方が|変化率|が小さい(=恒常性が良い)組み合わせが何通りあるかを調べる。

    区間定義はanalysis_27と同じ:
      拡張前(17点): 首の付け根(neck_nose,9)-背中(spine,7)-腰(hip,0) の2区間
      拡張後(19点): spine1(14)-spine2-spine3-spine4-spine5(18) の4区間(首側の1区間は除外)

出力: zikkenn2/28/
  frame_pair_search_summary.txt      全体の勝敗集計(件数・勝率、両モデルの|変化率|平均など)
  frame_pair_win_examples.csv/.png/.eps   拡張後が優位な組み合わせの上位例(区間別内訳つき)
  frame_pair_change_heatmap.png/.eps      全組み合わせの変化率差(|Δ17|-|Δ19|)のヒートマップ
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "28"
CAM = 1
STRAIGHT_FRAMES = list(range(1, 31))    # 背筋が伸びている: 1〜30 (frame0は存在しないため1始まり)
BENT_FRAMES = list(range(100, 151))     # 背筋が丸まっている: 100〜150
N_EXAMPLES = 3

MODEL17_LABEL = "拡張前(17点)"
MODEL19_LABEL = "拡張後(19点)"
CHAIN17 = [(9, 7, "首の付け根-背中"), (7, 0, "背中-腰")]
CHAIN19 = [(14, 15, "spine1-spine2"), (15, 16, "spine2-spine3"),
           (16, 17, "spine3-spine4"), (17, 18, "spine4-spine5")]

pred17 = c.load_3d_pred("17pt", CAM)
pred19 = c.load_3d_pred("19pt", CAM)


def seg_lengths_by_frame(pred_df, chain):
    """frame(1..386) -> {区間名: 長さ} の辞書、frameをキーにした辞書のリストで返す"""
    out = {}
    for a, b, label in chain:
        out[label] = pd.Series(
            c.dist(c.pred_xyz(pred_df, a), c.pred_xyz(pred_df, b)),
            index=pred_df["frame"].values,
        )
    return out


seg17 = seg_lengths_by_frame(pred17, CHAIN17)
seg19 = seg_lengths_by_frame(pred19, CHAIN19)
sum17 = sum(seg17.values())  # Series indexed by frame
sum19 = sum(seg19.values())

s17 = sum17.loc[STRAIGHT_FRAMES].values  # (30,)
s19 = sum19.loc[STRAIGHT_FRAMES].values
b17 = sum17.loc[BENT_FRAMES].values      # (51,)
b19 = sum19.loc[BENT_FRAMES].values

# change[i, j]: straight frame i -> bent frame j
change17 = (b17[None, :] - s17[:, None]) / s17[:, None] * 100
change19 = (b19[None, :] - s19[:, None]) / s19[:, None] * 100

abs17 = np.abs(change17)
abs19 = np.abs(change19)
win_mask = abs19 < abs17
margin = abs17 - abs19  # 正なら拡張後が優位

n_total = win_mask.size
n_win = int(win_mask.sum())

summary_lines = [
    f"対象: Camera1, 3D推定, 首の付け根〜腰の合計長",
    f"背筋が伸びている候補フレーム: {STRAIGHT_FRAMES[0]}〜{STRAIGHT_FRAMES[-1]} ({len(STRAIGHT_FRAMES)}枚)",
    f"背筋が丸まっている候補フレーム: {BENT_FRAMES[0]}〜{BENT_FRAMES[-1]} ({len(BENT_FRAMES)}枚)",
    f"全組み合わせ数: {n_total}",
    f"拡張後(19点)の方が|変化率|が小さい(=恒常性が良い)組み合わせ数: {n_win} ({n_win / n_total * 100:.2f}%)",
    f"拡張前(17点)の方が良い組み合わせ数: {n_total - n_win} ({(n_total - n_win) / n_total * 100:.2f}%)",
    "",
    f"|変化率|の平均 拡張前(17点): {abs17.mean():.4g}%  (min {abs17.min():.4g}% / max {abs17.max():.4g}%)",
    f"|変化率|の平均 拡張後(19点): {abs19.mean():.4g}%  (min {abs19.min():.4g}% / max {abs19.max():.4g}%)",
    f"|変化率|の中央値 拡張前(17点): {np.median(abs17):.4g}%",
    f"|変化率|の中央値 拡張後(19点): {np.median(abs19):.4g}%",
]
summary_txt = "\n".join(summary_lines)
(OUT / "frame_pair_search_summary.txt").write_text(summary_txt, encoding="utf-8")
print(summary_txt)

# ---------------- 上位例(拡張後が最も優位な組み合わせ)の詳細表 ----------------
flat_idx = np.argsort(-margin, axis=None)[:N_EXAMPLES]
top_pairs = [(STRAIGHT_FRAMES[i], BENT_FRAMES[j]) for i, j in
             [np.unravel_index(idx, margin.shape) for idx in flat_idx]]

rows = []
for fi, fj in top_pairs:
    for label, series in seg17.items():
        vi, vj = series.loc[fi], series.loc[fj]
        rows.append({"フレームペア": f"{fi}→{fj}", "モデル": MODEL17_LABEL, "区間": label,
                      f"frame_i": vi, f"frame_j": vj, "変化量": vj - vi, "変化率(%)": (vj - vi) / vi * 100})
    vi_sum, vj_sum = sum17.loc[fi], sum17.loc[fj]
    rows.append({"フレームペア": f"{fi}→{fj}", "モデル": MODEL17_LABEL, "区間": MODEL17_LABEL + " 合計",
                  "frame_i": vi_sum, "frame_j": vj_sum, "変化量": vj_sum - vi_sum,
                  "変化率(%)": (vj_sum - vi_sum) / vi_sum * 100})
    for label, series in seg19.items():
        vi, vj = series.loc[fi], series.loc[fj]
        rows.append({"フレームペア": f"{fi}→{fj}", "モデル": MODEL19_LABEL, "区間": label,
                      "frame_i": vi, "frame_j": vj, "変化量": vj - vi, "変化率(%)": (vj - vi) / vi * 100})
    vi_sum, vj_sum = sum19.loc[fi], sum19.loc[fj]
    rows.append({"フレームペア": f"{fi}→{fj}", "モデル": MODEL19_LABEL, "区間": MODEL19_LABEL + " 合計",
                  "frame_i": vi_sum, "frame_j": vj_sum, "変化量": vj_sum - vi_sum,
                  "変化率(%)": (vj_sum - vi_sum) / vi_sum * 100})

ex_df = pd.DataFrame(rows)
ex_df.to_csv(OUT / "frame_pair_win_examples.csv", index=False, encoding="utf-8-sig")
print(f"\ntop {N_EXAMPLES} pairs (straight_frame -> bent_frame): {top_pairs}")
print("wrote frame_pair_win_examples.csv")


def fmt(v):
    return f"{v:.4g}"


header = ["フレームペア", "モデル", "区間", "frame_i", "frame_j", "変化量", "変化率(%)"]
cell_text = [[row[h] if h in ("フレームペア", "モデル", "区間") else fmt(row[h]) for h in header]
             for _, row in ex_df.iterrows()]
all_rows = [header] + cell_text
sum_row_idxs = {1 + i for i, label in enumerate(ex_df["区間"]) if label.endswith("合計")}

fig, ax = plt.subplots(figsize=(13, 0.42 * len(all_rows) + 1))
ax.axis("off")
col_widths = [0.13, 0.15, 0.22, 0.125, 0.125, 0.125, 0.125]
table = ax.table(cellText=all_rows, loc="center", cellLoc="center", colWidths=col_widths)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.7)
for (r, cidx), cell in table.get_celld().items():
    cell.set_edgecolor("#444444")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(weight="bold")
    if r in sum_row_idxs:
        cell.set_facecolor("#f7f0e0")
        cell.set_text_props(weight="bold")
    if cidx in (0, 1, 2):
        cell.set_text_props(ha="left")
ax.set_title(f"Camera1 3D推定: 拡張後が最も優位な組み合わせ 上位{N_EXAMPLES}例", fontsize=12, pad=12)
fig.tight_layout()
fig.savefig(OUT / "frame_pair_win_examples.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "frame_pair_win_examples.eps", bbox_inches="tight")
plt.close(fig)
print("wrote frame_pair_win_examples.png / .eps")

# ---------------- ヒートマップ: margin = |Δ17|-|Δ19| (正=拡張後優位) ----------------
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(margin, aspect="auto", origin="lower", cmap="RdBu",
                vmin=-np.abs(margin).max(), vmax=np.abs(margin).max(),
                extent=[BENT_FRAMES[0], BENT_FRAMES[-1], STRAIGHT_FRAMES[0], STRAIGHT_FRAMES[-1]])
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("|変化率17点| - |変化率19点|  (正=拡張後が優位)")
ax.set_xlabel("丸まっているフレーム (100〜150)")
ax.set_ylabel("伸びているフレーム (1〜30)")
ax.set_title(f"拡張後の優位性マップ (青=拡張後が優位 n={n_win}/{n_total}, 赤=拡張前が優位)")
fig.tight_layout()
fig.savefig(OUT / "frame_pair_change_heatmap.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "frame_pair_change_heatmap.eps", bbox_inches="tight")
plt.close(fig)
print("wrote frame_pair_change_heatmap.png / .eps")
