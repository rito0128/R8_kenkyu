# -*- coding: utf-8 -*-
"""
29. スライド掲載用に、analysis_28の結果から要点(全体統計・代表例)だけを
    簡潔なCSVにまとめる。

出力: zikkenn2/29/
  slide_overall_stats.csv    伸び(1-30)x丸まり(100-150) 全1530通りの集計
  slide_best_examples.csv    拡張後が最も優位だった代表例(伸びframe25→丸まりframe149)。
                              縦方向に拡張前(17点)の各区間→合計、拡張後(19点)の各区間→合計
                              を並べ、横方向にframe25・frame149の長さを並べた表。
"""
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "29"
CAM = 1
STRAIGHT_FRAMES = list(range(1, 31))
BENT_FRAMES = list(range(100, 151))
N_EXAMPLES = 3

MODEL17_LABEL = "拡張前(17点)"
MODEL19_LABEL = "拡張後(19点)"
CHAIN17 = [(9, 7, "首の付け根-背中"), (7, 0, "背中-腰")]
CHAIN19 = [(14, 15, "spine1-spine2"), (15, 16, "spine2-spine3"),
           (16, 17, "spine3-spine4"), (17, 18, "spine4-spine5")]

pred17 = c.load_3d_pred("17pt", CAM)
pred19 = c.load_3d_pred("19pt", CAM)


def seg_series_by_frame(pred_df, chain):
    """{区間名: frame(1..386)をindexとするSeries}"""
    return {label: pd.Series(c.dist(c.pred_xyz(pred_df, a), c.pred_xyz(pred_df, b)), index=pred_df["frame"].values)
            for a, b, label in chain}


seg17 = seg_series_by_frame(pred17, CHAIN17)
seg19 = seg_series_by_frame(pred19, CHAIN19)
sum17 = sum(seg17.values())
sum19 = sum(seg19.values())

s17, s19 = sum17.loc[STRAIGHT_FRAMES].values, sum19.loc[STRAIGHT_FRAMES].values
b17, b19 = sum17.loc[BENT_FRAMES].values, sum19.loc[BENT_FRAMES].values

change17 = (b17[None, :] - s17[:, None]) / s17[:, None] * 100
change19 = (b19[None, :] - s19[:, None]) / s19[:, None] * 100
abs17, abs19 = np.abs(change17), np.abs(change19)
win_mask = abs19 < abs17
n_total, n_win = win_mask.size, int(win_mask.sum())

# ---------------- 全体統計 ----------------
overall = pd.DataFrame({
    "項目": [
        "対象組み合わせ数(伸び1-30 x 丸まり100-150)",
        "拡張後(19点)の方が|変化率|が小さい組み合わせ数",
        "拡張後が優位な割合(%)",
        "|変化率|平均 拡張前(17点)(%)",
        "|変化率|平均 拡張後(19点)(%)",
        "|変化率|中央値 拡張前(17点)(%)",
        "|変化率|中央値 拡張後(19点)(%)",
    ],
    "値": [
        n_total, n_win, round(n_win / n_total * 100, 2),
        round(abs17.mean(), 4), round(abs19.mean(), 4),
        round(float(np.median(abs17)), 4), round(float(np.median(abs19)), 4),
    ],
})
overall.to_csv(OUT / "slide_overall_stats.csv", index=False, encoding="utf-8-sig")
print("wrote slide_overall_stats.csv")
print(overall.to_string(index=False))

# ---------------- 代表例(拡張後が最も優位なペア=伸びframe25→丸まりframe149) ----------------
margin = abs17 - abs19
best_idx = np.argsort(-margin, axis=None)[0]
bi, bj = np.unravel_index(best_idx, margin.shape)
FI, FJ = STRAIGHT_FRAMES[bi], BENT_FRAMES[bj]  # 25, 149
COL_I, COL_J = f"frame{FI}", f"frame{FJ}"

rows = []
for label, series in seg17.items():
    vi, vj = float(series.loc[FI]), float(series.loc[FJ])
    rows.append({"モデル": MODEL17_LABEL, "区間": label,
                 COL_I: round(vi, 4), COL_J: round(vj, 4),
                 "変化量": round(vj - vi, 4), "変化率(%)": round((vj - vi) / vi * 100, 4)})
vi_sum, vj_sum = float(sum17.loc[FI]), float(sum17.loc[FJ])
rows.append({"モデル": MODEL17_LABEL, "区間": MODEL17_LABEL + " 合計",
             COL_I: round(vi_sum, 4), COL_J: round(vj_sum, 4),
             "変化量": round(vj_sum - vi_sum, 4), "変化率(%)": round((vj_sum - vi_sum) / vi_sum * 100, 4)})

for label, series in seg19.items():
    vi, vj = float(series.loc[FI]), float(series.loc[FJ])
    rows.append({"モデル": MODEL19_LABEL, "区間": label,
                 COL_I: round(vi, 4), COL_J: round(vj, 4),
                 "変化量": round(vj - vi, 4), "変化率(%)": round((vj - vi) / vi * 100, 4)})
vi_sum, vj_sum = float(sum19.loc[FI]), float(sum19.loc[FJ])
rows.append({"モデル": MODEL19_LABEL, "区間": MODEL19_LABEL + " 合計",
             COL_I: round(vi_sum, 4), COL_J: round(vj_sum, 4),
             "変化量": round(vj_sum - vi_sum, 4), "変化率(%)": round((vj_sum - vi_sum) / vi_sum * 100, 4)})

best_df = pd.DataFrame(rows, columns=["モデル", "区間", COL_I, COL_J, "変化量", "変化率(%)"])
best_df.to_csv(OUT / "slide_best_examples.csv", index=False, encoding="utf-8-sig")
print(f"\nbest pair: straight frame{FI} -> bent frame{FJ}")
print("wrote slide_best_examples.csv")
print(best_df.to_string(index=False))
