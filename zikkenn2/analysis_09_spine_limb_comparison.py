# -*- coding: utf-8 -*-
"""
9. 恒常性・柔軟性の脊柱/四肢比較表: 6番の9関節角度サマリから正解範囲の列を除外し、
   脊柱グループ→四肢グループの順に並べ、有効数字4桁に整形する。列の定義も付記する。

出力: zikkenn2/9/
  flexibility_spine_limb_comparison.csv
"""
import re
import pandas as pd

import common as c

OUT = c.BASE_DIR / "9"

df = pd.read_csv(c.BASE_DIR / "6" / "flexibility_9joint_summary.csv", encoding="utf-8-sig")

GROUP = {
    "spine0": "脊柱", "spine1": "脊柱", "spine2": "脊柱", "spine3": "脊柱", "spine4": "脊柱",
    "left_elbow": "四肢", "right_elbow": "四肢", "left_knee": "四肢", "right_knee": "四肢",
}
JOINT_ORDER = ["spine0", "spine1", "spine2", "spine3", "spine4", "left_elbow", "right_elbow", "left_knee", "right_knee"]


def range_sig4(s):
    lo, hi = re.split(r"〜", s.replace("°", ""))
    return f"{c.sig4(lo)}°〜{c.sig4(hi)}°"


out = pd.DataFrame({
    "camera": df["camera"],
    "関節": df["関節"],
    "グループ": df["関節"].map(GROUP),
    "2D可動範囲(°)": df["2D推定範囲"].map(range_sig4),
    "2D相関": df["2D相関"].map(c.sig4),
    "2D_MAE(°)": df["2D_MAE"].map(c.sig4),
    "3D可動範囲(°)": df["3D推定範囲"].map(range_sig4),
    "3D相関": df["3D相関"].map(c.sig4),
    "3D_MAE(°)": df["3D_MAE"].map(c.sig4),
})

out["_g"] = out["グループ"].map({"脊柱": 0, "四肢": 1})
out["_j"] = out["関節"].map({j: i for i, j in enumerate(JOINT_ORDER)})
out = out.sort_values(["camera", "_g", "_j"]).drop(columns=["_g", "_j"]).reset_index(drop=True)

definitions = pd.DataFrame({
    "項目": [
        "camera", "関節", "グループ", "2D可動範囲(°)", "2D相関", "2D_MAE(°)",
        "3D可動範囲(°)", "3D相関", "3D_MAE(°)",
    ],
    "定義": [
        "撮影カメラ番号(Camera1またはCamera4)",
        "3点角度を計算する対象の関節(脊柱はspine0〜spine4、四肢は肘・膝)。両端のhead/spine5は隣接点が1つのため単独では角度を計算できず、それぞれspine0/spine4の計算に腕として使用",
        "関節の分類。脊柱=spine0〜spine4、四肢=左右の肘・膝",
        "2D(19点拡張後モデル)推定角度の、全386フレーム中の最小値〜最大値",
        "2D推定角度と正解角度のフレーム間ピアソン相関係数(-1〜1、1に近いほど動きのタイミングが一致)",
        "2D推定角度と正解角度の平均絶対誤差(全386フレームの|推定-正解|の平均)",
        "3D(19点拡張後モデル)推定角度の、全386フレーム中の最小値〜最大値",
        "3D推定角度と正解角度のフレーム間ピアソン相関係数(-1〜1、1に近いほど動きのタイミングが一致)",
        "3D推定角度と正解角度の平均絶対誤差(全386フレームの|推定-正解|の平均)",
    ],
})

out_path = OUT / "flexibility_spine_limb_comparison.csv"
with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
    out.to_csv(f, index=False)
    f.write("\n# 各列の定義\n")
    definitions.to_csv(f, index=False)

print(f"wrote {out_path}")
print(out.to_string(index=False))
