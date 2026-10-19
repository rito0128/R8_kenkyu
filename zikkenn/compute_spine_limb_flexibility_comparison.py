# -*- coding: utf-8 -*-
"""
柔軟性評価(flexibility_angles_summary.csv)から、脊柱(spine0〜spine4)と
四肢(肘・膝)を比較しやすい形に整形する。

- 正解範囲の列は省略(相関・MAEに正解との比較情報が含まれるため)
- 各関節を「脊柱」「四肢」グループに分類
- 数値は有効数字4桁に統一
- 各列の定義を、データ表の下に別セクションとして追記

出力: flexibility_spine_limb_comparison.csv
"""

import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "flexibility_angles_summary.csv", encoding="utf-8-sig")
df.columns = ["関節", "2D推定範囲", "2D正解範囲", "2D相関", "2D_MAE", "3D推定範囲", "3D正解範囲", "3D相関", "3D_MAE"]

GROUP = {
    "spine0": "脊柱", "spine1": "脊柱", "spine2": "脊柱", "spine3": "脊柱", "spine4": "脊柱",
    "left_elbow": "四肢", "right_elbow": "四肢", "left_knee": "四肢", "right_knee": "四肢",
}


def sig4(x):
    return f"{float(x):.4g}"


def range_sig4(s):
    lo, hi = re.split(r"〜", s.replace("°", ""))
    return f"{sig4(lo)}°〜{sig4(hi)}°"


out = pd.DataFrame({
    "関節": df["関節"],
    "グループ": df["関節"].map(GROUP),
    "2D可動範囲(°)": df["2D推定範囲"].map(range_sig4),
    "2D相関": df["2D相関"].map(sig4),
    "2D_MAE(°)": df["2D_MAE"].map(sig4),
    "3D可動範囲(°)": df["3D推定範囲"].map(range_sig4),
    "3D相関": df["3D相関"].map(sig4),
    "3D_MAE(°)": df["3D_MAE"].map(sig4),
})

# 脊柱→四肢の順に並べ、グループ内は元の順序を維持
out["_order"] = out["グループ"].map({"脊柱": 0, "四肢": 1})
out = out.sort_values(["_order"]).drop(columns="_order").reset_index(drop=True)

definitions = pd.DataFrame({
    "項目": [
        "関節", "グループ", "2D可動範囲(°)", "2D相関", "2D_MAE(°)",
        "3D可動範囲(°)", "3D相関", "3D_MAE(°)",
    ],
    "定義": [
        "3点角度を計算する対象の関節(脊柱はspine0〜spine4、四肢は肘・膝)。両端のhead/spine5は隣接点が1つのため単独では角度を計算できず、それぞれspine0/spine4の計算に腕として使用",
        "関節の分類。脊柱=spine0〜spine4、四肢=左右の肘・膝",
        "2D(HRNet)推定角度の、全101フレーム中の最小値〜最大値",
        "2D推定角度と正解角度のフレーム間ピアソン相関係数(-1〜1、1に近いほど動きのタイミングが一致)",
        "2D推定角度と正解角度の平均絶対誤差(全101フレームの|推定-正解|の平均)",
        "3D(MotionBert, Camera1)推定角度の、全101フレーム中の最小値〜最大値",
        "3D推定角度と正解角度のフレーム間ピアソン相関係数(-1〜1、1に近いほど動きのタイミングが一致)",
        "3D推定角度と正解角度の平均絶対誤差(全101フレームの|推定-正解|の平均)",
    ],
})

out_path = BASE_DIR / "flexibility_spine_limb_comparison.csv"
with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
    out.to_csv(f, index=False)
    f.write("\n")
    f.write("# 各列の定義\n")
    definitions.to_csv(f, index=False)

print(f"wrote {out_path}")
