# -*- coding: utf-8 -*-
"""
neck_waist_length.csv(フレームごとの首-腰長さ、17点/19点/正解 x 2D/3D)から、
モデルごとの平均・標準偏差・CV(変動係数=std/mean)をまとめる。

CVは無次元(比率)なので、2D(px)と3D(正規化/実寸)のようにスケールが異なる
データ同士でも「本来一定であるはずの骨長がどれだけ揺れているか」を直接比較できる。

出力: neck_waist_cv_summary.csv
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "neck_waist_length.csv")

grp = df.groupby(["source", "label"])["length"].agg(mean="mean", std="std")
grp["cv"] = grp["std"] / grp["mean"]
grp = grp.reset_index()

label_order = ["17点モデル(拡張前)", "19点モデル(拡張後)", "正解データ"]

wide = grp.pivot(index="label", columns="source", values=["mean", "std", "cv"])
wide.columns = [f"{src}_{stat}" for stat, src in wide.columns]
wide = wide.reindex(label_order)
wide = wide[["2D_mean", "2D_std", "2D_cv", "3D_mean", "3D_std", "3D_cv"]]
wide.index.name = "label"

out_path = BASE_DIR / "neck_waist_cv_summary.csv"
wide.to_csv(out_path, encoding="utf-8-sig")
print(f"wrote {out_path}")
print(wide.to_string())
