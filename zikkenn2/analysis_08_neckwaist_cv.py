# -*- coding: utf-8 -*-
"""
8. 首-腰長さのスケール差を考慮した比較: 7番のデータから17点/19点/正解モデルの
   平均・標準偏差・CV(標準偏差/平均)を算出する。CVは無次元なので、
   2D(px)と3D(正規化/実寸)のようにスケールが異なるデータ同士でも直接比較できる。

出力: zikkenn2/8/
  neck_waist_cv_summary.csv
"""
import pandas as pd
from pathlib import Path

import common as c

OUT = c.BASE_DIR / "8"

df = pd.read_csv(c.BASE_DIR / "7" / "neck_waist_length.csv")

label_order = ["17点モデル(拡張前)", "19点モデル(拡張後)", "正解データ"]

rows = []
for cam in c.CAMERAS:
    sub = df[df.camera == cam]
    grp = sub.groupby(["source", "label"])["length"].agg(mean="mean", std="std")
    grp["cv"] = grp["std"] / grp["mean"]
    grp = grp.reset_index()
    wide = grp.pivot(index="label", columns="source", values=["mean", "std", "cv"])
    wide.columns = [f"{src}_{stat}" for stat, src in wide.columns]
    wide = wide.reindex(label_order)
    wide = wide[["2D_mean", "2D_std", "2D_cv", "3D_mean", "3D_std", "3D_cv"]]
    wide = wide.reset_index()
    wide.insert(0, "camera", cam)
    rows.append(wide)

out_df = pd.concat(rows, ignore_index=True)
out_df.to_csv(OUT / "neck_waist_cv_summary.csv", index=False, encoding="utf-8-sig")
print("wrote neck_waist_cv_summary.csv")
print(out_df.to_string(index=False))
