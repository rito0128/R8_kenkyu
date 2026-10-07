# -*- coding: utf-8 -*-
"""
3. 恒常性の評価: head-spine0-spine1-...-spine5 の6区間長のフレーム間CV(標準偏差/平均)を
   2D・3D、推定・正解で比較する(19点拡張後モデルのみ対象。17点モデルにはspine点が無い)。
   Camera1・Camera4それぞれで算出する。

出力: zikkenn2/3/
  spine_cv_summary.csv   区間×カメラ×(2D推定/2D正解/3D推定/3D正解)のCV表
"""
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "3"


def cv(arr):
    arr = np.asarray(arr, dtype=float)
    return arr.std() / arr.mean()


rows = []
for cam in c.CAMERAS:
    pred2d = c.load_2d_pred("19pt", cam)
    gt2d = c.load_2d_gt(cam)
    pred3d = c.load_3d_pred("19pt", cam)
    gt3d = c.load_3d_gt(cam)

    for (a, b), label in zip(c.SPINE_CHAIN, c.SPINE_CHAIN_LABELS):
        len_pred2d = c.dist(c.pred_xy(pred2d, a), c.pred_xy(pred2d, b))
        len_gt2d = c.dist(c.gt_xy(gt2d, a), c.gt_xy(gt2d, b))
        len_pred3d = c.dist(c.pred_xyz(pred3d, a), c.pred_xyz(pred3d, b))
        len_gt3d = c.dist(c.gt_xyz(gt3d, a), c.gt_xyz(gt3d, b))
        rows.append({
            "camera": cam, "区間": label,
            "2D推定CV": cv(len_pred2d), "2D正解CV": cv(len_gt2d),
            "3D推定CV": cv(len_pred3d), "3D正解CV": cv(len_gt3d),
        })

df = pd.DataFrame(rows)
df.to_csv(OUT / "spine_cv_summary.csv", index=False, encoding="utf-8-sig")
print("wrote spine_cv_summary.csv")
print(df.to_string(index=False))
