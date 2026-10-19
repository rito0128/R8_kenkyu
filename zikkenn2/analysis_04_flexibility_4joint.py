# -*- coding: utf-8 -*-
"""
4. 柔軟性の評価(初期版): 肩-肘-手首、股関節-膝-足首の3点角度(左右の肘・膝、計4関節)を算出。
   19点拡張後モデル、Camera1・Camera4。
   (このあと6番で脊柱側spine0〜4を加えた9関節版に拡張する)

出力: zikkenn2/4/
  flexibility_4joint_summary.csv   関節×カメラごとの可動範囲・相関・MAE
"""
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "4"

# (肩, 肘, 手首) / (股関節, 膝, 足首) のインデックス。KP19_NAMESの並びに対応。
JOINTS = {
    "left_elbow": (1, 3, 5),
    "right_elbow": (2, 4, 6),
    "left_knee": (7, 9, 11),
    "right_knee": (8, 10, 12),
}


def angle3pt(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos = (v1 * v2).sum(axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


rows = []
for cam in c.CAMERAS:
    pred2d = c.load_2d_pred("19pt", cam)
    gt2d = c.load_2d_gt(cam)
    pred3d = c.load_3d_pred("19pt", cam)
    gt3d = c.load_3d_gt(cam)

    for name, (a, b, cidx) in JOINTS.items():
        ang_pred2d = angle3pt(c.pred_xy(pred2d, a), c.pred_xy(pred2d, b), c.pred_xy(pred2d, cidx))
        ang_gt2d = angle3pt(c.gt_xy(gt2d, a), c.gt_xy(gt2d, b), c.gt_xy(gt2d, cidx))
        ang_pred3d = angle3pt(c.pred_xyz(pred3d, a), c.pred_xyz(pred3d, b), c.pred_xyz(pred3d, cidx))
        ang_gt3d = angle3pt(c.gt_xyz(gt3d, a), c.gt_xyz(gt3d, b), c.gt_xyz(gt3d, cidx))

        rows.append({
            "camera": cam, "関節": name,
            "2D推定範囲": f"{ang_pred2d.min():.4g}°〜{ang_pred2d.max():.4g}°",
            "2D正解範囲": f"{ang_gt2d.min():.4g}°〜{ang_gt2d.max():.4g}°",
            "2D相関": np.corrcoef(ang_pred2d, ang_gt2d)[0, 1],
            "2D_MAE": np.abs(ang_pred2d - ang_gt2d).mean(),
            "3D推定範囲": f"{ang_pred3d.min():.4g}°〜{ang_pred3d.max():.4g}°",
            "3D正解範囲": f"{ang_gt3d.min():.4g}°〜{ang_gt3d.max():.4g}°",
            "3D相関": np.corrcoef(ang_pred3d, ang_gt3d)[0, 1],
            "3D_MAE": np.abs(ang_pred3d - ang_gt3d).mean(),
        })

df = pd.DataFrame(rows)
for col in ["2D相関", "2D_MAE", "3D相関", "3D_MAE"]:
    df[col] = df[col].map(c.sig4)
df.to_csv(OUT / "flexibility_4joint_summary.csv", index=False, encoding="utf-8-sig")
print("wrote flexibility_4joint_summary.csv")
print(df.to_string(index=False))
