# -*- coding: utf-8 -*-
"""
5. セルフオクルージョンに対する精度の評価
   正解3Dデータのvisibilityフラグ(0/1=遮蔽, 2=可視)で、関節角度誤差(4番の4関節)を
   遮蔽時/可視時に層別して比較する。19点拡張後モデル、Camera1・Camera4。

出力: zikkenn2/5/
  occlusion_summary.csv   関節×カメラ×(遮蔽時MAE/可視時MAE/フレーム数)
"""
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "5"

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
    pred3d = c.load_3d_pred("19pt", cam)
    gt3d = c.load_3d_gt(cam)

    for name, (a, b, cidx) in JOINTS.items():
        ang_pred = angle3pt(c.pred_xyz(pred3d, a), c.pred_xyz(pred3d, b), c.pred_xyz(pred3d, cidx))
        ang_gt = angle3pt(c.gt_xyz(gt3d, a), c.gt_xyz(gt3d, b), c.gt_xyz(gt3d, cidx))
        abs_err = np.abs(ang_pred - ang_gt)

        # 中心関節(b)のvisibilityで遮蔽/可視を判定
        vis = gt3d[f"J{b}_visibility"].values
        occluded = vis < 2
        visible = vis == 2

        rows.append({
            "camera": cam, "関節": name,
            "遮蔽時MAE": abs_err[occluded].mean() if occluded.sum() > 0 else np.nan,
            "遮蔽時n": int(occluded.sum()),
            "可視時MAE": abs_err[visible].mean() if visible.sum() > 0 else np.nan,
            "可視時n": int(visible.sum()),
        })

df = pd.DataFrame(rows)
for col in ["遮蔽時MAE", "可視時MAE"]:
    df[col] = df[col].map(lambda v: c.sig4(v) if pd.notna(v) else "")
df.to_csv(OUT / "occlusion_summary.csv", index=False, encoding="utf-8-sig")
print("wrote occlusion_summary.csv")
print(df.to_string(index=False))
