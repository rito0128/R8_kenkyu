# -*- coding: utf-8 -*-
"""
2. 3D推定(MotionBert, 19点拡張後モデル)の精度評価・カメラ間重複バグの確認
   zikkenn/(旧データ)ではCamera2〜12の推定結果がバイト単位で重複するバグが見つかっていた。
   今回はCamera1・Camera4の2台のみだが、同様の重複が起きていないかをまず確認したうえで、
   3D側の推定誤差(スケールが異なるため絶対誤差ではなく参考値)をまとめる。

出力: zikkenn2/2/
  duplicate_check.txt       Camera1とCamera4の推定結果が重複していないかの確認結果
  keypoint_errors_3d.csv    フレーム・カメラ・キーポイントごとの3D位置誤差(推定側スケール)
  mpjpe3d_summary.csv       カメラ別・キーポイント別の要約
"""
import numpy as np
import pandas as pd

import common as c

OUT = c.BASE_DIR / "2"

# ---- 重複バグの確認: Camera1とCamera4の推定値が一致していないか ----
pred1 = c.load_3d_pred("19pt", 1)
pred4 = c.load_3d_pred("19pt", 4)
kp_cols = [col for col in pred1.columns if col != "frame"]
identical = np.allclose(pred1[kp_cols].values, pred4[kp_cols].values)

with open(OUT / "duplicate_check.txt", "w", encoding="utf-8") as f:
    f.write("Camera1とCamera4の3D推定結果(19点モデル)がバイト単位/数値的に一致しているかの確認\n")
    f.write(f"全キーポイント・全フレームで一致: {identical}\n")
    if identical:
        f.write("→ zikkenn(旧データ)で見つかったカメラ間重複バグが再発している可能性が高い。\n")
    else:
        diff = np.abs(pred1[kp_cols].values - pred4[kp_cols].values)
        f.write(f"→ 一致しない(正常)。平均絶対差: {diff.mean():.4g}, 最大絶対差: {diff.max():.4g}\n")
print(open(OUT / "duplicate_check.txt", encoding="utf-8").read())

# ---- 3D推定誤差(推定側スケール vs 正解、参考値) ----
rows = []
for cam in c.CAMERAS:
    pred = c.load_3d_pred("19pt", cam)
    gt = c.load_3d_gt(cam)
    assert len(pred) == len(gt) == c.N_FRAMES
    for i, name in enumerate(c.KP19_NAMES):
        err = c.dist(c.pred_xyz(pred, i), c.gt_xyz(gt, i))
        for frame, e in zip(pred["frame"], err):
            rows.append({"camera": cam, "frame": frame, "keypoint": name, "kp_idx": i, "error": e})

err_df = pd.DataFrame(rows)
err_df.to_csv(OUT / "keypoint_errors_3d.csv", index=False, encoding="utf-8-sig")

summary_cam = err_df.groupby("camera")["error"].agg(mean="mean", median="median", std="std").reset_index()
summary_kp = err_df.groupby(["keypoint", "kp_idx"])["error"].agg(mean="mean", median="median", std="std").reset_index().sort_values("kp_idx")

with open(OUT / "mpjpe3d_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    f.write("# 注意: 3D推定値は正規化されたスケールであり、正解(実寸)とスケールが異なるため\n")
    f.write("# 以下の誤差は絶対値としては参考値(スケール差含む)。トレンド比較用。\n\n")
    f.write("# カメラ別\n")
    summary_cam.to_csv(f, index=False)
    f.write("\n# キーポイント別\n")
    summary_kp.to_csv(f, index=False)
print("wrote mpjpe3d_summary.csv")
print(summary_cam)
