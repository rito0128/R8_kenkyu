# -*- coding: utf-8 -*-
"""
推定結果動画について、既存の見づらい骨格オーバーレイに代えて、
推定データ(19pt_2d, 17pt_2d)のキーポイントだけを黄緑色の点でプロットし直す
(骨格線は描画しない)。背景はオーバーレイの無いクリーンな入力動画を使用する。
Camera1・Camera4、17点・19点モデルの計4本。

出力: zikkenn2/28/{model}_camera{cam}_keypoints.mp4
"""
import cv2
import numpy as np

import common as c

OUT = c.BASE_DIR / "28"
OUT.mkdir(exist_ok=True)
COLOR_BGR = (50, 205, 154)  # 黄緑色(yellowgreen, RGB154,205,50)のBGR表現
RADIUS = 8
N_KP = {"19pt": 19, "17pt": 17}

for cam in c.CAMERAS:
    for model, n_kp in N_KP.items():
        input_video = c.BASE_DIR / f"{model}_2d3d" / f"camera{cam}" / "input_video" / f"motion_00x_test_Camera{cam}.mp4"
        pred = c.load_2d_pred(model, cam).sort_values("frame").reset_index(drop=True)

        cap = cv2.VideoCapture(str(input_video))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = OUT / f"{model}_camera{cam}_keypoints.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            row = pred[pred.frame == frame_idx]
            if len(row) == 1:
                row = row.iloc[0]
                for i in range(n_kp):
                    x, y = row[f"kp{i}_x"], row[f"kp{i}_y"]
                    if np.isfinite(x) and np.isfinite(y):
                        cv2.circle(frame, (int(round(x)), int(round(y))), RADIUS, COLOR_BGR, thickness=-1,
                                   lineType=cv2.LINE_AA)
            writer.write(frame)

        cap.release()
        writer.release()
        print(f"{model} camera{cam}: {frame_idx}frames -> {out_path.name} ({w}x{h})")
