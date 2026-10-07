# -*- coding: utf-8 -*-
"""
zikkenn2/30/19pt_camera4_lefthalf.mp4(既に右半分削除済み)に対して、
首から腰にかけてのキーポイント(spine0〜spine4、spine5は除外、
19点モデルのみ存在)をRGB(0,255,0)の丸で全フレーム描画し直す。
17点モデルにはこの区間の点が存在しないため対象外。

出力: zikkenn2/30/19pt_camera4_lefthalf_spine.mp4 (同じ1000x1000, 386フレーム)
"""
import cv2
import numpy as np

import common as c

OUT = c.BASE_DIR / "30"
CAM = 4
COLOR_BGR = (0, 255, 0)  # RGB(0,255,0) = BGR(0,255,0)
RADIUS = 10

SPINE_IDX = [13, 14, 15, 16, 17]  # spine0〜spine4(spine5は除外)

src = OUT / "19pt_camera4_lefthalf.mp4"
pred = c.load_2d_pred("19pt", CAM).sort_values("frame").reset_index(drop=True)

cap = cv2.VideoCapture(str(src))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out_path = OUT / "19pt_camera4_lefthalf_spine.mp4"
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
        for idx in SPINE_IDX:
            x, y = row[f"kp{idx}_x"], row[f"kp{idx}_y"]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(frame, (int(round(x)), int(round(y))), RADIUS, COLOR_BGR, thickness=-1,
                           lineType=cv2.LINE_AA)
    writer.write(frame)

cap.release()
writer.release()
print(f"{src.name} ({w}x{h}, {frame_idx}frames) -> {out_path.name}")
