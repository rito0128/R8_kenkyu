# -*- coding: utf-8 -*-
"""
Camera1の推定結果動画から、以下2パターンをキーポイントのみ(骨格線なし)でプロットする。

① 首から腰にかけてのキーポイント(spine0〜spine5、19点モデルのみ存在)を、
   前回より明るい緑でプロット。17点モデルにはこの区間の点が無いため対象外。
② 両腕のキーポイント(肩・肘・手首)を、左腕=黄緑、右腕=赤でプロット。
   17点・19点モデル両方で作成。

出力: zikkenn2/28/
  19pt_camera1_spine_keypoints.mp4
  19pt_camera1_arms_keypoints.mp4
  17pt_camera1_arms_keypoints.mp4
"""
import cv2
import numpy as np

import common as c

OUT = c.BASE_DIR / "28"
OUT.mkdir(exist_ok=True)
CAM = 1
RADIUS = 8

BRIGHT_GREEN_BGR = (60, 255, 60)      # 明るい緑
YELLOWGREEN_BGR = (0, 255, 170)       # 彩度を上げた黄緑(左腕)
RED_BGR = (0, 0, 255)                 # 赤(右腕)

# 19点: spine0〜spine5
SPINE_IDX = [13, 14, 15, 16, 17, 18]

# 腕のキーポイント: 19点(idx), 17点2D=COCO(idx)
ARMS_19 = {"left": [1, 3, 5], "right": [2, 4, 6]}
ARMS_17 = {"left": [5, 7, 9], "right": [6, 8, 10]}


def render(model, indices_with_colors, out_name):
    input_video = c.BASE_DIR / f"{model}_2d3d" / f"camera{CAM}" / "input_video" / f"motion_00x_test_Camera{CAM}.mp4"
    pred = c.load_2d_pred(model, CAM).sort_values("frame").reset_index(drop=True)

    cap = cv2.VideoCapture(str(input_video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = OUT / out_name
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
            for idx, color in indices_with_colors:
                x, y = row[f"kp{idx}_x"], row[f"kp{idx}_y"]
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(frame, (int(round(x)), int(round(y))), RADIUS, color, thickness=-1,
                               lineType=cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"wrote {out_path.name} ({frame_idx}frames, {w}x{h})")


# ② 両腕, 左=黄緑(彩度アップ)・右=赤, 19点・17点 のみ再生成
arms19 = [(i, YELLOWGREEN_BGR) for i in ARMS_19["left"]] + [(i, RED_BGR) for i in ARMS_19["right"]]
render("19pt", arms19, "19pt_camera1_arms_keypoints.mp4")

arms17 = [(i, YELLOWGREEN_BGR) for i in ARMS_17["left"]] + [(i, RED_BGR) for i in ARMS_17["right"]]
render("17pt", arms17, "17pt_camera1_arms_keypoints.mp4")
