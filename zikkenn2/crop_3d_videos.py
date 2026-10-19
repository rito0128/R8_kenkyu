# -*- coding: utf-8 -*-
"""
3D推定結果の可視化動画(2000x1000、左半分=2Dオーバーレイ、右半分=3Dプロット)を編集する。
1. 右半分を削除(左半分の2Dオーバーレイパネルのみ残す) -> 1000x1000
2. 左半分の画角の上下左右それぞれ10%をカット -> 800x800

対象: 17pt_2d3d・19pt_2d3d の camera1・camera4、計4本。

出力: zikkenn2/27/{model}_{camera}_cropped.mp4
"""
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "27"

SOURCES = [
    ("19pt", 1, BASE_DIR / "19pt_2d3d/camera1/visualizations/motion_00x_test_Camera1.mp4"),
    ("19pt", 4, BASE_DIR / "19pt_2d3d/camera4/visualizations/motion_00x_test_Camera4.mp4"),
    ("17pt", 1, BASE_DIR / "17pt_2d3d/camera1/visualizations/motion_00x_test_Camera1.mp4"),
    ("17pt", 4, BASE_DIR / "17pt_2d3d/camera4/visualizations/motion_00x_test_Camera4.mp4"),
]

for model, cam, src in SOURCES:
    cap = cv2.VideoCapture(str(src))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    left_w = w // 2  # 右半分削除後の幅
    margin_x = int(left_w * 0.10)
    margin_y = int(h * 0.10)
    x0, x1 = margin_x, left_w - margin_x
    y0, y1 = margin_y, h - margin_y
    out_w, out_h = x1 - x0, y1 - y0

    out_path = OUT / f"{model}_camera{cam}_cropped.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        left = frame[:, :left_w]  # 右半分削除
        cropped = left[y0:y1, x0:x1]  # 4辺10%カット
        writer.write(cropped)
        count += 1

    cap.release()
    writer.release()
    print(f"{model} camera{cam}: {src.name} ({w}x{h}, {n}frames) -> {out_path.name} ({out_w}x{out_h}, {count}frames)")
