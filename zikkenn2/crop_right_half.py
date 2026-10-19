# -*- coding: utf-8 -*-
"""
19pt_2d3d/camera1・camera4/visualizations/ の動画(2000x1000、左半分=2Dオーバーレイ、
右半分=3Dプロット)から、右半分を削除して左半分(1000x1000)のみを残す。
それ以外の編集(マージンのカット等)は行わない。

出力: zikkenn2/30/{model}_camera{cam}_lefthalf.mp4
"""
import cv2
from pathlib import Path

BASE_DIR = Path(r"C:\Users\sinki\R8_kenkyu\zikkenn2")
OUT = BASE_DIR / "30"

SOURCES = [
    (1, BASE_DIR / "19pt_2d3d/camera1/visualizations/motion_00x_test_Camera1.mp4"),
    (4, BASE_DIR / "19pt_2d3d/camera4/visualizations/motion_00x_test_Camera4.mp4"),
]

for cam, src in SOURCES:
    cap = cv2.VideoCapture(str(src))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    left_w = w // 2
    out_path = OUT / f"19pt_camera{cam}_lefthalf.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (left_w, h))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame[:, :left_w])
        count += 1

    cap.release()
    writer.release()
    print(f"camera{cam}: {src.name} ({w}x{h}, {n}frames) -> {out_path.name} ({left_w}x{h}, {count}frames)")
