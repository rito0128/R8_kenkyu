# -*- coding: utf-8 -*-
"""
セルフオクルージョンの評価(20番)に使ったCamera1の動画から、
frame1〜frame222(28番・29番の解析対象と同じ範囲)のみを切り出す。

元動画: 19pt_2d3d/camera1/input_video/motion_00x_test_Camera1.mp4
        (17pt/19ptで共通の元映像。386フレーム, 24fps, 1000x1000)

出力: zikkenn2/30/motion_00x_test_Camera1_frame1-222.mp4
"""
import cv2
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "19pt_2d3d" / "camera1" / "input_video" / "motion_00x_test_Camera1.mp4"
OUT_DIR = BASE / "30"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "motion_00x_test_Camera1_frame1-222.mp4"

FRAME_RANGE = (1, 222)  # 1-indexed, 両端含む

cap = cv2.VideoCapture(str(SRC))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"source: {total} frames, {fps}fps, {w}x{h}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_PATH), fourcc, fps, (w, h))

start_idx = FRAME_RANGE[0] - 1  # 0-indexed
end_idx = FRAME_RANGE[1] - 1

frame_idx = 0
written = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if start_idx <= frame_idx <= end_idx:
        writer.write(frame)
        written += 1
    frame_idx += 1
    if frame_idx > end_idx:
        break

cap.release()
writer.release()
print(f"wrote {OUT_PATH} ({written} frames)")
