"""キーポイント名・ペア定義・距離計算など、各スクリプトで共有する定数/関数。(MotionBert 3D用)"""

import math

KEYPOINT_NAMES = [
    "head",           # 0
    "left_shoulder",  # 1
    "right_shoulder", # 2
    "left_elbow",     # 3
    "right_elbow",    # 4
    "left_wrist",     # 5
    "right_wrist",    # 6
    "left_hip",       # 7
    "right_hip",      # 8
    "left_knee",      # 9
    "right_knee",     # 10
    "left_ankle",     # 11
    "right_ankle",    # 12
    "spine0",         # 13
    "spine1",         # 14
    "spine2",         # 15
    "spine3",         # 16
    "spine4",         # 17
    "spine5",         # 18
]

KEYPOINT_PAIRS = [
    (11, 9),
    (9, 7),
    (12, 10),
    (10, 8),
    (7, 8),
    (1, 7),
    (2, 8),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),
]


def distance(p1, p2):
    return math.dist(p1, p2)


def pair_column_names():
    return [f"len_{KEYPOINT_NAMES[a]}_{KEYPOINT_NAMES[b]}" for a, b in KEYPOINT_PAIRS]
