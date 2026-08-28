"""
MotionBertによる3D姿勢推定結果(JSON)から、キーポイントペアごとの3D距離を計算し、
CSVにまとめて書き出す検証スクリプト。

入力: predictions/motion_007_Camera{N}.json
      (1ファイルに101フレーム分の [{"frame_id", "instances":[{"keypoints","keypoint_scores"}]}] が格納)
出力: motionbert_keypoint_lengths.csv

MotionBertの出力座標はカメラ空間・ルート相対の正規化された無次元値であり、ピクセルやメートルなどの
実寸単位ではない点に注意。
"""

import csv
import json
import re
from pathlib import Path

from keypoint_common import KEYPOINT_NAMES, KEYPOINT_PAIRS, distance

BASE_DIR = Path(__file__).resolve().parent
PRED_DIR = BASE_DIR / "predictions"
OUT_CSV = BASE_DIR / "motionbert_keypoint_lengths.csv"

FILENAME_RE = re.compile(r"^motion_007_Camera(\d+)\.json$")


def build_header():
    header = ["camera", "frame", "filename"]
    for idx, name in enumerate(KEYPOINT_NAMES):
        header += [f"kp{idx}_{name}_x", f"kp{idx}_{name}_y", f"kp{idx}_{name}_z", f"kp{idx}_{name}_score"]
    for a, b in KEYPOINT_PAIRS:
        header.append(f"len_{KEYPOINT_NAMES[a]}_{KEYPOINT_NAMES[b]}")
    return header


def process_frame(frame_entry):
    instances = frame_entry["instances"]
    if not instances:
        return None

    inst = instances[0]
    keypoints = inst["keypoints"]
    scores = inst["keypoint_scores"]

    row = {}
    for idx, name in enumerate(KEYPOINT_NAMES):
        x, y, z = keypoints[idx]
        row[f"kp{idx}_{name}_x"] = x
        row[f"kp{idx}_{name}_y"] = y
        row[f"kp{idx}_{name}_z"] = z
        row[f"kp{idx}_{name}_score"] = scores[idx]

    for a, b in KEYPOINT_PAIRS:
        row[f"len_{KEYPOINT_NAMES[a]}_{KEYPOINT_NAMES[b]}"] = distance(keypoints[a], keypoints[b])

    return row


def main():
    rows = []
    for path in sorted(PRED_DIR.glob("motion_007_Camera*.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        camera = int(m.group(1))

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for frame_entry in data:
            row = process_frame(frame_entry)
            if row is None:
                continue
            row["camera"] = camera
            row["frame"] = frame_entry["frame_id"] + 1
            row["filename"] = path.name
            rows.append(row)

    rows.sort(key=lambda r: (r["camera"], r["frame"]))

    header = build_header()
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
