"""
2D姿勢推定結果(JSON)からキーポイント間の距離を計算し、CSVにまとめて書き出す検証スクリプト。

入力: zikkenn/for_test/predictions/out_Camera{N}_{frame:04d}.json
      (motion_007_out_ 接頭辞のファイルは内容が同一のため対象外)
出力: zikkenn/for_test/keypoint_lengths.csv
"""

import csv
import json
import re
from pathlib import Path

from keypoint_common import KEYPOINT_NAMES, KEYPOINT_PAIRS, distance

BASE_DIR = Path(__file__).resolve().parent
PRED_DIR = BASE_DIR / "predictions"
OUT_CSV = BASE_DIR / "keypoint_lengths.csv"

FILENAME_RE = re.compile(r"^out_Camera(\d+)_(\d+)\.json$")


def build_header():
    header = ["camera", "frame", "filename"]
    for idx, name in enumerate(KEYPOINT_NAMES):
        header += [f"kp{idx}_{name}_x", f"kp{idx}_{name}_y", f"kp{idx}_{name}_score"]
    for a, b in KEYPOINT_PAIRS:
        header.append(f"len_{KEYPOINT_NAMES[a]}_{KEYPOINT_NAMES[b]}")
    return header


def process_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return None

    detection = data[0]
    keypoints = detection["keypoints"]
    scores = detection["keypoint_scores"]

    row = {}
    for idx, name in enumerate(KEYPOINT_NAMES):
        x, y = keypoints[idx]
        row[f"kp{idx}_{name}_x"] = x
        row[f"kp{idx}_{name}_y"] = y
        row[f"kp{idx}_{name}_score"] = scores[idx]

    for a, b in KEYPOINT_PAIRS:
        row[f"len_{KEYPOINT_NAMES[a]}_{KEYPOINT_NAMES[b]}"] = distance(keypoints[a], keypoints[b])

    return row


def main():
    files = []
    for path in PRED_DIR.glob("out_Camera*_*.json"):
        if "_ploted" in path.name:
            continue
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        camera, frame = int(m.group(1)), int(m.group(2))
        files.append((camera, frame, path))

    files.sort(key=lambda t: (t[0], t[1]))

    header = build_header()
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for camera, frame, path in files:
            row = process_file(path)
            if row is None:
                continue
            row["camera"] = camera
            row["frame"] = frame
            row["filename"] = path.name
            writer.writerow(row)

    print(f"wrote {len(files)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
