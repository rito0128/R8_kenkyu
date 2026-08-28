"""
correct_2d_annotation.csv (正解キーポイント座標) から、
キーポイントペアごとの長さを計算してCSVに書き出す。

入力: zikkenn/for_test/correct_2d_annotation.csv (J0..J18 の x/y/visibility, filename)
出力: zikkenn/for_test/correct_keypoint_lengths.csv
"""

import re
from pathlib import Path

import pandas as pd

from keypoint_common import KEYPOINT_NAMES, KEYPOINT_PAIRS, distance

BASE_DIR = Path(__file__).resolve().parent
GT_CSV = BASE_DIR / "correct_2d_annotation.csv"
OUT_CSV = BASE_DIR / "correct_keypoint_lengths.csv"

FILENAME_RE = re.compile(r"^out_Camera(\d+)_(\d+)$")


def main():
    gt = pd.read_csv(GT_CSV)

    rows = []
    for _, r in gt.iterrows():
        m = FILENAME_RE.match(r["filename"])
        if not m:
            continue
        camera, frame = int(m.group(1)), int(m.group(2))

        row = {"camera": camera, "frame": frame, "filename": r["filename"]}
        keypoints = []
        for idx, name in enumerate(KEYPOINT_NAMES):
            x, y = r[f"J{idx}_x"], r[f"J{idx}_y"]
            v = r[f"J{idx}_visibility"]
            row[f"kp{idx}_{name}_x"] = x
            row[f"kp{idx}_{name}_y"] = y
            row[f"kp{idx}_{name}_visibility"] = v
            keypoints.append((x, y))

        for a, b in KEYPOINT_PAIRS:
            row[f"len_{KEYPOINT_NAMES[a]}_{KEYPOINT_NAMES[b]}"] = distance(keypoints[a], keypoints[b])

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["camera", "frame"])
    out_df.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(out_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
