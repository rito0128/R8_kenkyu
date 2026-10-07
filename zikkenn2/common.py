# -*- coding: utf-8 -*-
"""
zikkenn2 (motion_00x_test データ)向けの共通データ読み込み・幾何計算関数。
zikkenn/ 側の各種スクリプトと同じ命名・区間定義を踏襲し、1〜11の解析スクリプトで共有する。

データ構成:
  17pt_2d/camera{1,4}/predictions/out_Camera{N}_XXXX.json   17点モデル 2D推定(COCO17順)
  19pt_2d/camera{1,4}/predictions/out_Camera{N}_XXXX.json   19点モデル 2D推定(拡張後の順)
  17pt_2d3d/camera{1,4}/predictions/motion_00x_test_Camera{N}.json  17点モデル 3D推定(H36M17順)
  19pt_2d3d/camera{1,4}/predictions/motion_00x_test_Camera{N}.json  19点モデル 3D推定(拡張後の順)
  motion_00x_test_2d_annotation.csv  正解2D(19点, J0〜J18, 12カメラ×386フレーム)
  motion_00x_test_3d_annotation.csv  正解3D(19点, J0〜J18, 12カメラ×386フレーム)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
N_FRAMES = 386
CAMERAS = [1, 4]

# 拡張後(19点)モデルのキーポイント名・区間定義(zikkenn/for_test_motionbert/keypoint_common.py と同一)
KP19_NAMES = [
    "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle", "spine0", "spine1", "spine2",
    "spine3", "spine4", "spine5",
]
SPINE_CHAIN = [(0, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18)]
SPINE_CHAIN_LABELS = [f"{KP19_NAMES[a]}-{KP19_NAMES[b]}" for a, b in SPINE_CHAIN]

# 17点モデル: 2D=COCO17順, 3D=H36M17順(zikkenn/for_test_keypoint_17_motion_007と同一の想定)
KP17_2D_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP17_3D_NAMES = [
    "hip", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee",
    "left_ankle", "spine", "thorax", "neck_nose", "head_top",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
]


def _frame_from_filename(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1))


# ---------------- 2D推定結果(HRNet) ----------------
def load_2d_pred(model, camera):
    """model: '17pt' or '19pt'. 戻り値: frame(1..386) x keypoints(x,y,score) のDataFrame"""
    n_kp = 17 if model == "17pt" else 19
    pred_dir = BASE_DIR / f"{model}_2d" / f"camera{camera}" / "predictions"
    rows = []
    for f in sorted(pred_dir.glob(f"out_Camera{camera}_*.json")):
        frame = _frame_from_filename(f.stem)
        data = json.load(open(f, encoding="utf-8"))
        kp = np.asarray(data[0]["keypoints"])  # (n_kp, 2)
        score = np.asarray(data[0]["keypoint_scores"])
        row = {"frame": frame}
        for i in range(n_kp):
            row[f"kp{i}_x"] = kp[i, 0]
            row[f"kp{i}_y"] = kp[i, 1]
            row[f"kp{i}_score"] = score[i]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


# ---------------- 3D推定結果(MotionBert) ----------------
def load_3d_pred(model, camera):
    """model: '17pt' or '19pt'. 戻り値: frame(1..386) x keypoints(x,y,z) のDataFrame"""
    pred_path = BASE_DIR / f"{model}_2d3d" / f"camera{camera}" / "predictions" / f"motion_00x_test_Camera{camera}.json"
    data = json.load(open(pred_path, encoding="utf-8"))
    n_kp = 17 if model == "17pt" else 19
    rows = []
    for entry in data:
        frame = entry["frame_id"] + 1
        kp = np.asarray(entry["instances"][0]["keypoints"])  # (n_kp, 3)
        row = {"frame": frame}
        for i in range(n_kp):
            row[f"kp{i}_x"] = kp[i, 0]
            row[f"kp{i}_y"] = kp[i, 1]
            row[f"kp{i}_z"] = kp[i, 2]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


# ---------------- 正解データ(19点) ----------------
def _load_gt(csv_name, dims):
    df = pd.read_csv(BASE_DIR / csv_name)
    m = df["filename"].str.extract(r"_Camera(\d+)_(\d+)$")
    df = df.assign(camera=m[0].astype(int), frame=m[1].astype(int))
    return df


def load_2d_gt(camera):
    df = _load_gt("motion_00x_test_2d_annotation.csv", 2)
    df = df[df.camera == camera].sort_values("frame").reset_index(drop=True)
    return df


def load_3d_gt(camera):
    df = _load_gt("motion_00x_test_3d_annotation.csv", 3)
    df = df[df.camera == camera].sort_values("frame").reset_index(drop=True)
    return df


def gt_xy(df, j):
    return df[[f"J{j}_x", f"J{j}_y"]].values


def gt_xyz(df, j):
    return df[[f"J{j}_x", f"J{j}_y", f"J{j}_z"]].values


def pred_xy(df, i):
    return df[[f"kp{i}_x", f"kp{i}_y"]].values


def pred_xyz(df, i):
    return df[[f"kp{i}_x", f"kp{i}_y", f"kp{i}_z"]].values


def dist(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)


def sig4(x):
    return f"{float(x):.4g}"
