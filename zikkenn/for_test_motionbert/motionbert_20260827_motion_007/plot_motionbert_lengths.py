"""
motionbert_keypoint_lengths.csv から、カメラごと・キーポイントペアごとに
フレーム-長さの折れ線グラフを作成する。

出力:
  graphs/Camera{N}/len_{a}_{b}.png    (個別画像)
  graphs/summary/Camera{N}_summary.png (カメラごとに18ペアをまとめた画像)

MotionBertの出力はルート相対に正規化された無次元値のため、縦軸は「length [normalized]」とする。
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from keypoint_common import pair_column_names

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "motionbert_keypoint_lengths.csv"
GRAPH_DIR = BASE_DIR / "graphs"

N_COLS = 4
YLABEL = "length [normalized]"


def grid_shape(n):
    n_cols = N_COLS
    n_rows = -(-n // n_cols)  # ceil division
    return n_rows, n_cols


def plot_individual(df, len_columns, cameras):
    count = 0
    for camera in cameras:
        cam_df = df[df["camera"] == camera].sort_values("frame")
        out_dir = GRAPH_DIR / f"Camera{camera}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for col in len_columns:
            pair_name = col[len("len_"):]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(cam_df["frame"], cam_df[col], marker="o", markersize=2, linewidth=1)
            ax.set_title(f"Camera{camera}: {pair_name}")
            ax.set_xlabel("frame")
            ax.set_ylabel(YLABEL)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            fig.savefig(out_dir / f"len_{pair_name}.png", dpi=120)
            plt.close(fig)
            count += 1

    print(f"saved {count} individual graphs under {GRAPH_DIR}")


def plot_summary(df, len_columns, cameras):
    summary_dir = GRAPH_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = grid_shape(len(len_columns))

    for camera in cameras:
        cam_df = df[df["camera"] == camera].sort_values("frame")

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten()

        for ax, col in zip(axes, len_columns):
            pair_name = col[len("len_"):]
            ax.plot(cam_df["frame"], cam_df[col], marker="o", markersize=1.5, linewidth=1)
            ax.set_title(pair_name, fontsize=9)
            ax.set_xlabel("frame", fontsize=8)
            ax.set_ylabel(YLABEL, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        for ax in axes[len(len_columns):]:
            ax.axis("off")

        fig.suptitle(f"Camera{camera}: keypoint pair 3D lengths per frame", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        fig.savefig(summary_dir / f"Camera{camera}_summary.png", dpi=120)
        plt.close(fig)

    print(f"saved {len(cameras)} summary graphs under {summary_dir}")


def main():
    df = pd.read_csv(CSV_PATH)

    len_columns = pair_column_names()
    cameras = sorted(df["camera"].unique())

    plot_individual(df, len_columns, cameras)
    plot_summary(df, len_columns, cameras)


if __name__ == "__main__":
    main()
