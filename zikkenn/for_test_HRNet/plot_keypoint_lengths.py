"""
キーポイントペアの長さ(frame-length)グラフを作成する。

1. 推定結果(keypoint_lengths.csv)          -> graphs/
2. 正解データ(correct_keypoint_lengths.csv) -> graphs_gt/
3. 推定結果と正解データを重ねたもの          -> graphs_comparison/

各ケースについて、カメラ×ペアごとの個別画像と、カメラごとに13ペアをまとめたsummary画像を作成する。
単位はどちらもピクセル。
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from keypoint_common import pair_column_names

BASE_DIR = Path(__file__).resolve().parent
PRED_CSV = BASE_DIR / "keypoint_lengths.csv"
GT_CSV = BASE_DIR / "correct_keypoint_lengths.csv"

GRAPH_DIR = BASE_DIR / "graphs"
GT_GRAPH_DIR = BASE_DIR / "graphs_gt"
CMP_GRAPH_DIR = BASE_DIR / "graphs_comparison"

N_COLS = 4


def grid_shape(n):
    n_cols = N_COLS
    n_rows = -(-n // n_cols)  # ceil division
    return n_rows, n_cols


def plot_individual_single(df, len_columns, cameras, out_dir, label, ylabel="length [px]"):
    """1系列(推定 or 正解のみ)の個別グラフをカメラ×ペアごとに保存する。"""
    count = 0
    for camera in cameras:
        cam_df = df[df["camera"] == camera].sort_values("frame")
        out_cam_dir = out_dir / f"Camera{camera}"
        out_cam_dir.mkdir(parents=True, exist_ok=True)

        for col in len_columns:
            pair_name = col[len("len_"):]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(cam_df["frame"], cam_df[col], marker="o", markersize=2, linewidth=1, label=label)
            ax.set_title(f"Camera{camera}: {pair_name}")
            ax.set_xlabel("frame")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()

            fig.savefig(out_cam_dir / f"len_{pair_name}.png", dpi=120)
            plt.close(fig)
            count += 1

    print(f"saved {count} individual graphs under {out_dir}")


def plot_summary_single(df, len_columns, cameras, out_dir, label, ylabel="length [px]"):
    """1系列(推定 or 正解のみ)のsummaryグラフをカメラごとに保存する。"""
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = grid_shape(len(len_columns))

    for camera in cameras:
        cam_df = df[df["camera"] == camera].sort_values("frame")

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten()

        for ax, col in zip(axes, len_columns):
            pair_name = col[len("len_"):]
            ax.plot(cam_df["frame"], cam_df[col], marker="o", markersize=1.5, linewidth=1, label=label)
            ax.set_title(pair_name, fontsize=9)
            ax.set_xlabel("frame", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        for ax in axes[len(len_columns):]:
            ax.axis("off")

        fig.suptitle(f"Camera{camera}: keypoint pair lengths per frame", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        fig.savefig(summary_dir / f"Camera{camera}_summary.png", dpi=120)
        plt.close(fig)

    print(f"saved {len(cameras)} summary graphs under {summary_dir}")


def plot_individual_comparison(pred_df, gt_df, len_columns, cameras, out_dir):
    """推定結果と正解データを重ねた個別グラフをカメラ×ペアごとに保存する。"""
    count = 0
    for camera in cameras:
        p_df = pred_df[pred_df["camera"] == camera].sort_values("frame")
        g_df = gt_df[gt_df["camera"] == camera].sort_values("frame")
        out_cam_dir = out_dir / f"Camera{camera}"
        out_cam_dir.mkdir(parents=True, exist_ok=True)

        for col in len_columns:
            pair_name = col[len("len_"):]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(g_df["frame"], g_df[col], marker="o", markersize=2, linewidth=1, label="ground truth")
            ax.plot(p_df["frame"], p_df[col], marker="o", markersize=2, linewidth=1, label="prediction")
            ax.set_title(f"Camera{camera}: {pair_name}")
            ax.set_xlabel("frame")
            ax.set_ylabel("length [px]")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()

            fig.savefig(out_cam_dir / f"len_{pair_name}.png", dpi=120)
            plt.close(fig)
            count += 1

    print(f"saved {count} comparison graphs under {out_dir}")


def plot_summary_comparison(pred_df, gt_df, len_columns, cameras, out_dir):
    """推定結果と正解データを重ねたsummaryグラフをカメラごとに保存する。"""
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = grid_shape(len(len_columns))

    for camera in cameras:
        p_df = pred_df[pred_df["camera"] == camera].sort_values("frame")
        g_df = gt_df[gt_df["camera"] == camera].sort_values("frame")

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten()

        for ax, col in zip(axes, len_columns):
            pair_name = col[len("len_"):]
            ax.plot(g_df["frame"], g_df[col], marker="o", markersize=1.5, linewidth=1, label="ground truth")
            ax.plot(p_df["frame"], p_df[col], marker="o", markersize=1.5, linewidth=1, label="prediction")
            ax.set_title(pair_name, fontsize=9)
            ax.set_xlabel("frame", fontsize=8)
            ax.set_ylabel("length [px]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        for ax in axes[len(len_columns):]:
            ax.axis("off")

        fig.suptitle(f"Camera{camera}: prediction vs ground truth", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        fig.savefig(summary_dir / f"Camera{camera}_summary.png", dpi=120)
        plt.close(fig)

    print(f"saved {len(cameras)} comparison summary graphs under {summary_dir}")


def main():
    pred_df = pd.read_csv(PRED_CSV)
    gt_df = pd.read_csv(GT_CSV)

    len_columns = pair_column_names()
    cameras = sorted(set(pred_df["camera"]).union(gt_df["camera"]))

    plot_individual_single(pred_df, len_columns, cameras, GRAPH_DIR, label="prediction")
    plot_summary_single(pred_df, len_columns, cameras, GRAPH_DIR, label="prediction")

    plot_individual_single(gt_df, len_columns, cameras, GT_GRAPH_DIR, label="ground truth")
    plot_summary_single(gt_df, len_columns, cameras, GT_GRAPH_DIR, label="ground truth")

    plot_individual_comparison(pred_df, gt_df, len_columns, cameras, CMP_GRAPH_DIR)
    plot_summary_comparison(pred_df, gt_df, len_columns, cameras, CMP_GRAPH_DIR)


if __name__ == "__main__":
    main()
