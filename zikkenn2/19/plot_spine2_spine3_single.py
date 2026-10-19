# -*- coding: utf-8 -*-
"""
spine2・spine3それぞれの単体グラフ(2D・3D)を、文字を大きめにして作成する。
18番のflexibility_axes_frames.csv(Camera4, frame270-386, 19点モデル)を使用。

タイトル: 「関節(背2)の角度」「関節(背3)の角度」(2D/3Dは末尾に付記)
軸ラベル: 横=フレーム、縦=関節の角度[度]

出力: zikkenn2/19/
  joint_back2_2d.png/.eps  joint_back2_3d.png/.eps
  joint_back3_2d.png/.eps  joint_back3_3d.png/.eps
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent

frames_df = pd.read_csv(OUT / ".." / "18" / "flexibility_axes_frames.csv")

TARGETS = [
    ("spine2", "背2", "joint_back2"),
    ("spine3", "背3", "joint_back3"),
]

for joint, label, stub in TARGETS:
    sub = frames_df[frames_df["関節"] == joint].sort_values("frame")
    for dim in ["2D", "3D"]:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(sub["frame"], sub[f"{dim}推定"], color="tab:red", linewidth=2.2, label="推定")
        ax.plot(sub["frame"], sub[f"{dim}正解"], color="black", linestyle="--", linewidth=2.2, label="正解")

        ax.set_title(f"関節（{label}）の角度 ({dim})", fontsize=24, pad=16)
        ax.set_xlabel("フレーム", fontsize=20)
        ax.set_ylabel("関節の角度 [度]", fontsize=20)
        ax.tick_params(axis="both", labelsize=16)
        ax.legend(fontsize=18, loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(OUT / f"{stub}_{dim.lower()}.png", dpi=150, bbox_inches="tight")
        fig.savefig(OUT / f"{stub}_{dim.lower()}.eps", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {stub}_{dim.lower()}.png / .eps")
