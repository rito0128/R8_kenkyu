import pandas as pd
import numpy as np

def convert_blender_csv_to_motionbert_npz(
    csv_2d_path: str,
    csv_3d_path: str,
    output_npz_path: str,
    num_joints: int = 19
):
    """
    Blenderで出力した2D/3D CSVアノテーションを MMPose MotionBERT (.npz) 形式に変換します。
    """
    print(f"📖 CSVファイルを読み込んでいます...")
    df_2d = pd.read_csv(csv_2d_path)
    df_3d = pd.read_csv(csv_3d_path)
    
    # 行数の整合性チェック
    assert len(df_2d) == len(df_3d), f"2D({len(df_2d)}行)と3D({len(df_3d)}行)のデータ数が一致しません。"
    num_samples = len(df_2d)
    print(f"✅ 合計 {num_samples} フレームのデータを検出しました (関節数: {num_joints})。")

    # 1. 2D座標 (joints_2d) とスコア (confidence) の抽出 [N, K, 2] / [N, K, 1]
    joints_2d = np.zeros((num_samples, num_joints, 2), dtype=np.float32)
    confidence = np.zeros((num_samples, num_joints, 1), dtype=np.float32)

    for i in range(num_joints):
        joints_2d[:, i, 0] = df_2d[f'J{i}_x'].values
        joints_2d[:, i, 1] = df_2d[f'J{i}_y'].values
        # visibility: Blender側が 2=visible, 0=invisible などの場合、1.0 / 0.0 スコアに正規化
        vis = df_2d[f'J{i}_visibility'].values
        confidence[:, i, 0] = np.where(vis > 0, 1.0, 0.0)

    # 2. 3D座標 (S) の抽出 [N, K, 3]
    S = np.zeros((num_samples, num_joints, 3), dtype=np.float32)
    for i in range(num_joints):
        S[:, i, 0] = df_3d[f'J{i}_x'].values
        S[:, i, 1] = df_3d[f'J{i}_y'].values
        S[:, i, 2] = df_3d[f'J{i}_z'].values

    # 3. バウンディングボックス (bbox) の抽出 [N, 4]
    bbox = np.zeros((num_samples, 4), dtype=np.float32)
    bbox[:, 0] = df_2d['bbox_xmin'].values
    bbox[:, 1] = df_2d['bbox_ymin'].values
    bbox[:, 2] = df_2d['bbox_xmax'].values
    bbox[:, 3] = df_2d['bbox_ymax'].values

    # 4. 画像ファイル名 (image_path) の抽出
    # filename が 'out_Camera1_0001' の場合、'.jpg' 拡張子を補完
    filenames = df_2d['filename'].astype(str).values
    image_paths = np.array([
        f"{fn}.jpg" if not fn.endswith(('.jpg', '.png', '.jpeg')) else fn 
        for fn in filenames
    ])

    # 5. .npz ファイルとして保存
    np.savez_compressed(
        output_npz_path,
        S=S,                    # 3D 座標 [N, 19, 3]
        joints_2d=joints_2d,    # 2D 座標 [N, 19, 2]
        confidence=confidence,  # 関節スコア [N, 19, 1]
        bbox=bbox,              # バウンディングボックス [N, 4]
        image_path=image_paths  # 画像名リスト [N]
    )

    print(f"🎉 正常に変換が完了しました: {output_npz_path}")

    # ---------------------------------------------------------
    # 検証: 出力された npz の中身を確認表示
    # ---------------------------------------------------------
    data = np.load(output_npz_path)
    print("\n--- 📦 生成された .npz 内部構造 ---")
    for key in data.files:
        print(f"  - {key:<12}: shape = {data[key].shape}, dtype = {data[key].dtype}")

if __name__ == '__main__':
    # 実行
    convert_blender_csv_to_motionbert_npz(
        csv_2d_path='test_2d_annotation.csv',
        csv_3d_path='test_3d_annotation.csv',
        output_npz_path='judo_train.npz',
        num_joints=19
    )