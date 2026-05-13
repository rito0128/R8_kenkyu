import numpy as np

def load_and_display_npz(file_path):
    try:
        # ファイルの読み込み
        # allow_pickle=Trueは、オブジェクト配列が含まれる場合に必要です
        with np.load(file_path, allow_pickle=True) as data:
            # 1. 含まれているキー（変数名）の一覧を表示
            keys = data.files
            print(f"--- ファイル構成: {file_path} ---")
            print(f"含まれるキー: {keys}\n")

            # 2. 各データの中身をループで表示
            for key in keys:
                content = data[key]
                print(f"Key: {key}")
                print(f"Shape: {content.shape}")
                print(f"Data:\n{content}")
                print("-" * 30)

    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

# 実行例
# load_and_display_npz('your_file.npz')