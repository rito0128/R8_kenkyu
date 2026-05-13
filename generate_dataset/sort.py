import os
import glob

# 1. 実行したい関数を定義
def my_process(file_path):
    print("処理したよ")
    # ここに実際の処理（データの読み込みなど）を書く
    pass

# 2. 対象のディレクトリと拡張子を指定
target_dir = './npz_files'
extension = 'npz'  # 取得したい拡張子

# 3. ファイルパスのリストを取得してソート（名前順）
# glob.globを使うとワイルドカードが使えて便利です
files = sorted(glob.glob(os.path.join(target_dir, f'*.{extension}')))

# 4. ループで関数に渡す
if not files:
    print("ファイルが見つかりませんでした。")
else:
    for f in files:
        print(str(f))
        my_process(f)

print("すべての処理が完了しました。")