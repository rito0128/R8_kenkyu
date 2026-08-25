import os
import glob
import subprocess

# --- 設定 ---
# Blenderの実行ファイルパス（インストール環境に合わせて変更してください）
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

# 対象の .blend ファイルが入っているフォルダ
BLEND_DIR = r"C:\Users\sinki\R8_kenkyu\generate_dataset\blender_file\run"

# 実行するBlenderスクリプトのフルパス
SCRIPT_PATH = r"C:\Users\sinki\R8_kenkyu\generate_dataset\generate_anotation_ver3.py"

def main():
    blend_files = glob.glob(os.path.join(BLEND_DIR, "*.blend"))
    
    if not blend_files:
        print(f"❌ '{BLEND_DIR}' に .blend ファイルが見つかりませんでした。")
        return

    print(f"📂 合計 {len(blend_files)} 個の .blend ファイルを処理します。\n")

    for i, blend_file in enumerate(blend_files, 1):
        blend_name = os.path.basename(blend_file)
        print(f"==================================================")
        print(f"[{i}/{len(blend_files)}] 処理開始: {blend_name}")
        print(f"==================================================")

        # Blenderをバックグラウンド(-b)で起動し、スクリプト(-P)を実行
        cmd = [
            BLENDER_PATH,
            "-b", blend_file,
            "-P", SCRIPT_PATH
        ]

        # 実行（エラーが出ても次のファイルに進めるように例外処理）
        try:
            result = subprocess.run(cmd, check=True)
            print(f"✅ 完了: {blend_name}\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ エラー発生 ({blend_name}): 終了コード {e.returncode}\n")
        except Exception as e:
            print(f"❌ 予期せぬエラー ({blend_name}): {e}\n")

if __name__ == "__main__":
    main()