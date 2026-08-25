import os
import glob
import pandas as pd

# --- 設定 ---
# 結合対象のCSVファイルが格納されているルートフォルダ
BASE_DIR = r"C:\Users\sinki\R8_kenkyu\generated_demo_img"

# 出力先の設定
OUTPUT_2D_MERGED = os.path.join(BASE_DIR, "merged_2d_annotation.csv")
OUTPUT_3D_MERGED = os.path.join(BASE_DIR, "merged_3d_annotation.csv")

def merge_csv_files(file_pattern, output_filepath):
    """
    指定されたパターンに一致するすべてのCSVファイルを検索し、
    ヘッダーの列名に合わせて縦方向に結合して保存する
    """
    # サブフォルダ内も含めて再帰的にCSVを検索
    csv_files = glob.glob(os.path.join(BASE_DIR, "**", file_pattern), recursive=True)
    
    # 既に出力先の結合ファイルが存在する場合は除外
    csv_files = [f for f in csv_files if os.path.abspath(f) != os.path.abspath(output_filepath)]

    if not csv_files:
        print(f"⚠️ 対象ファイルが見つかりませんでした: {file_pattern}")
        return

    print(f"🔍 {len(csv_files)} 個のファイルを検出 ({file_pattern})")

    df_list = []
    for filepath in sorted(csv_files):
        try:
            # CSVの読み込み
            df = pd.read_csv(filepath)
            df_list.append(df)
            print(f"  └ 読込完了: {filepath} ({len(df)} 行)")
        except Exception as e:
            print(f"  ❌ 読込エラー ({filepath}): {e}")

    if df_list:
        # 列名を基準に縦方向に結合 (行を追加)
        merged_df = pd.concat(df_list, axis=0, ignore_index=True)
        
        # 結合結果をCSVとして保存
        merged_df.to_csv(output_filepath, index=False, encoding='utf-8')
        print(f"✅ 保存完了: {output_filepath} (合計: {len(merged_df)} 行, {len(merged_df.columns)} 列)\n")

if __name__ == "__main__":
    print("=== CSV結合処理を開始 ===")
    
    # 2DアノテーションCSVの結合 (*_2d_annotation.csv または test_2d_annotation.csv など)
    merge_csv_files("*2d_annotation.csv", OUTPUT_2D_MERGED)
    
    # 3DアノテーションCSVの結合 (*_3d_annotation.csv または test_3d_annotation.csv など)
    merge_csv_files("*3d_annotation.csv", OUTPUT_3D_MERGED)
    
    print("=== すべての結合処理が完了しました ===")