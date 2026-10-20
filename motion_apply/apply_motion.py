"""
Mixamo のモーション(FBX)を .blend 内の Armature に一括適用するランチャー。

ディレクトリ構成:
    motion_apply/
        base_model.blend          ← 作業用 .blend（複数あればすべて処理）
        motion/*.fbx              ← Mixamo からダウンロードした FBX
        output/<blend名>_<モーション名>.blend  ← 生成物

実行:
    python apply_motion.py
"""
import os
import glob
import shutil
import subprocess

# --- 設定 ---
# Blenderの実行ファイルパス（インストール環境に合わせて変更してください）
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MOTION_DIR = os.path.join(WORK_DIR, "motion")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
BLENDER_SCRIPT = os.path.join(WORK_DIR, "apply_motion_blender.py")

# モーションを適用するArmatureのオブジェクト名（None なら .blend 内の唯一のArmatureを自動選択）
TARGET_ARMATURE = None

# True なら出力済みの .blend はスキップ
SKIP_EXISTING = False


def main():
    blend_files = sorted(glob.glob(os.path.join(WORK_DIR, "*.blend")))
    fbx_files = sorted(glob.glob(os.path.join(MOTION_DIR, "*.fbx")))

    if not blend_files:
        print(f"❌ '{WORK_DIR}' に .blend ファイルが見つかりませんでした。")
        return
    if not fbx_files:
        print(f"❌ '{MOTION_DIR}' に .fbx ファイルが見つかりませんでした。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    jobs = [(b, f) for b in blend_files for f in fbx_files]
    print(f"📂 .blend {len(blend_files)} 個 × モーション {len(fbx_files)} 個 = {len(jobs)} ファイルを生成します。\n")

    failed = []
    for i, (blend_file, fbx_file) in enumerate(jobs, 1):
        blend_stem = os.path.splitext(os.path.basename(blend_file))[0]
        motion_stem = os.path.splitext(os.path.basename(fbx_file))[0]
        out_name = f"{blend_stem}_{motion_stem.replace(' ', '_')}.blend"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        print("==================================================")
        print(f"[{i}/{len(jobs)}] {motion_stem} → {out_name}")
        print("==================================================")

        if SKIP_EXISTING and os.path.exists(out_path):
            print("⏭ 出力済みのためスキップ\n")
            continue

        # 作業用 .blend を複製し、複製側にモーションを適用して上書き保存する
        shutil.copy2(blend_file, out_path)
        cmd = [BLENDER_PATH, "-b", out_path, "--python-exit-code", "1",
               "-P", BLENDER_SCRIPT, "--", fbx_file]
        if TARGET_ARMATURE:
            cmd.append(TARGET_ARMATURE)

        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"✅ 完了: {out_name}\n")
        else:
            print(f"❌ エラー発生 ({out_name}): 終了コード {result.returncode}\n")
            failed.append(out_name)
            if os.path.exists(out_path):
                os.remove(out_path)  # 適用に失敗したコピーは残さない

    print(f"🎉 終了: 成功 {len(jobs) - len(failed)} / {len(jobs)}")
    if failed:
        print("失敗:", ", ".join(failed))


if __name__ == "__main__":
    main()
