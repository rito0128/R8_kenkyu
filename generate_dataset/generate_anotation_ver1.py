import bpy
import math
import os
import glob
import time
import numpy as np
import csv
from mathutils import Vector, Quaternion, Matrix
from bpy_extras.object_utils import world_to_camera_view

# --- 設定 ---
CAMERA_NAMES = ["Camera1", "Camera2", "Camera3", "Camera4", "Camera5", "Camera6", "Camera7", "Camera8", "Camera9", "Camera10", "Camera11", "Camera12"]
OUTPUT_DIR = "C:/Users/sinki/R8_kenkyu/generated_demo_img/"

# アノテーションデータの書き出し先ファイル
OUTPUT_2d = os.path.join(OUTPUT_DIR, 'test_2d_annotation.csv')
OUTPUT_3d = os.path.join(OUTPUT_DIR, 'test_3d_annotation.csv')

ARMATURE_NAME = "Armature"

# レンダリング画像の設定
IMAGE_FORMAT = 'PNG'
RESOLUTION_X = 1000
RESOLUTION_Y = 1000

# キーポイントインデックスとボーン名の対応付け
BONE_INDEX_MAP = {
    'mixamorig:Hips': 7, 'mixamorig:Spine': 8, 'mixamorig:Neck': 9, 'mixamorig:Head': 10,
    'mixamorig:LeftShoulder': 11, 'mixamorig:LeftArm': 12, 'mixamorig:LeftForeArm': 13,
    'mixamorig:RightShoulder': 14, 'mixamorig:RightArm': 15, 'mixamorig:RightForeArm': 16,
    'mixamorig:LeftUpLeg': 5, 'mixamorig:LeftLeg': 6, 'mixamorig:LeftFoot': 0, 'mixamorig:RightUpLeg': 2,
    'mixamorig:RightLeg': 3, 'mixamorig:RightFoot': 0, 'mixamorig:Spine1': 17, 'mixamorig:Spine2': 18
}
# --------------------

def setup_render_settings(scene, output_dir, format):
    """シーンのレンダリング基本設定を行う"""
    scene.render.image_settings.file_format = format
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y

def check_visibility(scene, camera, target_world_location):
    """カメラからターゲット点に向かってレイを飛ばし、メッシュに遮られているか判定する"""
    cam_location = camera.matrix_world.to_translation()
    direction = target_world_location - cam_location
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # ターゲットの直前でヒットするか判定 (小さなオフセットを持たせる)
    hit, loc, norm, index, obj, matrix = scene.ray_cast(
        depsgraph, cam_location, direction, distance=direction.length - 0.01
    )
    return 0 if hit else 1

def get_keypoint2d(scene, camera, armature_name):
    """ボーンのワールド座標を画像平面のピクセル座標(と可視性)に変換"""
    keypoint_2d = {}
    obj = bpy.data.objects.get(armature_name)
    if not obj: return None

    matrix_world = obj.matrix_world

    for pbone in obj.pose.bones:
        if pbone.name not in BONE_INDEX_MAP: continue
        
        tail_world = matrix_world @ pbone.tail
        tail_view = world_to_camera_view(scene, camera, tail_world)
        tail_px = (tail_view.x * RESOLUTION_X, (1.0 - tail_view.y) * RESOLUTION_Y)

        # 可視性判定
        if not (0 <= tail_view.x <= 1 and 0 <= tail_view.y <= 1 and tail_view.z > 0):
            visibility = 0
        else:
            visibility = check_visibility(scene, camera, tail_world)

        keypoint_2d[BONE_INDEX_MAP[pbone.name]] = (tail_px[0], tail_px[1], visibility)
        
        # 特殊処理: spine.001 の head をインデックス 0 と判定する場合
        if pbone.name == "spine.001":
            head_world = matrix_world @ pbone.head
            head_view = world_to_camera_view(scene, camera, head_world)
            head_px = (head_view.x * RESOLUTION_X, (1.0 - head_view.y) * RESOLUTION_Y)
            head_vis = check_visibility(scene, camera, head_world) if (0 <= head_view.x <= 1 and 0 <= head_view.y <= 1 and head_view.z > 0) else 0
            keypoint_2d[0] = (head_px[0], head_px[1], head_vis)

    return keypoint_2d

def get_keypoint3d(scene, camera, armature_name):
    """ボーン座標をワールド3D座標(と可視性)で取得"""
    keypoint_3d = {}
    obj = bpy.data.objects.get(armature_name)
    if not obj: return None

    matrix_world = obj.matrix_world

    for pbone in obj.pose.bones:
        if pbone.name not in BONE_INDEX_MAP: continue
        
        tail_world = matrix_world @ pbone.tail
        tail_view = world_to_camera_view(scene, camera, tail_world)
        visibility = check_visibility(scene, camera, tail_world) if (0 <= tail_view.x <= 1 and 0 <= tail_view.y <= 1 and tail_view.z > 0) else 0

        keypoint_3d[BONE_INDEX_MAP[pbone.name]] = (tail_world.x, tail_world.y, tail_world.z, visibility)
        
        if pbone.name == "spine.001":
            head_world = matrix_world @ pbone.head
            head_view = world_to_camera_view(scene, camera, head_world)
            head_vis = check_visibility(scene, camera, head_world) if (0 <= head_view.x <= 1 and 0 <= head_view.y <= 1 and head_view.z > 0) else 0
            keypoint_3d[0] = (head_world.x, head_world.y, head_world.z, head_vis)

    return keypoint_3d

def flatten_keypoint_data(keypoint_data):
    """辞書データをキーポイントのナンバリング順(昇順)に平坦なリストへ展開する"""
    sorted_keys = sorted(keypoint_data.keys(), key=lambda x: int(x))
    flat_list = []
    for k in sorted_keys:
        flat_list.extend(keypoint_data[k])  # (x, y, vis) または (X, Y, Z, vis) を1つのリストに追加
    return flat_list

def generate_csv_headers(is_3d=False):
    """可読性とMotionBERT連携のためのCSVヘッダーを作成"""
    headers = ["filename"]
    num_joints = max(BONE_INDEX_MAP.values()) + 1  # インデックスの最大値から関節数を定義
    coords = ["x", "y", "z", "visibility"] if is_3d else ["x", "y", "visibility"]
    
    for j in range(num_joints):
        for c in coords:
            headers.append(f"J{j}_{c}")
    return headers

def process_animation_frames():
    """アーマチュアのアニメーションを1フレームずつ進めてデータを抽出・保存する"""
    scene = bpy.context.scene
    armature = bpy.data.objects.get(ARMATURE_NAME)
    
    if not armature or not armature.animation_data or not armature.animation_data.action:
        print(f"❌ エラー: アニメーションが適用されたアーマチュア '{ARMATURE_NAME}' が見つかりません。")
        return

    action = armature.animation_data.action
    
    # ====================================================================
    # 【確実な対策】HipsボーンのX移動(array_index=0)とY移動(array_index=1)のFカーブを削除
    # ====================================================================
    fcurves_to_remove = []
    for fc in action.fcurves:
        # mixamorig:Hipsボーンの位置アニメーションを探す
        if 'pose.bones["mixamorig:Hips"].location' in fc.data_path:
            # array_index: 0=X, 1=Y, 2=Z。 XとY（水平移動）のデータを削除対象にする
            if fc.array_index in [0, 1]:
                fcurves_to_remove.append(fc)
                
    for fc in fcurves_to_remove:
        action.fcurves.remove(fc)
    print(f"🧹 Hipsボーンの水平移動データをアニメーションから削除しました。")
    # ====================================================================
    
    start_frame = int(action.frame_range[0])
    end_frame = int(action.frame_range[1])
    
    print(f"🎬 モモーション検出: {action.name} (Frames: {start_frame} to {end_frame})")
    setup_render_settings(scene, OUTPUT_DIR, IMAGE_FORMAT)

    # --- CSVファイルの新規オープンとヘッダー初期化 ---
    with open(OUTPUT_2d, mode='w', newline='', encoding='utf-8') as f2d, \
         open(OUTPUT_3d, mode='w', newline='', encoding='utf-8') as f3d:
         
        writer_2d = csv.writer(f2d)
        writer_3d = csv.writer(f3d)
        
        # ヘッダー行を書き込み
        writer_2d.writerow(generate_csv_headers(is_3d=False))
        writer_3d.writerow(generate_csv_headers(is_3d=True))

        # --- フレームループ ---
        for camera_name in CAMERA_NAMES:
            camera = bpy.data.objects.get(camera_name)
            if not camera or camera.type != 'CAMERA': 
                print(f"⚠️ 警告: カメラ '{camera_name}' が見つからないためスキップします。")
                continue
            
            # カメラを切り替える
            scene.camera = camera
            print(f"📷 カメラを切り替えました: {camera_name} (モーション全体の連続処理を開始)")

            # 現在のカメラでモーションを最初のフレームから最後まで連続処理
            for frame in range(start_frame, end_frame + 1):
                scene.frame_set(frame)  # シーンのフレームを変更（ポーズが自動更新される）
                bpy.context.view_layer.update()  # 依存グラフの更新

                # 2D/3D 座標辞書の取得
                kp2d = get_keypoint2d(scene, camera, ARMATURE_NAME)
                kp3d = get_keypoint3d(scene, camera, ARMATURE_NAME)
                
                # 画像名（先頭カラムの値）を定義：例「out_Camera1_0001」
                formatted_number = f"{frame:04d}"
                image_filename = f"out_{camera_name}_{formatted_number}"
                
                # レンダリング実行
                scene.render.filepath = f"{OUTPUT_DIR}{image_filename}"
                bpy.ops.render.render(write_still=True)
                print(f"  └ Rendered: {image_filename}")

                # データのフラット化とCSVへの即時一行書き込み
                if kp2d:
                    row_2d = [image_filename] + flatten_keypoint_data(kp2d)
                    writer_2d.writerow(row_2d)
                    
                if kp3d:
                    row_3d = [image_filename] + flatten_keypoint_data(kp3d)
                    writer_3d.writerow(row_3d)

    print(f"💾 CSV保存完了!\n2D: {OUTPUT_2d}\n3D: {OUTPUT_3d}")

if __name__ == "__main__":
    # ---- 実行前にボーン名の一致チェック ----
    armature = bpy.data.objects.get(ARMATURE_NAME)
    if armature:
        print("\n--- 【デバッグ】Blender内のボーン名一覧 ---")
        blender_bones = [b.name for b in armature.pose.bones]
        print(blender_bones)
        
        print("\n--- 【デバッグ】一致チェック結果 ---")
        match_count = 0
        for b_name in BONE_INDEX_MAP.keys():
            if b_name in blender_bones:
                match_count += 1
            else:
                print(f"⚠️ マッチしないボーン: {b_name}")
        print(f"結果: {len(BONE_INDEX_MAP)}個中 {match_count}個が一致\n")
    # ----------------------------------------
    
    start_time = time.time()

    process_animation_frames()

    end_time = time.time()
    
    # 4. 差分を計算して表示
    elapsed_time = end_time - start_time
    print(f"\n⏱️ 処理時間: {elapsed_time:.4f} 秒")