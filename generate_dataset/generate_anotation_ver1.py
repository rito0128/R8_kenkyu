import bpy
import math
import os
import glob
import numpy as np
from mathutils import Vector, Quaternion, Matrix
from bpy_extras.object_utils import world_to_camera_view

# --- 設定 ---
CAMERA_NAMES = ["Camera1"]
OUTPUT_DIR = "C:/Users/sinki/R8_kenkyu/generate_dateset/ganerated_demo_img/"

# アノテーションデータの書き出し先ファイル
OUTPUT_2d = os.path.join(OUTPUT_DIR, 'test_2d_annotation.npz')
OUTPUT_3d = os.path.join(OUTPUT_DIR, 'test_3d_annotation.npz')

ARMATURE_NAME = "Armature"

# レンダリング画像の設定
IMAGE_FORMAT = 'PNG'
RESOLUTION_X = 1000
RESOLUTION_Y = 1000

# キーポイントインデックスとボーン名の対応付け
BONE_INDEX_MAP = {
    'mixamorig:Hips': 7, 'mixamorig:Spine': 8, 'mixamorig:Neck': 9, 'mixamorig:Head': 10,
    'waist.001.l': 4, 'waist.001.r': 1, 'mixamorig:LeftShoulder': 11, 'mixamorig:LeftArm': 12,
    'mixamorig:LefForcetArm': 13, 'mixamorig:RightShoulder': 14, 'mixamorig:RightArm': 15, 'mixamorig:RightForceArm': 16,
    'leg.001.l': 5, 'leg.002.l': 6, 'feet.001.l': 0, 'leg.001.r': 2,
    'leg.002.r': 3, 'feet.001.r': 0, 'mixamorig:Spine1': 17, 'mixamorig:Spine2': 18
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

def arrange_and_collect_keypoint(keypoint_data):
    """辞書データをソートされたNumPy配列(1, 17, N)に整形する"""
    sorted_keys = sorted(keypoint_data.keys(), key=lambda x: int(x))
    two_d_list = [keypoint_data[k] for k in sorted_keys]
    return np.array(two_d_list)[np.newaxis, ...]  # (1, 17, N) に拡張

def process_animation_frames():
    """アーマチュアのアニメーションを1フレームずつ進めてデータを抽出・保存する"""
    scene = bpy.context.scene
    armature = bpy.data.objects.get(ARMATURE_NAME)
    
    if not armature or not armature.animation_data or not armature.animation_data.action:
        print(f"❌ エラー: アニメーションが適用されたアーマチュア '{ARMATURE_NAME}' が見つかりません。")
        return

    action = armature.animation_data.action
    start_frame = int(action.frame_range[0])
    end_frame = int(action.frame_range[1])
    
    print(f"🎬 モモーション検出: {action.name} (Frames: {start_frame} to {end_frame})")
    setup_render_settings(scene, OUTPUT_DIR, IMAGE_FORMAT)

    # 既存の出力ファイルを初期化（新規書き込み用）
    if os.path.exists(OUTPUT_2d): os.remove(OUTPUT_2d)
    if os.path.exists(OUTPUT_3d): os.remove(OUTPUT_3d)

    all_frames_2d = []
    all_frames_3d = []

    # --- フレームループ ---
    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)  # シーンのフレームを変更（ポーズが自動更新される）
        print(f"Processing Frame: {frame}/{end_frame}")
        
        frame_2d_data = None
        frame_3d_data = None

        for camera_name in CAMERA_NAMES:
            camera = bpy.data.objects.get(camera_name)
            if not camera or camera.type != 'CAMERA': continue
            
            scene.camera = camera
            # 依存グラフの更新（これがないと正しいボーン位置が取得できない場合がある）
            bpy.context.view_layer.update()

            # 2D/3D 座標抽出
            kp2d = get_keypoint2d(scene, camera, ARMATURE_NAME)
            kp3d = get_keypoint3d(scene, camera, ARMATURE_NAME)
            
            if kp2d and kp3d:
                frame_2d_data = arrange_and_collect_keypoint(kp2d)
                frame_3d_data = arrange_and_collect_keypoint(kp3d)

            # --- 必要に応じてここでレンダリングを実行 ---
            formatted_number = f"{frame:04d}"
            scene.render.filepath = f"{OUTPUT_DIR}out_{camera_name}_{formatted_number}"
            bpy.ops.render.render(write_still=True)

        if frame_2d_data is not None and frame_3d_data is not None:
            all_frames_2d.append(frame_2d_data)
            all_frames_3d.append(frame_3d_data)

    # --- 全フレームのデータを結合してNPZに一括保存 ---
    if all_frames_2d and all_frames_3d:
        final_2d_array = np.concatenate(all_frames_2d, axis=0) # (N, 17, 3)
        final_3d_array = np.concatenate(all_frames_3d, axis=0) # (N, 17, 4)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.savez_compressed(OUTPUT_2d, keypoints_2d=final_2d_array)
        np.savez_compressed(OUTPUT_3d, S=final_3d_array)
        
        print(f"💾 保存完了!\n2D: {final_2d_array.shape} -> {OUTPUT_2d}\n3D: {final_3d_array.shape} -> {OUTPUT_3d}")

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
    
    process_animation_frames()