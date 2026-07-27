import bpy
import math
import os
import glob
import time
import numpy as np
import csv

import sys
sys.path.append("C:\\Users\\sinki\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\cv2\\")
import cv2

from mathutils import Vector, Quaternion, Matrix
from bpy_extras.object_utils import world_to_camera_view

# --- 設定 ---
CAMERA_NAMES = ["Camera1"]
# CAMERA_NAMES = ["Camera1", "Camera2", "Camera3", "Camera4", "Camera5", "Camera6", "Camera7", "Camera8", "Camera9", "Camera10", "Camera11", "Camera12"]
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

def check_visibility_multi_ray(scene, camera, target_world_location, samples=5):
    """
    ターゲット周辺から複数のレイを飛ばし、可視割合(0.0 ~ 1.0)または段階評価(0, 1, 2)を返す
    """
    cam_matrix = camera.matrix_world
    cam_location = cam_matrix.to_translation()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # ターゲットからカメラ方向への単位ベクトル
    to_cam = (cam_location - target_world_location).normalized()
    
    # 1. 骨の内部（肉の中）にレイが埋まるのを防ぐため、カメラ側に1.5cm手前に寄せる
    adjusted_target = target_world_location + (to_cam * 0.015)
    
    # カメラの「右方向(X)」と「上方向(Y)」の基底ベクトルを取得（ワールド座標系）
    cam_right = cam_matrix.to_3x3() @ Vector((1, 0, 0))
    cam_up = cam_matrix.to_3x3() @ Vector((0, 1, 0))
    
    # 2. カメラの画角平面に沿った安全なオフセット（半径5mm）
    r = 0.005
    offsets = [
        Vector((0, 0, 0)),
        cam_right * r,
        cam_right * (-r),
        cam_up * r,
        cam_up * (-r)
    ]
    
    visible_hits = 0
    
    for offset_vec in offsets[:samples]:
        sample_target = adjusted_target + offset_vec
        direction = sample_target - cam_location
        dist = direction.length
        
        if dist < 0.001:
            visible_hits += 1
            continue
            
        # レイを飛ばす（ターゲットの2cm手前まででヒットするか判定して余裕を持たせる）
        hit, loc, norm, index, obj, matrix = scene.ray_cast(
            depsgraph, 
            cam_location, 
            direction.normalized(), 
            distance=dist - 0.2
        )
        
        if not hit:
            visible_hits += 1

    # 可視率を計算
    vis_ratio = visible_hits / samples
    
    # COCOフォーマット等の定義に合わせて判定
    if vis_ratio >= 0.6:
        return 2  # Visible（はっきり見える）
    elif vis_ratio >= 0.2:
        return 1  # Partially visible / Occluded（一部見えている/遮蔽）
    else:
        return 0  # Not visible（見えない）

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
            visibility = check_visibility_multi_ray(scene, camera, tail_world)

        keypoint_2d[BONE_INDEX_MAP[pbone.name]] = (tail_px[0], tail_px[1], visibility)
        
        # 特殊処理: spine.001 の head をインデックス 0 と判定する場合
        if pbone.name == "spine.001":
            head_world = matrix_world @ pbone.head
            head_view = world_to_camera_view(scene, camera, head_world)
            head_px = (head_view.x * RESOLUTION_X, (1.0 - head_view.y) * RESOLUTION_Y)
            head_vis = check_visibility_multi_ray(scene, camera, head_world) if (0 <= head_view.x <= 1 and 0 <= head_view.y <= 1 and head_view.z > 0) else 0
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
        visibility = check_visibility_multi_ray(scene, camera, tail_world) if (0 <= tail_view.x <= 1 and 0 <= tail_view.y <= 1 and tail_view.z > 0) else 0

        keypoint_3d[BONE_INDEX_MAP[pbone.name]] = (tail_world.x, tail_world.y, tail_world.z, visibility)
        
        if pbone.name == "spine.001":
            head_world = matrix_world @ pbone.head
            head_view = world_to_camera_view(scene, camera, head_world)
            head_vis = check_visibility_multi_ray(scene, camera, head_world) if (0 <= head_view.x <= 1 and 0 <= head_view.y <= 1 and head_view.z > 0) else 0
            keypoint_3d[0] = (head_world.x, head_world.y, head_world.z, head_vis)

    return keypoint_3d

def flatten_keypoint_data(keypoint_data):
    """辞書データをキーポイントのナンバリング順(昇順)に平坦なリストへ展開する"""
    sorted_keys = sorted(keypoint_data.keys(), key=lambda x: int(x))
    flat_list = []
    for k in sorted_keys:
        flat_list.extend(keypoint_data[k])  # (x, y, vis) または (X, Y, Z, vis) を1つのリストに追加
    return flat_list

# --- 【変更点1】ヘッダー作成関数に BBOX 用カラムを追加 ---
def generate_csv_headers(is_3d=False):
    """可読性とMotionBERT連携のためのCSVヘッダーを作成"""
    headers = ["filename"]
    num_joints = max(BONE_INDEX_MAP.values()) + 1  # インデックスの最大値から関節数を定義
    coords = ["x", "y", "z", "visibility"] if is_3d else ["x", "y", "visibility"]
    
    for j in range(num_joints):
        for c in coords:
            headers.append(f"J{j}_{c}")
            
    # 各行の末尾に BBOX カラムを追加
    headers.extend(["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"])
    return headers

def get_avatar_bounding_box_2d(scene, camera, armature_name):
    """
    アーマチュアに属するすべての子メッシュの頂点から、
    カメラに映る正確な2Dバウンディングボックス(xmin, ymin, xmax, ymax)を算出する
    """
    armature_obj = bpy.data.objects.get(armature_name)
    if not armature_obj:
        return None

    x_coords = []
    y_coords = []
    
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for child in armature_obj.children:
        if child.type != 'MESH':
            continue
            
        eval_obj = child.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        matrix_world = eval_obj.matrix_world

        for vertex in mesh.vertices:
            v_world = matrix_world @ vertex.co
            v_view = world_to_camera_view(scene, camera, v_world)
            
            if 0 <= v_view.x <= 1 and 0 <= v_view.y <= 1 and v_view.z > 0:
                px = v_view.x * RESOLUTION_X
                py = (1.0 - v_view.y) * RESOLUTION_Y
                x_coords.append(px)
                y_coords.append(py)
                
        eval_obj.to_mesh_clear()

    if not x_coords:
        return None

    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def draw_and_save_plots_cv2(image_path, kp2d, bbox):
    """保存された画像をOpenCVで読み込み、可視性(信頼度)に応じた色でキーポイントとインデックスを描画して別名保存する"""
    if not kp2d or not os.path.exists(image_path):
        return

    # 画像を読み込む (OpenCVはBGR形式)
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # バウンディングボックスの描画
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        pt1 = (int(xmin), int(ymin))
        pt2 = (int(xmax), int(ymax))
        cv2.rectangle(img, pt1, pt2, color=(0, 0, 255), thickness=2)
        cv2.putText(img, "Person", (int(xmin), int(ymin) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    for idx, (px, py, vis) in kp2d.items():
        # --- 【変更点】 信頼度（vis）の値に応じて描画色を分岐 (OpenCVはBGR順) ---
        if vis == 1:
            plot_color = (0, 255, 0)     # 緑色
        elif vis == 0:
            plot_color = (0, 165, 255)   # オレンジ色
        else:
            plot_color = (255, 0, 0)     # 青色 (信頼度2など)
        # -------------------------------------------------------------------

        # 座標を整数型にキャスト
        center = (int(px), int(py))
            
        # 1. キーポイントの点を描画 (中心座標, 半径4ピクセル, 塗りつぶし=-1)
        cv2.circle(img, center, radius=4, color=plot_color, thickness=-1)
            
        # 2. インデックス番号の描画 (点の少し右上へずらす。フォントサイズ0.4, 太さ1)
        text_pos = (int(px) + 6, int(py) - 6)
        cv2.putText(img, str(idx), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, plot_color, 1, cv2.LINE_AA)

    # 元のファイルパスから拡張子を分離し、末尾に _ploted を追加
    base, ext = os.path.splitext(image_path)
    output_plot_path = f"{base}_ploted{ext}"
    
    # 描画済みの別名画像を保存
    cv2.imwrite(output_plot_path, img)
    print(f"  └ Plotted (cv2): {os.path.basename(output_plot_path)}")

def process_animation_frames():
    """アーマチュアのアニメーションを1フレームずつ進めてデータを抽出・保存する"""
    scene = bpy.context.scene
    armature = bpy.data.objects.get(ARMATURE_NAME)
    
    if not armature or not armature.animation_data or not armature.animation_data.action:
        print(f"❌ エラー: アニメーションが適用されたアーマチュア '{ARMATURE_NAME}' が見つかりません。")
        return

    action = armature.animation_data.action
    
    fcurves_to_remove = []
    for fc in action.fcurves:
        if 'pose.bones["mixamorig:Hips"].location' in fc.data_path:
            if fc.array_index in [0, 1]:
                fcurves_to_remove.append(fc)
                
    for fc in fcurves_to_remove:
        action.fcurves.remove(fc)
    print(f"🧹 Hipsボーンの水平移動データをアニメーションから削除しました。")
    
    start_frame = int(action.frame_range[0])
    end_frame = int(action.frame_range[1])
    
    print(f"🎬 モーション検出: {action.name} (Frames: {start_frame} to {end_frame})")
    setup_render_settings(scene, OUTPUT_DIR, IMAGE_FORMAT)

    with open(OUTPUT_2d, mode='w', newline='', encoding='utf-8') as f2d, \
         open(OUTPUT_3d, mode='w', newline='', encoding='utf-8') as f3d:
         
        writer_2d = csv.writer(f2d)
        writer_3d = csv.writer(f3d)
        
        writer_2d.writerow(generate_csv_headers(is_3d=False))
        writer_3d.writerow(generate_csv_headers(is_3d=True))

        for camera_name in CAMERA_NAMES:
            camera = bpy.data.objects.get(camera_name)
            if not camera or camera.type != 'CAMERA': 
                print(f"⚠️ 警告: カメラ '{camera_name}' が見つからないためスキップします。")
                continue
            
            scene.camera = camera
            print(f"📷 カメラを切り替えました: {camera_name} (モーション全体の連続処理を開始)")

            for frame in range(start_frame, end_frame + 1):
                scene.frame_set(frame)
                bpy.context.view_layer.update()

                kp2d = get_keypoint2d(scene, camera, ARMATURE_NAME)
                kp3d = get_keypoint3d(scene, camera, ARMATURE_NAME)
                
                bbox = get_avatar_bounding_box_2d(scene, camera, ARMATURE_NAME)
                
                # --- 【変更点2】BBOXデータをリスト形式に整形 ---
                bbox_data = list(bbox) if bbox else ["", "", "", ""]
                
                image_filename = f"out_{camera_name}_{frame:04d}"
                
                render_output_path = os.path.join(OUTPUT_DIR, image_filename)

                scene.render.filepath = render_output_path
                bpy.ops.render.render(write_still=True)
                print(f"  └ Rendered: {image_filename}")

                actual_image_path = f"{render_output_path}.{IMAGE_FORMAT.lower()}"
                draw_and_save_plots_cv2(actual_image_path, kp2d, bbox)

                # --- 【変更点3】各行の末尾に bbox_data を追加して書き込み ---
                if kp2d:
                    row_2d = [image_filename] + flatten_keypoint_data(kp2d) + bbox_data
                    writer_2d.writerow(row_2d)
                    
                if kp3d:
                    row_3d = [image_filename] + flatten_keypoint_data(kp3d) + bbox_data
                    writer_3d.writerow(row_3d)

    print(f"💾 CSV保存完了!\n2D: {OUTPUT_2d}\n3D: {OUTPUT_3d}")

if __name__ == "__main__":
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
    
    start_time = time.time()

    process_animation_frames()

    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"\n⏱️ 処理時間: {elapsed_time:.4f} 秒")