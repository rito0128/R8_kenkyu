import bpy
import os

def apply_fbx_motion(fbx_path, target_armature_name):
    # 1. FBXをインポート
    # 読み込み時に一時的に選択状態になるのを利用
    # bpy.ops.import_scene.fbx(filepath=fbx_path)
    bpy.ops.import_scene.fbx(filepath=fbx_path, global_scale=0.01)
    imported_objs = bpy.context.selected_objects
    
    # インポートされたアーマチュアを探す
    source_armature = None
    for obj in imported_objs:
        if obj.type == 'ARMATURE':
            source_armature = obj
            break
            
    if not source_armature:
        print("FBX内にアーマチュアが見つかりませんでした。")
        return

    # 2. ターゲットのアバターを取得
    target_armature = bpy.data.objects.get(target_armature_name)
    if not target_armature:
        print(f"ターゲット '{target_armature_name}' が見つかりません。")
        return

    # 3. アニメーションデータ（Action）の取得と適用
    if source_armature.animation_data and source_armature.animation_data.action:
        motion_action = source_armature.animation_data.action
        
        # ターゲットにアニメーションデータを設定するための準備
        if not target_armature.animation_data:
            target_armature.animation_data_create()
            
        # アクションを割り当て
        target_armature.animation_data.action = motion_action
        print(f"アクション '{motion_action.name}' を {target_armature.name} に適用しました。")
    else:
        print("ソースFBXにアニメーションが含まれていません。")

    # 4. 不要になったインポートオブジェクト（ソース）を削除
    # （必要に応じてコメントアウトしてください）
    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported_objs:
        obj.select_set(True)
    bpy.ops.object.delete()

# --- 設定項目 ---
fbx_file_path = r"C:\Users\a24k0\R8_kenkyu\generate_dataset\fbx_motion\usiro_ukemi.fbx"  # FBXファイルのフルパス
avatar_name = "Armature"            # シーン内にあるアバターのオブジェクト名

apply_fbx_motion(fbx_file_path, avatar_name)