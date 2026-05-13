import bpy
import math
import os
import glob
import numpy as np
from mathutils import Vector, Quaternion, Matrix
from bpy_extras.object_utils import world_to_camera_view

# 【重要】キーポイントインデックスとボーン名の対応付け (省略せず記述)
BONE_INDEX_MAP = {
    'spine.001': 7,
    'spine.002': 8,
    'head.001': 9,
    'head.002': 10,
    'waist.001.l': 4,
    'waist.001.r': 1,
    'shoulder.001.l': 11,
    'arm.001.l': 12,
    'arm.002.l': 13,
    'shoulder.001.r': 14,
    'arm.001.r': 15,
    'arm.002.r': 16,
    'leg.001.l': 5,
    'leg.002.l': 6,
    'feet.001.l': 0,
    'leg.001.r': 2,
    'leg.002.r': 3,
    'feet.001.r': 0,
}

BONE_INDEX_MAP_REVERSE = {
    7: 'spine.001',
    8: 'spine.002',
    9: 'head.001',
    10: 'head.002',
    4: 'waist.001.l',
    1: 'waist.001.r',
    11: 'shoulder.001.l',
    12: 'arm.001.l',
    13: 'arm.002.l',
    14: 'shoulder.001.r',
    15: 'arm.001.r',
    16: 'arm.002.r',
    5: 'leg.001.l',
    6: 'leg.002.l',
    2: 'leg.001.r',
    3: 'leg.002.r'
}

PAIR_LIST = {
    7: 0,
    8: 7,
    11: 8,
    12: 11,
    13: 12,
    9: 8,
    10: 9,
    14: 8,
    15: 14,
    16: 15,
    4: 0,
    5: 4,
    6: 5,
    1: 0,
    2: 1,
    3: 2
}

# 回転の計算に用いる, 子ボーンのtail：親ボーンのtail
PARENT_LIST = {
    8: 7, #
    9: 8,
    10: 9,
    11: 8, #
    12: 11,
    13: 12,
    14: 8,
    15: 14,
    16: 15,
    1: 7, 
    2: 1,
    3: 2,
    4: 7, #
    5: 4,
    6: 5
}

# --- 設定 ---
# 4台のカメラの名称リスト
# シーン内の実際のカメラ名に合わせて変更してください。
# CAMERA_NAMES = ["CameraA", "CameraB", "CameraC", "CameraD"]
CAMERA_NAMES = ["Camera12"]

# 出力ディレクトリのパス (Blenderの実行環境に合わせて絶対パスを指定)
# 例: "/tmp/renders/" または "C:/Users/YourName/Desktop/renders/"
OUTPUT_DIR = "C:/Users/a24k0/R6_blender2/scripts/img/" # "//" はBlenderファイルからの相対パスを意味します。

# NPZ_FILEPATH = "C:/Users/a24k0/R6_blender2/scripts/keypoints.npz" 
TAGET_DIR = './m3_npz'
EXTENSION = 'npz'

# アノテーションデータの書き出し先ファイル
OUTPUT_2d = 'c12_output_2d_anotation.npz'
OUTPUT_3d = 'c12_output_3d_anotation.npz'

ARMATURE_NAME = "Armature"

# レンダリング画像の設定
IMAGE_FORMAT = 'PNG'
RESOLUTION_X = 1000
RESOLUTION_Y = 1000
# --------------------

# ====================================================================
# レンダリング
# ====================================================================
def setup_render_settings(scene, output_dir, format):
    """シーンのレンダリング基本設定を行う"""
    scene.render.image_settings.file_format = format
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.filepath = output_dir # 出力パスの基本設定

def render_from_multiple_cameras(ARMATURE_NAME, image_number):
    keypoint_2d = {}
    keypoint_3d = {}
    """複数のカメラから順番にレンダリングを実行するメイン関数"""
    scene = bpy.context.scene
    formatted_number = f"{image_number:04d}"
    
    # 1. 基本設定の適用
    setup_render_settings(scene, OUTPUT_DIR, IMAGE_FORMAT)
    
    # 2. カメラリストを反復処理
    for i, camera_name in enumerate(CAMERA_NAMES):
        camera = bpy.data.objects.get(camera_name)
        
        if camera and camera.type == 'CAMERA':
            #print(f"--- レンダリング開始: {camera_name} ({i + 1}/{len(CAMERA_NAMES)}) ---")
            
            # **現在のカメラをアクティブなシーンカメラとして設定**
            scene.camera = camera
            
            keypoint_2d = get_keypoint2d(scene, camera, ARMATURE_NAME)
            arrange_keypoint(keypoint_2d, OUTPUT_2d, 'keypoints_2d')
            keypoint_3d = get_keypoint3d(scene, camera, ARMATURE_NAME)
            arrange_keypoint(keypoint_3d, OUTPUT_3d, 'S')

            print("keypoint_2d")
            #print(keypoint_2d)
            print("keypoint_3d")
            #print(keypoint_3d)
            
            # **出力ファイル名をカメラごとに設定**
            # 例: //renders/output_Camera.001
            scene.render.filepath = f"{OUTPUT_DIR}output_{camera_name}_{formatted_number}"
            
            # **レンダリングの実行**
            # write_still=True: レンダリングが完了した後、ファイルに画像を保存します。
            bpy.ops.render.render(write_still=True)
            
            print(f"レンダリング完了。出力: {scene.render.filepath}")
        else:
            print(f"警告: カメラ '{camera_name}' が見つからないか、カメラオブジェクトではありません。スキップします。")

def get_keypoint2d(scene, camera, armature_name):
    keypoint_2d = {}

    """ボーンのワールド座標を画像平面のピクセル座標に変換して表示する"""
    obj = bpy.data.objects.get(armature_name)
    if not obj or obj.type != 'ARMATURE':
        print(f"❌ アーマチュア '{armature_name}' が見つかりません。")
        return

    matrix_world = obj.matrix_world
    #print(f"\n[Camera: {camera.name}] Bone 2D Coordinates (Pixel):")

    for pbone in obj.pose.bones:
        # 1. ワールド座標を計算
        head_world = matrix_world @ pbone.head
        tail_world = matrix_world @ pbone.tail

        # 2. 3Dワールド座標をカメラビューの2D比率(0.0 - 1.0)に変換
        # 返り値は Vector((x, y, z))。zはカメラからの深度（正ならカメラの前方）
        head_view = world_to_camera_view(scene, camera, head_world)
        tail_view = world_to_camera_view(scene, camera, tail_world)

        # 3. 比率をピクセル座標に変換
        # Blenderの画像座標系は左下が(0,0)のため、必要に応じて上下反転の処理を行う
        head_px = (head_view.x * RESOLUTION_X, (1.0 - head_view.y) * RESOLUTION_Y)
        tail_px = (tail_view.x * RESOLUTION_X, (1.0 - tail_view.y) * RESOLUTION_Y)

        # カメラの画角内（0.0~1.0の範囲内）にあるかチェック
        in_view = "In Frame" if (0 <= head_view.x <= 1 and 0 <= head_view.y <= 1) else "Out of Frame"

        keypoint_2d[BONE_INDEX_MAP[pbone.name]] = tail_px
        if pbone.name == "spine.001":
            keypoint_2d[0] = head_px

    return keypoint_2d
    
def get_keypoint3d(scene, camera, armature_name):
    keypoint_3d = {}

    """ボーン座標を『画像ピクセル座標』と『Root原点の3D座標』の両方で計算する"""
    obj = bpy.data.objects.get(armature_name)
    if not obj or obj.type != 'ARMATURE':
        print(f"❌ アーマチュア '{armature_name}' が見つかりません。")
        return

    matrix_world = obj.matrix_world
    
    # --- Rootボーンを基準とした変換行列の準備 ---
    # 一般的に最初のボーン(obj.pose.bones[0])をRootと見なすか、名前で指定します
    root_bone = obj.pose.bones.get("Root") or obj.pose.bones[0]
    # Rootボーンのヘッドのワールド座標系における位置・回転行列
    # (matrix_world @ bone.matrix) でワールド空間でのボーン行列が得られる
    root_world_matrix = matrix_world @ root_bone.matrix
    # その逆行列を作成することで「Rootから見た相対座標」へ変換可能にする
    inv_root_matrix = root_world_matrix.inverted()

    #print(f"\n[Camera: {camera.name}] Reference Root: {root_bone.name}")

    for pbone in obj.pose.bones:
        # 1. ワールド座標を計算 (絶対的な空間位置)
        head_world = matrix_world @ pbone.head
        tail_world = matrix_world @ pbone.tail

        # 2. Root基準の3D座標に変換
        # 逆行列を掛けることで、RootのHeadが(0,0,0)になる空間へ飛ばす
        head_root_rel = inv_root_matrix @ head_world
        tail_root_rel = inv_root_matrix @ tail_world

        # 3. カメラビューの比率座標 (0.0 - 1.0)
        head_view = world_to_camera_view(scene, camera, head_world)
        tail_view = world_to_camera_view(scene, camera, tail_world)


        keypoint_3d[BONE_INDEX_MAP[pbone.name]] = tail_world
        if pbone.name == "spine.001":
            keypoint_3d[0] = head_world

    return keypoint_3d            
            
# ====================================================================
# npzファイルを読み込む
# ====================================================================
def load_keypoints_data_from_npz(filepath, scale_factor=1):
    """NPZファイルからキーポイントデータを読み込み、スケール調整し、Vectorのリストに変換する"""
    try:
        data = np.load(filepath)
        keypoints_data = None
        
        # どのキー名でデータが保存されているかを確認
        if 'keypoints_4d' in data:
            # 4次元データ (N, 17, 4) の場合
            keypoints_data = data['keypoints_4d']
        elif 'keypoints_3d' in data:
            # 3次元データ (N, 17, 3) の場合
            keypoints_data = data['keypoints_3d']
        else:
            print(f"エラー: NPZファイルに 'keypoints_4d' または 'keypoints_3d' キーが見つかりません。")
            return None

        # 複数のフレームが含まれている場合、最初のフレーム ([0]) を取得
        if keypoints_data.ndim == 3:
            keypoints_data = keypoints_data[0] 
            print(keypoints_data)
        
        # 【機能追加】: 4列目（信頼度スコア）を削除する処理
        if keypoints_data.shape[-1] == 4:
            print("💡 データに4列目（信頼度）が含まれています。3次元座標のみに整形します。")
            # 最後の軸の最初の3列 (X, Y, Z) のみを選択
            keypoints_data = keypoints_data[:, :3] 

        # 最終的な形状の検証
        if keypoints_data.shape != (17, 3):
            print(f"エラー: 処理後のキーポイント配列の形状が (17, 3) ではありません: {keypoints_data.shape}")
            return None
            
        # 座標のスケール調整
        keypoints_array_scaled = keypoints_data
            
        # NumPy配列を mathutils.Vector のリストに変換
        keypoints_list = [Vector(kp) for kp in keypoints_array_scaled]
        
        print(f"✅ キーポイントをスケールファクタ {scale_factor} でロードしました。")
        return keypoints_list
        
    except Exception as e:
        print(f"❌ エラー: NPZファイルの読み込みに失敗しました: {e}")
        return None

# ====================================================================
# 1. ポーズ適用メイン関数
# ====================================================================
def apply_pose_fk_method(armature, keypoints_list):
    # 各キーポイントでの回転を計算
    rotation_list = calculate_rotation_from_npz(keypoints_list)
    
    """FK（フォワードキネマティクス）ベースでポーズを適用する (階層順で処理)"""
    
    view_layer = bpy.context.view_layer
    
    bpy.ops.object.mode_set(mode='OBJECT')
    view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE') 
    
    #ポーズボーンの初期化
    for pbone in armature.pose.bones:
        pbone.rotation_mode = 'QUATERNION'
        pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pbone.location = (0.0, 0.0, 0.0)

    # ボーン階層をソート (get_bone_hierarchy関数は省略 - 以前のコードにあるものを使用)
    def get_bone_hierarchy(bone, hierarchy_list):
        hierarchy_list.append(bone.name) 
        for child in bone.children: get_bone_hierarchy(child, hierarchy_list)
        return hierarchy_list
    
    sorted_bones = []
    armature_data = armature.data
    if not armature_data.bones:
        print("エラー: アーマチュアにボーンがありません。")
        bpy.ops.object.mode_set(mode='OBJECT')
        return
    for bone in armature_data.bones:
        if bone.parent is None:
            get_bone_hierarchy(bone, sorted_bones)
    sorted_bones = list(dict.fromkeys(sorted_bones))
    
    sorted_bones.reverse()
            
    # ヒエラルキー順にポーズを適用
    for bone_name in sorted_bones:
        if bone_name not in armature.pose.bones: continue
        
        #print(f"🔄 処理中のボーン: {bone_name}") 
        
        # 処理するボーンの情報を取得
        bone_index = BONE_INDEX_MAP[bone_name]
        pbone = armature.pose.bones[bone_name]
        
        # 回転を取得
        if bone_index in rotation_list:
            target_rotation = rotation_list[bone_index]
            #print(str(bone_index) + " : " + str(bone_name) + "に回転を適用")

            # 回転を適用
            pbone.rotation_quaternion = target_rotation
        

    bpy.ops.object.mode_set(mode='OBJECT') 
    print("✅ FKベースのポーズ適用が完了しました。")
    

# ====================================================================
# 1. ボーンの向きを同じ方向にそろえる操作を適用
# ====================================================================
def apply_clear_pose_fk_method(armature):
    # 各キーポイントでの回転を計算
    rotation_list = calculate_clear_rotation()
    
    """FK（フォワードキネマティクス）ベースでポーズを適用する (階層順で処理)"""
    
    view_layer = bpy.context.view_layer
    
    bpy.ops.object.mode_set(mode='OBJECT')
    view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE') 

    # ボーン階層をソート (get_bone_hierarchy関数は省略 - 以前のコードにあるものを使用)
    def get_bone_hierarchy(bone, hierarchy_list):
        hierarchy_list.append(bone.name) 
        for child in bone.children: get_bone_hierarchy(child, hierarchy_list)
        return hierarchy_list
    
    sorted_bones = []
    armature_data = armature.data
    if not armature_data.bones:
        #print("エラー: アーマチュアにボーンがありません。")
        bpy.ops.object.mode_set(mode='OBJECT')
        return
    for bone in armature_data.bones:
        if bone.parent is None:
            get_bone_hierarchy(bone, sorted_bones)
    sorted_bones = list(dict.fromkeys(sorted_bones))
    
    sorted_bones.reverse()
            
    # ヒエラルキー順にポーズを適用
    for bone_name in sorted_bones:
        if bone_name not in armature.pose.bones: continue
        
        #print(f"🔄 処理中のボーン: {bone_name}") 
        
        # 処理するボーンの情報を取得
        bone_index = BONE_INDEX_MAP[bone_name]
        pbone = armature.pose.bones[bone_name]
        
        # 回転を取得
        if bone_index in rotation_list:
            pbone.rotation_mode = 'XYZ'
            target_rotation = rotation_list[bone_index]
            #print(str(bone_index) + " : " + str(bone_name) + "に回転を適用")
            #print(str(target_rotation))

            # 回転を適用
            pbone.rotation_euler = target_rotation

    bpy.ops.object.mode_set(mode='OBJECT') 
    print("✅ ポーズのクリアが完了しました")
    
    
# ====================================================================
# ボーンの向きを同じ方向にそろえる
# ====================================================================
def calculate_clear_rotation(armature_name='Armature'):
    #print("🔄 ポーズ計算開始（行列ベース）")
    obj = bpy.data.objects.get(armature_name)
    if not obj:
        print(f"❌ エラー: アーマチュア '{armature_name}' が見つかりません。")
        return {}
        
    clear_rotation_list = {}
    last_pair = None
    
    # 処理対象のペアをループ
    for pair in PAIR_LIST:
        # 最初の要素は親としての参照用（Rootなど）
        if last_pair is None:
            last_pair = pair
            #print(f"📍 Rootボーンインデックス {pair} をスキップ（基準として保持）")
            continue

        if pair in PARENT_LIST:
            try:
                # 自身のボーン情報を取得
                bone_name = BONE_INDEX_MAP_REVERSE.get(pair)
                bone = obj.data.bones.get(bone_name)
                
                # 親のボーン情報を取得
                parent_kp_index = PARENT_LIST[pair]
                parent_bone_name = BONE_INDEX_MAP_REVERSE.get(parent_kp_index)
                parent_bone = obj.data.bones.get(parent_bone_name)
                
                if not bone or not parent_bone:
                    #print(f"⚠️ スキップ: {bone_name} または親が見つかりません。")
                    continue

                # ==========================================================
                # 行列ベースのローカル回転計算
                # ==========================================================
                
                # 1. 親ボーンの Rest Pose 行列 (Armature空間)
                # matrix_local にはボーンの向き・ボーンロールが全て含まれています
                m_parent = parent_bone.matrix_local
                
                # 2. 子ボーンの Rest Pose 行列 (Armature空間)
                m_child = bone.matrix_local
                
                # 3. 親の向きを基準とした「理想的な方向」を定義
                # ここでは「親と同じ向きに向かせる」ための計算を行います
                # 親の行列をそのままターゲットとします
                m_target = m_parent
                
                # 4. 「現在の自分の姿勢」から「目標の姿勢」への差分行列を求める
                # 式: Local_Diff = (Child_Rest_Matrix^-1) @ Target_Matrix
                # これにより、ボーンロールの差異を含んだ「打ち消し回転」が算出されます
                m_diff = m_child.inverted() @ m_target
                
                # 5. クォータニオンに変換し、さらにオイラー角へ
                # ジンバルロックを防ぐため一度クォータニオンを経由します
                rotation_local_quat = m_diff.to_quaternion()
                rotation_euler = rotation_local_quat.to_euler('XYZ')
                
                #print(f"✅ {bone_name} (KP {pair}): 計算完了")
                # print(f"   Euler: {rotation_euler}")

                # 回転リストに格納
                clear_rotation_list[pair] = rotation_euler
                
            except Exception as e:
                print(f"❌ エラー (KP {pair}): {e}")
                
    return clear_rotation_list

# ====================================================================
# npzファイルから各関節の回転を求める
# ====================================================================

def calculate_rotation_from_npz (keypoints_list):
    last_pair = None
    rotation_list = {}
    
    for pair in PAIR_LIST:
        #print("prosessing : " + str(pair))
        # pair : ボーンのtailの座標
        # root boneを設定する　or　親ボーンと子ボーンの回転を求める
        if not last_pair is None:
            if pair in PARENT_LIST:
                # 親ボーンの方向ベクトルを求める
                #print("親ボーン : " + str(PARENT_LIST[pair]) + ", " + str(PAIR_LIST[PARENT_LIST[pair]]))
                parent_vec = keypoints_list[PARENT_LIST[pair]] - keypoints_list[PAIR_LIST[PARENT_LIST[pair]]]
                
                # 子ボーンの方向ベクトルを計算
                #print("子ボーン : " + str(pair) + ", " + str(PAIR_LIST[pair]))
                target_vec = keypoints_list[pair] - keypoints_list[PAIR_LIST[pair]]
               
            
                # 親ボーンから子ボーンへの回転を計算
                rotation_result = parent_vec.normalized().rotation_difference(target_vec.normalized())
                # 連想配列に格納
                rotation_list[pair] = rotation_result
        else :
            last_pair = pair 
            #print(str(pair) + "," + str(PAIR_LIST[pair]) + "ついて計算")

    return rotation_list

# ====================================================================
# 2. メイン実行関数
# ====================================================================
# (run_pose_application関数は省略せず記述)
DATA_KEY_NAME = 'keypoints_3d' # グローバル変数として再定義

def run_pose_application(npz_filepath):
    if bpy.context.view_layer.objects.active:
        bpy.context.view_layer.objects.active.select_set(False)
    
    # --- 【ユーザー設定エリア】 ---
    # NPZ_FILEPATH = "C:/Users/sinki/R6_blender/keypoints.npz" 
    # ARMATURE_NAME = "Armature" 
    
    try:
        armature = bpy.data.objects[ARMATURE_NAME]
    except KeyError:
        print(f"❌ エラー: アーマチュア '{ARMATURE_NAME}' がシーンに見つかりません。")
        return
        
    keypoints_list = load_keypoints_data_from_npz(npz_filepath, DATA_KEY_NAME)
    
    #print(str(keypoints_list[0]))
    
    if keypoints_list is None:
        print("処理を中断します。")
        return
        
    apply_pose_fk_method(armature, keypoints_list)

# ====================================================================
# 複数のnpzファイルを読み込む
# ====================================================================
def read_npz_files():
    files = sorted(glob.glob(os.path.join(TAGET_DIR, f'*.{EXTENSION}')))
    image_number = 1

    # 4. ループで関数に渡す
    if not files:
        print("ファイルが見つかりませんでした。")
    else:
        for f in files:
            print(str(f))
            generate_anotation_from_frame(f, image_number)
            image_number = image_number + 1

    print("すべての処理が完了しました。")

# ====================================================================
# poseをリセットする、poseをつける、レンダリング、アノテーションデータの作成
# ====================================================================
def generate_anotation_from_frame(npz_filepath, image_number):
    print("========================================================================")
    print("=============================poseのリセット=============================")
    print("=======================================================================")
    apply_clear_pose_fk_method(bpy.data.objects["Armature"])
    
    print("========================================================================")
    print("=============================poseをつける=============================")
    print("========================================================================")
    run_pose_application(npz_filepath)
    
    print("========================================================================")
    print("=============================レンダリング、アノテーションデータの作成=============================")
    print("========================================================================")
    render_from_multiple_cameras(ARMATURE_NAME, image_number)
    
# ====================================================================
# 連想配列を2次元配列に変換
# ====================================================================
def arrange_keypoint(keypoint_data, output_filepath, numpy_key):
    # 2. 辞書の要素を順番に取り出して2次元配列にする
    # キーを整数(int)として評価して昇順に並べ替えます
    sorted_keys = sorted(keypoint_data.keys(), key=lambda x: int(x))

    # リスト内包表記で順番に値(value)を抽出
    two_d_list = [keypoint_data[k] for k in sorted_keys]

    # 3. NumPy配列に変換（形状: 行数 x 要素数）
    two_d_array = np.array(two_d_list)

    generate_npz_file(output_filepath, two_d_array, numpy_key)

    print("作成されたn次元配列:")
    print(two_d_array)
    print("\n形状 (行数, 列数):", two_d_array.shape)

# ====================================================================
# numpyファイルに書き出す
# ====================================================================
# def generate_npz_file(output_filepath, keypoint, key):
#     if os.path.exists(output_filepath):
#         # 1. 既存データの読み込み
#         with np.load(output_filepath, allow_pickle=True) as data:
#             # 既存の配列を取り出す
#             combined_data = data[key]
        
#         # 2. 垂直方向に結合 (既存のNフレーム + 新しい1フレーム)
#         combined_data = np.concatenate([combined_data, keypoint], axis=0)
#     else:
#         # 新規作成
#         combined_data = keypoint
#         print(f"新規ファイルとして作成を開始します: {output_filepath}")

#     np.savez_compressed(output_filepath, **{key: combined_data})
def generate_npz_file(output_filepath, keypoint, key):
    # keypoint の形状が (17, 4) の場合、(1, 17, 4) に変換して「1フレーム分」として扱う
    if keypoint.ndim == 2:
        keypoint = keypoint[np.newaxis, ...]

    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        # 1. 既存データの読み込み
        with np.load(output_filepath, allow_pickle=True) as data:
            combined_data = data[key]
        
        # 2. 0番目の軸（フレーム軸）方向に結合
        # これにより (N, 17, 4) + (1, 17, 4) = (N+1, 17, 4) になる
        combined_data = np.concatenate([combined_data, keypoint], axis=0)
    else:
        # 新規作成時は (1, 17, 4) の状態で保存
        combined_data = keypoint
        print(f"新規ファイルとして作成を開始します: {output_filepath}")

    np.savez_compressed(output_filepath, **{key: combined_data})

# 実行
if __name__ == "__main__":
    read_npz_files()
    # print("========================================================================")
    # print("=============================poseのリセット=============================")
    # print("=======================================================================")
    # apply_clear_pose_fk_method(bpy.data.objects["Armature"])
    
    # print("========================================================================")
    # print("=============================poseをつける=============================")
    # print("========================================================================")
    # run_pose_application()
    
    # print("========================================================================")
    # print("=============================レンダリング、アノテーションデータの作成=============================")
    # print("========================================================================")
    # render_from_multiple_cameras(ARMATURE_NAME)