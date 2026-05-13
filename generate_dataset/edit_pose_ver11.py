import bpy
import math
import os
import numpy as np
from mathutils import Vector, Quaternion, Matrix

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

NPZ_FILEPATH = "C:/Users/a24k0/R6_blender2/scripts/keypoints.npz" 
ARMATURE_NAME = "Armature"
            
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
# def apply_pose_fk_method(armature, keypoints_list):
#     # 各キーポイントでの回転を計算
#     rotation_list = calculate_rotation_from_npz(keypoints_list)
    
#     #rotaton_listがクオータニオンになっているからダ
    
    
#     """FK（フォワードキネマティクス）ベースでポーズを適用する (階層順で処理)"""
    
#     view_layer = bpy.context.view_layer
    
#     bpy.ops.object.mode_set(mode='OBJECT')
#     view_layer.objects.active = armature
#     bpy.ops.object.mode_set(mode='POSE') 

#     # ボーン階層をソート (get_bone_hierarchy関数は省略 - 以前のコードにあるものを使用)
#     def get_bone_hierarchy(bone, hierarchy_list):
#         hierarchy_list.append(bone.name) 
#         for child in bone.children: get_bone_hierarchy(child, hierarchy_list)
#         return hierarchy_list
    
#     sorted_bones = []
#     armature_data = armature.data
#     if not armature_data.bones:
#         print("エラー: アーマチュアにボーンがありません。")
#         bpy.ops.object.mode_set(mode='OBJECT')
#         return
#     for bone in armature_data.bones:
#         if bone.parent is None:
#             get_bone_hierarchy(bone, sorted_bones)
#     sorted_bones = list(dict.fromkeys(sorted_bones))
    
#     #sorted_bones.reverse()
#     print("sorted_bone")
#     print(str(sorted_bones))
            
#     # ヒエラルキー順にポーズを適用
#     for bone_name in sorted_bones:
#         if bone_name not in armature.pose.bones: continue
        
#         print(f"🔄 処理中のボーン: {bone_name}") 
        
#         # 処理するボーンの情報を取得
#         bone_index = BONE_INDEX_MAP[bone_name]
#         pbone = armature.pose.bones[bone_name]
        
#         # 回転を取得
#         if bone_index in rotation_list:
#             pbone.rotation_mode = 'XYZ'
#             target_rotation = rotation_list[bone_index]
#             #print(str(bone_index) + " : " + str(bone_name) + "に回転を適用")
#             #print(str(target_rotation))
            
#             #オイラー角に変換
#             target_rotation_euler = target_rotation.to_euler('XYZ')

#             # 回転を適用
#             pbone.rotation_euler = target_rotation_euler

#     bpy.ops.object.mode_set(mode='OBJECT') 
#     print("✅ ポーズを適用")
    
def apply_pose_fk_method(armature, keypoints_list):
   # 各キーポイントでの回転を計算
   rotation_list = calculate_rotation_from_npz(keypoints_list)
   
   """FK（フォワードキネマティクス）ベースでポーズを適用する (階層順で処理)"""
   
   view_layer = bpy.context.view_layer
   
   bpy.ops.object.mode_set(mode='OBJECT')
   view_layer.objects.active = armature
   bpy.ops.object.mode_set(mode='POSE') 
   
   #ポーズボーンの初期化
#    for pbone in armature.pose.bones:
#        pbone.rotation_mode = 'QUATERNION'
#        pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
#        pbone.location = (0.0, 0.0, 0.0)

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
   
   #sorted_bones.reverse()
           
   # ヒエラルキー順にポーズを適用
   for bone_name in sorted_bones:
       if bone_name not in armature.pose.bones: continue
       
       print(f"🔄 処理中のボーン: {bone_name}") 
       
       # 処理するボーンの情報を取得
       bone_index = BONE_INDEX_MAP[bone_name]
       pbone = armature.pose.bones[bone_name]
       
       # 回転を取得
       if bone_index in rotation_list:
           target_rotation = rotation_list[bone_index]
           print(str(bone_index) + " : " + str(bone_name) + "に回転を適用")

           # 回転を適用
           pbone.rotation_quaternion = target_rotation
       

#    bpy.ops.object.mode_set(mode='OBJECT') 
#    print("✅ FKベースのポーズ適用が完了しました。")
    

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
        
        print(f"🔄 処理中のボーン: {bone_name}") 
        
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
# ボーンの向きを同じ方向にそろえる回転を計算
# ====================================================================
def calculate_clear_rotation(armature_name='Armature'):
    print("🔄 ポーズ計算開始（行列ベース）")
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
                    print(f"⚠️ スキップ: {bone_name} または親が見つかりません。")
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
        print("prosessing : " + str(pair))
        # pair : ボーンのtailの座標
        # root boneを設定する　or　親ボーンと子ボーンの回転を求める
        if not last_pair is None:
            if pair in PARENT_LIST:
                # 親ボーンの方向ベクトルを求める
                print("親ボーン : " + str(PARENT_LIST[pair]) + ", " + str(PAIR_LIST[PARENT_LIST[pair]]))
                parent_vec = keypoints_list[PARENT_LIST[pair]] - keypoints_list[PAIR_LIST[PARENT_LIST[pair]]]
                
                # 子ボーンの方向ベクトルを計算
                print("子ボーン : " + str(pair) + ", " + str(PAIR_LIST[pair]))
                target_vec = keypoints_list[pair] - keypoints_list[PAIR_LIST[pair]]
               
            
                # 親ボーンから子ボーンへの回転を計算
                rotation_result = parent_vec.normalized().rotation_difference(target_vec.normalized())
                # 連想配列に格納
                rotation_list[pair] = rotation_result
        else :
            last_pair = pair 
            print(str(pair) + "," + str(PAIR_LIST[pair]) + "ついて計算")

    return rotation_list

# ====================================================================
# 2. メイン実行関数
# ====================================================================
# (run_pose_application関数は省略せず記述)
DATA_KEY_NAME = 'keypoints_3d' # グローバル変数として再定義

def run_pose_application():
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
        
    keypoints_list = load_keypoints_data_from_npz(NPZ_FILEPATH, DATA_KEY_NAME)
    
    #print(str(keypoints_list[0]))
    
    if keypoints_list is None:
        print("処理を中断します。")
        return
        
    apply_pose_fk_method(armature, keypoints_list)
    

# 実行
if __name__ == "__main__":
    print("========================================================================")
    print("=============================poseのリセット=============================")
    print("=======================================================================")
    apply_clear_pose_fk_method(bpy.data.objects["Armature"])
    
    print("========================================================================")
    print("=============================poseをつける=============================")
    print("========================================================================")
    run_pose_application()
    
#    print("========================================================================")
#    print("=============================レンダリング=============================")
#    print("========================================================================")
#    render_from_multiple_cameras()
#    
#    print("========================================================================")
#    print("=============================アノテーションデータの作成=============================")
#    print("========================================================================")
#    render_from_multiple_cameras()