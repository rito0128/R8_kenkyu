import bpy
from mathutils import Vector

def apply_rotation_to_child(armature_name="Armature", points=None):
    # 1. オブジェクトの取得
    obj = bpy.data.objects.get(armature_name)
    if not obj or obj.type != 'ARMATURE':
        print(f"Error: {armature_name} が見つかりません。")
        return

    # 2. ポーズモードへ切り替え
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='POSE')
    
    pose_bones = obj.pose.bones
    if len(pose_bones) < 2:
        print("Error: ボーンが2つ以上必要です。")
        return

    # 3. ベクトル計算 (mathutils.Vectorを使用)
    # リストからVectorオブジェクトを生成して計算
    p = [Vector(pt) for pt in points]
    vec1 = p[1] - p[0]
    vec2 = p[2] - p[1]

    # vec2からvec1への回転（クォータニオン）を計算
    quat = vec1.rotation_difference(vec2)

    # 4. 子ボーン（2番目のボーン）に適用
    child_bone = pose_bones[1]
    
    # 回転モードをクォータニオンに設定
    child_bone.rotation_mode = 'QUATERNION'
    
    # 計算した回転を代入
    child_bone.rotation_quaternion = quat

    print(f"Applied Quaternion to {child_bone.name}: {quat}")

# 実行データ
# data_list = [
#     [0, 0, 0],
#     [0, 0, 1],
#     [1, 0, 1]
# ]

#座標系[x, z, y]
data_list = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 0]
]

apply_rotation_to_child("Armature", data_list)