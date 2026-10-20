"""
Blender内で実行されるスクリプト（apply_motion.py から呼ばれる）。

使い方（単体実行する場合）:
    blender -b <コピー済み.blend> -P apply_motion_blender.py -- <motion.fbx> [ターゲットArmature名]

処理内容:
    1. Mixamo FBX をインポート
    2. FBX側Armatureの各ボーンのワールド回転を、ターゲットArmatureへリターゲットしてキーフレーム化
       （ボーン軸・レストポーズの違いを補正する。Hipsのみ位置も身長比でスケールして転写）
    3. インポートしたオブジェクトを削除して .blend を上書き保存
"""
import sys
import os

import bpy
from mathutils import Matrix


# 複数の子を持つボーンで「ボーンの向き」を決める子ボーン
PREFERRED_CHILD = {
    "Hips": "Spine",
    "Spine2": "Neck",
    "LeftHand": "LeftHandMiddle1",
    "RightHand": "RightHandMiddle1",
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv:
        raise SystemExit("❌ FBXファイルのパスを '--' の後に指定してください")
    fbx_path = argv[0]
    target_name = argv[1] if len(argv) > 1 else None
    return fbx_path, target_name


def find_target_armature(target_name):
    if target_name:
        obj = bpy.data.objects.get(target_name)
        if obj is None or obj.type != 'ARMATURE':
            raise SystemExit(f"❌ Armature '{target_name}' が見つかりません")
        return obj
    arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    if len(arms) != 1:
        raise SystemExit(f"❌ Armatureが {len(arms)} 個あります。ターゲット名を指定してください: {[a.name for a in arms]}")
    return arms[0]


def import_fbx(fbx_path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    new_objs = [o for o in bpy.data.objects if o not in before]
    src_arms = [o for o in new_objs if o.type == 'ARMATURE']
    if not src_arms:
        raise SystemExit("❌ FBX内にArmatureがありません")
    src = src_arms[0]
    if not (src.animation_data and src.animation_data.action):
        raise SystemExit("❌ FBXにアニメーションが含まれていません")
    return src, new_objs


def short_name(bone_name):
    return bone_name.split(":")[-1]


def bone_depth(bone):
    d = 0
    while bone.parent:
        bone = bone.parent
        d += 1
    return d


def direction_child(bone, src_bones):
    """ボーンの向きの基準とする子ボーン（ソース側にも存在するもの）を返す"""
    children = [c for c in bone.children if c.name in src_bones]
    if not children:
        return None
    pref = PREFERRED_CHILD.get(short_name(bone.name))
    for c in children:
        if pref and short_name(c.name) == pref:
            return c
    return children[0]


def world_head(arm_obj, bone):
    return arm_obj.matrix_world @ bone.head_local


def rot3(m):
    return m.to_3x3().normalized()


def find_bone(bones, short):
    for b in bones:
        if short_name(b.name) == short:
            return b
    return None


def body_axes(arm_obj):
    """レスト時の体の基準軸（左右・上下・前後）をワールド空間で返す"""
    bones = arm_obj.data.bones
    l, r = find_bone(bones, "LeftUpLeg"), find_bone(bones, "RightUpLeg")
    hips, neck = find_bone(bones, "Hips"), find_bone(bones, "Neck")
    if None in (l, r, hips, neck):
        return None
    up = (world_head(arm_obj, neck) - world_head(arm_obj, hips)).normalized()
    side = world_head(arm_obj, l) - world_head(arm_obj, r)
    side = (side - side.dot(up) * up).normalized()
    return [side, up, up.cross(side)]


def frame_from(d, ref):
    """主軸 d と参照ベクトル ref から正規直交基底（列ベクトル）を作る"""
    x = (ref - ref.dot(d) * d).normalized()
    m = Matrix((x, d, x.cross(d))).transposed()
    return m


def build_offsets(src, tgt, bones):
    """
    各ボーンについて定数補正行列 C を計算する。
        ターゲットのワールド回転(t) = ソースのワールド回転(t) @ C
    C = S0^-1 @ A^-1 @ T0
        S0: ソースのレスト時ワールド回転, T0: ターゲットのレスト時ワールド回転
        A : ソースのレスト時ボーン方向 → ターゲットのレスト時ボーン方向 への回転
            （Tポーズ/Aポーズなどレストポーズの違いを吸収する）
            ボーン方向に加えて体の基準軸も合わせ、ボーン軸周りのねじれ（向きの違い）も補正する
    """
    src_bones = src.data.bones
    src_axes, tgt_axes = body_axes(src), body_axes(tgt)
    align = {}
    offsets = {}
    for b in bones:
        child = direction_child(b, src_bones)
        if child is not None:
            s_b, s_c = src_bones[b.name], src_bones[child.name]
            d_src = (world_head(src, s_c) - world_head(src, s_b)).normalized()
            d_tgt = (world_head(tgt, child) - world_head(tgt, b)).normalized()
            if src_axes and tgt_axes:
                # ボーン方向と最も直交に近い体の基準軸を参照に使う
                i = min(range(3), key=lambda k: abs(d_tgt.dot(tgt_axes[k])))
                A = frame_from(d_tgt, tgt_axes[i]) @ frame_from(d_src, src_axes[i]).transposed()
            else:
                A = d_src.rotation_difference(d_tgt).to_matrix()
        elif b.parent is not None and b.parent.name in align:
            A = align[b.parent.name]  # 末端ボーンは親の補正を流用
        else:
            A = Matrix.Identity(3)
        align[b.name] = A

        S0 = rot3(src.matrix_world @ src_bones[b.name].matrix_local)
        T0 = rot3(tgt.matrix_world @ b.matrix_local)
        offsets[b.name] = S0.inverted() @ A.inverted() @ T0
    return offsets


def retarget(src, tgt, src_action):
    scene = bpy.context.scene
    src_bones = src.data.bones
    bones = sorted([b for b in tgt.data.bones if b.name in src_bones], key=bone_depth)
    skipped = [b.name for b in tgt.data.bones if b.name not in src_bones]
    if skipped:
        print(f"⚠ FBXに存在しないため無視するボーン: {skipped}")

    root = bones[0]
    offsets = build_offsets(src, tgt, bones)

    # Hips移動量のスケール（レスト時の腰の高さの比）
    src_root_h = world_head(src, src_bones[root.name]).z
    tgt_root_h = world_head(tgt, root).z
    loc_scale = tgt_root_h / src_root_h if abs(src_root_h) > 1e-6 else 1.0

    # ターゲットに新しいActionを割り当てる
    motion_name = os.path.splitext(os.path.basename(FBX_PATH))[0]
    action = bpy.data.actions.new(name=motion_name)
    action.use_fake_user = True
    if tgt.animation_data is None:
        tgt.animation_data_create()
    tgt.animation_data.action = action

    for pb in tgt.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.matrix_basis = Matrix.Identity(4)

    f_start, f_end = (int(round(v)) for v in src_action.frame_range)
    tgt_mw_inv = tgt.matrix_world.inverted()
    tgt_mw_rot_inv = rot3(tgt.matrix_world).inverted()
    prev_quat = {}

    for f in range(f_start, f_end + 1):
        scene.frame_set(f)
        src_eval = src.evaluated_get(bpy.context.evaluated_depsgraph_get())
        pose_mats = {}  # ターゲットのアーマチュア空間ポーズ行列
        for b in bones:
            spb = src_eval.pose.bones[b.name]
            S_t = rot3(src.matrix_world @ spb.matrix)
            R = (tgt_mw_rot_inv @ S_t @ offsets[b.name]).to_4x4()

            if b.parent is None or b.parent.name not in pose_mats:
                # ルート（Hips）: 位置も転写
                src_pos = (src.matrix_world @ spb.matrix).translation
                pos = tgt_mw_inv @ (src_pos * loc_scale)
                pose = Matrix.Translation(pos) @ R
                parent_part = b.matrix_local
            else:
                rel_rest = b.parent.matrix_local.inverted() @ b.matrix_local
                parent_part = pose_mats[b.parent.name] @ rel_rest
                pose = Matrix.Translation(parent_part.translation) @ R
            pose_mats[b.name] = pose

            basis = parent_part.inverted() @ pose
            tpb = tgt.pose.bones[b.name]
            q = basis.to_quaternion()
            if b.name in prev_quat and prev_quat[b.name].dot(q) < 0:
                q.negate()  # クォータニオンの符号反転を防ぐ
            prev_quat[b.name] = q
            tpb.rotation_quaternion = q
            tpb.keyframe_insert("rotation_quaternion", frame=f, group=b.name)
            if b is root:
                tpb.location = basis.translation
                tpb.keyframe_insert("location", frame=f, group=b.name)

    scene.frame_start = f_start
    scene.frame_end = f_end
    return action, (f_start, f_end)


def cleanup(new_objs, src_action):
    for o in new_objs:
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.actions.remove(src_action)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


FBX_PATH, TARGET_NAME = parse_args()


def main():
    tgt = find_target_armature(TARGET_NAME)
    src, new_objs = import_fbx(FBX_PATH)
    src_action = src.animation_data.action
    action, (fs, fe) = retarget(src, tgt, src_action)
    cleanup(new_objs, src_action)
    bpy.context.preferences.filepaths.save_version = 0  # .blend1 バックアップを作らない
    bpy.ops.wm.save_mainfile()
    print(f"✅ '{action.name}' を '{tgt.name}' に適用しました (frame {fs}-{fe}, {bpy.context.scene.render.fps}fps)")


main()
