import bpy
import os
import time

# --- 設定 ---
BASE_OUTPUT_DIR = "C:/Users/sinki/R8_kenkyu/generated_demo_img/"

# 開いている .blend ファイル名（拡張子なし）を取得
blend_filename = bpy.path.basename(bpy.context.blend_data.filepath)
blend_stem = os.path.splitext(blend_filename)[0] if blend_filename else "default_scene"

# blendファイルごとに出力先フォルダを分ける
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, blend_stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAMERA_NAMES = ["Camera1", "Camera2", "Camera3", "Camera4", "Camera5", "Camera6", "Camera7", "Camera8", "Camera9", "Camera10", "Camera11", "Camera12"]
ARMATURE_NAME = "Armature"
BACKGROUND_PLANE_NAME = "Background"

IMAGE_FORMAT = 'PNG'
RESOLUTION_X = 1000
RESOLUTION_Y = 1000

# アバターに属するメッシュへ割り当てる Object Index (Pass Index)。
# Compositor の ID Mask ノードでこの値と一致するピクセルだけを前景(白)として抽出する。
AVATAR_PASS_INDEX = 1

# マスク画像のファイル名サフィックス（例: out_Camera1_0001_mask.png）
MASK_SUFFIX = "_mask"
# File Output ノードが自動でフレーム番号を付与するため、一旦この名前で出力してからリネームする
MASK_TMP_PREFIX = "_mask_tmp_"
# マスクレンダリング中に Composite 経由で書き出される不要なカラー画像の一時ファイル名
MASK_COLOR_DUMMY_NAME = "_mask_render_dummy"

# Compositor の File Output ノードは EEVEE Next では Object Index (IndexOB) パスからの
# 出力に対応していない（レンダリングは成功するがファイルが生成されない）。
# そのためマスク生成時のみ Cycles に一時的に切り替える。ID は幾何情報だけで決まり
# シェーディングのノイズを受けないため、samples=1 かつ denoising 無効で十分きれいに求まる。
MASK_RENDER_ENGINE = 'CYCLES'
MASK_RENDER_SAMPLES = 1
# --------------------

def setup_render_settings(scene):
    """シーンのレンダリング基本設定を行う"""
    scene.render.image_settings.file_format = IMAGE_FORMAT
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y

def get_avatar_center(armature):
    """アバターの回転中心を取得する。"""
    if "mixamorig:Hips" in armature.pose.bones:
        return armature.matrix_world @ armature.pose.bones["mixamorig:Hips"].head
    return armature.matrix_world.to_translation()

def capture_background_reference(reference_camera, plane_name, armature_name):
    """開始時のカメラ、アバター、背景の相対変換を保存する。"""
    plane = bpy.data.objects.get(plane_name)
    armature = bpy.data.objects.get(armature_name)
    if not plane or not armature or not reference_camera:
        return None

    avatar_center = get_avatar_center(armature)
    return {
        "camera_offset": reference_camera.matrix_world.to_translation() - avatar_center,
        "plane_offset": plane.matrix_world.to_translation() - avatar_center,
        "plane_rotation": plane.matrix_world.to_quaternion(),
        "plane_scale": plane.scale.copy(),
    }

def align_plane_with_camera_orbit(camera, plane_name, armature_name, reference):
    """背景をアバター中心に回転させ、開始時の相対関係を維持する。"""
    plane = bpy.data.objects.get(plane_name)
    armature = bpy.data.objects.get(armature_name)
    if not plane or not armature or not camera or not reference:
        return

    avatar_center = get_avatar_center(armature)
    current_camera_offset = camera.matrix_world.to_translation() - avatar_center
    if reference["camera_offset"].length == 0 or current_camera_offset.length == 0:
        return

    orbit_rotation = reference["camera_offset"].rotation_difference(current_camera_offset)
    plane.location = avatar_center + orbit_rotation @ reference["plane_offset"]
    plane.rotation_mode = 'QUATERNION'
    plane.rotation_quaternion = orbit_rotation @ reference["plane_rotation"]
    plane.scale = reference["plane_scale"]

def assign_avatar_pass_index(armature_name, pass_index=AVATAR_PASS_INDEX):
    """アバターに属する子メッシュに Object Index (Pass Index) を設定する"""
    armature_obj = bpy.data.objects.get(armature_name)
    if not armature_obj:
        return
    for child in armature_obj.children:
        if child.type == 'MESH':
            child.pass_index = pass_index

def get_enabled_output_socket(node, socket_name):
    """
    use_pass_* を有効化した直後は、対応する出力ソケットの enabled フラグが
    まだ False のままになっていることがある（名前でのdictアクセスが失敗する）。
    ソケットを名前で探し、enabled を明示的に立ててから返す。
    """
    for socket in node.outputs:
        if socket.name == socket_name:
            socket.enabled = True
            return socket
    raise KeyError(f"output socket '{socket_name}' not found on node {node.name}")

def setup_mask_compositor(scene, output_dir, pass_index=AVATAR_PASS_INDEX):
    """
    Object Index (IndexOB) パスから、アバター領域=白 / それ以外(背景含む)=黒 の
    2値マスクを生成し、File Output ノードで別画像として保存する Compositor を構築する。
    通常のカラー画像は Composite ノード経由で従来どおり render.filepath に保存される。
    """
    view_layer = bpy.context.view_layer
    view_layer.use_pass_object_index = True

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.layer = view_layer.name
    composite = tree.nodes.new('CompositorNodeComposite')
    tree.links.new(render_layers.outputs['Image'], composite.inputs['Image'])

    id_mask = tree.nodes.new('CompositorNodeIDMask')
    id_mask.index = pass_index
    id_mask.use_antialiasing = True
    index_ob_socket = get_enabled_output_socket(render_layers, 'IndexOB')
    tree.links.new(index_ob_socket, id_mask.inputs['ID value'])

    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.base_path = output_dir
    file_output.format.file_format = 'PNG'
    file_output.format.color_mode = 'BW'
    file_output.file_slots[0].path = MASK_TMP_PREFIX
    tree.links.new(id_mask.outputs['Alpha'], file_output.inputs[0])

    return file_output

def finalize_mask_filename(output_dir, frame, image_filename):
    """File Output ノードが自動付与したフレーム番号付きファイル名を、対応するカラー画像名に揃える"""
    tmp_path = os.path.join(output_dir, f"{MASK_TMP_PREFIX}{frame:04d}.png")
    final_path = os.path.join(output_dir, f"{image_filename}{MASK_SUFFIX}.png")
    if os.path.exists(tmp_path):
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(tmp_path, final_path)

def render_mask_only(scene, output_dir, frame, image_filename):
    """
    Cycles に一時的に切り替えて Object Index ベースの2値マスクだけをレンダリングし、
    対応するカラー画像名に揃えてリネームしたあと、元のレンダーエンジンに戻す。
    """
    original_engine = scene.render.engine
    original_samples = scene.cycles.samples
    original_denoising = scene.cycles.use_denoising

    scene.render.engine = MASK_RENDER_ENGINE
    scene.cycles.samples = MASK_RENDER_SAMPLES
    scene.cycles.use_denoising = False
    # Composite 経由で書き出される（マスク生成には不要な）カラー画像は使い捨てファイルに集約する
    scene.render.filepath = os.path.join(output_dir, MASK_COLOR_DUMMY_NAME)

    bpy.ops.render.render(write_still=True)
    finalize_mask_filename(output_dir, frame, image_filename)

    scene.render.engine = original_engine
    scene.cycles.samples = original_samples
    scene.cycles.use_denoising = original_denoising

def process_animation_frames():
    """アーマチュアのアニメーションを1フレームずつ進め、各フレームのレンダリング画像と
    アバター/背景の2値セグメンテーションマスクを保存する"""
    scene = bpy.context.scene
    armature = bpy.data.objects.get(ARMATURE_NAME)

    if not armature or not armature.animation_data or not armature.animation_data.action:
        print(f"エラー: アニメーションが適用されたアーマチュア '{ARMATURE_NAME}' が見つかりません。")
        return

    action = armature.animation_data.action

    fcurves_to_remove = []
    for fc in action.fcurves:
        if 'pose.bones["mixamorig:Hips"].location' in fc.data_path:
            if fc.array_index in [0, 1]:
                fcurves_to_remove.append(fc)
    for fc in fcurves_to_remove:
        action.fcurves.remove(fc)

    start_frame = int(action.frame_range[0])
    end_frame = int(action.frame_range[1])

    setup_render_settings(scene)
    assign_avatar_pass_index(ARMATURE_NAME)
    setup_mask_compositor(scene, OUTPUT_DIR)

    scene.frame_set(start_frame)
    bpy.context.view_layer.update()
    reference_camera = bpy.data.objects.get(CAMERA_NAMES[0])
    background_reference = capture_background_reference(
        reference_camera, BACKGROUND_PLANE_NAME, ARMATURE_NAME
    )
    if not background_reference:
        print(f"警告: 背景 '{BACKGROUND_PLANE_NAME}' または基準カメラが見つかりません。")

    for camera_name in CAMERA_NAMES:
        camera = bpy.data.objects.get(camera_name)
        if not camera or camera.type != 'CAMERA':
            print(f"警告: カメラ '{camera_name}' が見つからないためスキップします。")
            continue

        scene.camera = camera

        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)
            bpy.context.view_layer.update()

            # 開始時の相対関係を保ったまま、背景をカメラの周回に追従させる
            align_plane_with_camera_orbit(
                camera, BACKGROUND_PLANE_NAME, ARMATURE_NAME, background_reference
            )
            bpy.context.view_layer.update()

            image_filename = f"out_{camera_name}_{frame:04d}"
            render_output_path = os.path.join(OUTPUT_DIR, image_filename)

            # 1. カラー画像（元のレンダーエンジンでレンダリング）
            scene.render.filepath = render_output_path
            bpy.ops.render.render(write_still=True)

            # 2. アバター/背景の2値セグメンテーションマスク（Cyclesに一時切替）
            render_mask_only(scene, OUTPUT_DIR, frame, image_filename)

    dummy_color_path = os.path.join(OUTPUT_DIR, f"{MASK_COLOR_DUMMY_NAME}.png")
    if os.path.exists(dummy_color_path):
        os.remove(dummy_color_path)

if __name__ == "__main__":
    start_time = time.time()

    process_animation_frames()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n処理時間: {elapsed_time:.4f} 秒")
