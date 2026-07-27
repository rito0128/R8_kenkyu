import csv
import json
import os

# --- 設定 ---
CSV_FILE_PATH = "C:/Users/sinki/R8_kenkyu/generated_demo_img/test_2d_annotation.csv"
JSON_OUTPUT_PATH = "C:/Users/sinki/R8_kenkyu/generated_demo_img/coco_annotations.json"

# 画像の解像度（これまでのコード設定に合わせる）
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1000

# キーポイント数 (J0 〜 J18 の計19関節)
NUM_KEYPOINTS = 19


def convert_csv_to_coco_json(csv_path, json_path):
    if not os.path.exists(csv_path):
        print(f"❌ エラー: CSVファイルが見つかりません: {csv_path}")
        return

    images = []
    annotations = []

    # カテゴリ情報（単一人物カテゴリ）
    categories = [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": [f"J{i}" for i in range(NUM_KEYPOINTS)],
            "skeleton": [],  # 必要に応じて骨格の接続定義を追記可能
        }
    ]

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader):
            image_id = idx + 1
            annotation_id = idx + 1

            # ファイル名に拡張子がない場合は .png を補完
            filename = row["filename"]
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filename += ".png"

            # 1. images 構造の作成
            image_info = {
                "id": image_id,
                "file_name": filename,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
            }
            images.append(image_info)

            # 2. Keypoints データの抽出・成形 ([x1, y1, v1, x2, y2, v2, ...])
            keypoints = []
            num_visible_keypoints = 0

            for j in range(NUM_KEYPOINTS):
                x_str = row.get(f"J{j}_x", "")
                y_str = row.get(f"J{j}_y", "")
                v_str = row.get(f"J{j}_visibility", "")

                if x_str != "" and y_str != "" and v_str != "":
                    x = float(x_str)
                    y = float(y_str)
                    v = int(float(v_str))  # 信頼度 (0, 1, 2)

                    keypoints.extend([x, y, v])
                    if v > 0:
                        num_visible_keypoints += 1
                else:
                    keypoints.extend([0.0, 0.0, 0])

            # 3. BBOX データの算出 ([xmin, ymin, xmax, ymax] -> [x, y, w, h])
            try:
                xmin = float(row["bbox_xmin"])
                ymin = float(row["bbox_ymin"])
                xmax = float(row["bbox_xmax"])
                ymax = float(row["bbox_ymax"])

                bbox_w = max(0.0, xmax - xmin)
                bbox_h = max(0.0, ymax - ymin)
                bbox_coco = [xmin, ymin, bbox_w, bbox_h]
                area = bbox_w * bbox_h
            except (ValueError, KeyError):
                # BBOXが存在しない/空の場合のフォールバック
                bbox_coco = [0.0, 0.0, 0.0, 0.0]
                area = 0.0

            # 4. annotations 構造の作成
            ann_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": num_visible_keypoints,
                "bbox": bbox_coco,
                "area": area,
                "iscrowd": 0,
            }
            annotations.append(ann_info)

    # COCO全体のJSON辞書構造
    coco_data = {
        "info": {
            "description": "Blender Pose Dataset",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # JSONファイルへ書き出し
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)

    print(f"🎉 変換完了! COCO形式JSONを保存しました:\n{json_path}")


if __name__ == "__main__":
    convert_csv_to_coco_json(CSV_FILE_PATH, JSON_OUTPUT_PATH)