##edit_pose_ver9
-clear_poseまでデバック済み
-npzからのポーズ計算は修正しない

##edit_pose_ver10
-edit_pose_ver9に画像のレンダリング機能、boneの座標を取得する機能を実装
-

##edit_pose_ver11
-edit_pose_ver9のポーズ計算を修正中

##edit_pose_ver12
-ver10から
-keypoint2d(画像平面の座標系)を書き出す機能を実装
-keypoint3d(rootboneのheadを原点とした座標系)を書き出す機能を実装
-17点のキーポイントの座標として保存する機能を実装

##edit_pose_ver13
-ver10から
-keypoint2d(画像平面の座標系)を書き出す機能を実装
-keypoint3d(rootboneのheadを原点とした座標系)を書き出す機能を実装
-17点のキーポイントの座標として保存する機能を実装
-複数ファイルに対応
-anotationデータを2dと3dに分けて、npzファイルに書き出す機能を実装（複数カメラには、未対応）

##edit_pose_ver14
-ver10から
-keypoint2d(画像平面の座標系)を書き出す機能を実装
-keypoint3d(rootboneのheadを原点とした座標系)を書き出す機能を実装
-17点のキーポイントの座標として保存する機能を実装
-複数ファイルに対応
-anotationデータを2dと3dに分けて、npzファイルに書き出す機能を実装（複数カメラには、未対応）
-モデルにテクスチャを反映

##edit_pose_ver15
-ver10から
-keypoint2d(画像平面の座標系)を書き出す機能を実装
-keypoint3d(rootboneのheadを原点とした座標系)を書き出す機能を実装
-17点のキーポイントの座標として保存する機能を実装
-複数ファイルに対応
-anotationデータを2dと3dに分けて、npzファイルに書き出す機能を実装（複数カメラには、未対応）
-モデルにテクスチャを反映
-信頼度を算出する機能を実装

##get_keypoint
-カメラごとに画像をレンダリングする機能を実装
-アーマチュアのheadとtailの座標を取得する機能を実装
-keypoint2d(画像平面の座標系)を書き出す機能を実装
-keypoint3d(rootboneのheadを原点とした座標系)を書き出す機能を実装