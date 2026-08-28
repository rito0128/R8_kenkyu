# 姿勢推定モデル検証 セッションログ（2026-08-27〜28）

このファイルは、Claude Codeとの作業セッションで行った検証内容・発見事項・成果物をまとめたものです。
別PCでこのブランチをpullした後、新しいClaude Codeセッションにこのファイルを読ませることで、同等の文脈から作業を再開できます。

## 背景・目的

独自に作成したデータセット（Blenderで生成した合成動画+正解キーポイント座標）で追加学習した
2D姿勢推定モデル（HRNetベース、`my_scripts/20260827_best_epoch_HRNet.pth`）と、
その2D結果を3Dに持ち上げるMotionBertモデル（`work_dirs/motionbert_expansion_1_20260827_4000_seq81_bs2_amp/best_MPJPE_epoch_90.pth`）
の推定精度を、テスト用データセット（motion_007、12台のカメラで撮影、101フレーム）で検証している。

キーポイントは19点（0:head, 1:left_shoulder, 2:right_shoulder, 3:left_elbow, 4:right_elbow,
5:left_wrist, 6:right_wrist, 7:left_hip, 8:right_hip, 9:left_knee, 10:right_knee,
11:left_ankle, 12:right_ankle, 13〜18:spine0〜spine5）。

## ディレクトリ構成

```
zikkenn/
├── for_test_HRNet/                    # 2D推定の検証一式（旧: for_test/ からリネーム済み）
│   ├── predictions/                   # HRNet推定結果 (JSON, 12カメラ×101フレーム)
│   ├── image/                         # 入力画像
│   ├── correct_2d_annotation.csv      # 正解2Dキーポイント座標(px)
│   ├── keypoint_common.py             # キーポイント名・ペア定義・距離関数（2D用）
│   ├── verify_keypoint_lengths.py     # predictions/*.json → keypoint_lengths.csv
│   ├── compute_gt_lengths.py          # correct_2d_annotation.csv → correct_keypoint_lengths.csv
│   ├── plot_keypoint_lengths.py       # 上記2つのCSVからグラフ生成(3種類)
│   ├── keypoint_lengths.csv           # 推定結果の関節間距離(px)
│   ├── correct_keypoint_lengths.csv   # 正解の関節間距離(px)
│   ├── graphs/                        # 推定結果のみ (個別 + summary)
│   ├── graphs_gt/                     # 正解のみ (個別 + summary)
│   └── graphs_comparison/             # 推定と正解を重ねたグラフ (個別 + summary)
│
└── for_test_motionbert/
    └── motionbert_20260827_motion_007/   # 3D推定(MotionBert)の検証一式
        ├── predictions/                  # motion_007_Camera{N}.json (frame_idごとにkeypoints[x,y,z])
        ├── input_videos/, visualizations/
        ├── correct_3d_anotation.csv      # 正解3Dキーポイント座標(実寸/メートル相当)+ visibility
        ├── camera1_run.log, cameras2to12_run.log
        ├── keypoint_common.py            # キーポイント名・ペア定義（3D用、2Dとはペア構成が異なる）
        ├── compute_motionbert_lengths.py # predictions/*.json → motionbert_keypoint_lengths.csv
        ├── plot_motionbert_lengths.py    # 上記CSVからグラフ生成
        ├── motionbert_keypoint_lengths.csv
        └── graphs/                       # 個別 + summary
```

2Dと3Dでキーポイントのペア定義(`KEYPOINT_PAIRS`)が異なるため、`keypoint_common.py`は
それぞれのフォルダに別々に用意している（内容は独立、共有していない）。

## 主要な発見事項

### 1. 【重大・要対応】MotionBert推論のバグ疑い

`for_test_motionbert/.../predictions/` 内のJSONを比較したところ、
**Camera2〜Camera12の推定結果がフレーム単位で完全に一致（バイト単位で同一）**していた。
Camera1のみ異なる値。入力動画はカメラごとに別ファイル（サイズも異なる）であることを確認済み。

- ログが `camera1_run.log` と `cameras2to12_run.log` に分かれており、
  **2〜12番をまとめて回すバッチ推論スクリプト側で、前カメラの結果(または中間の2Dキーポイント)を
  使い回してしまうループバグ**が疑われる。
- **対応が必要**: MotionBert側の推論スクリプトのループ処理を確認し、Camera2〜12を再実行すること。
  修正・再実行するまで、Camera2〜12の3D推定結果はカメラ間比較・個別評価に使用できない。
  そのため直近の3D考察は **Camera1のみ** に限定して実施した。

### 2. 2D推定（HRNet）の精度傾向

`correct_2d_annotation.csv` と `keypoint_lengths.csv` を突き合わせた結果:

- **全体の平均位置誤差(MPJPE-2D): 約47.5px**（中央値は関節ごとに25〜36px程度で、平均より大きく低い
  → 一部フレームで大きく外す「裾の重い」誤差分布）
- **カメラ別誤差**: Camera2(93.5px)・Camera3(105.1px)が突出して悪い（他カメラは23〜54px）。
  原因を追うと、Camera2/3のフレーム12〜20付近で`spine3-spine4`長が本来36px程度のところ
  700〜6800pxという明白な誤検出（キーポイントが画面外等に飛ぶ）が発生している。
  この区間だけ推定が破綻している可能性が高い。
- **関節別誤差**: 肘・手首・膝など末端関節(65〜78px)が、体幹(spine, head: 20〜42px)より誤差が大きい。
  可動域が広い部位ほど誤差が乗りやすい一般的な傾向と一致。
- **推定長と正解長のフレーム間相関はほぼゼロ〜弱い負**（例: left_knee-left_ankle 相関0.14など）。
  誤差が定数バイアスではなく、姿勢変化への追従性が低い「ノイズ的」な性質であることを示唆。

### 3. 3D推定（MotionBert, Camera1のみ）の考察

**(a) 姿勢の恒常性（首〜腰: head-spine0-spine1-...-spine5）**

区間長のフレーム間変動係数(CV=std/mean、無次元なので推定/正解で直接比較可能)を算出。

| 区間 | 推定CV | 正解CV |
|---|---|---|
| head-spine0 | 0.041 | 0.042 |
| spine0-spine1 | 0.214 | 0.000 |
| spine1-spine2 | 0.267 | 0.003 |
| spine2-spine3 | 0.251 | 0.000 |
| spine3-spine4 | 0.281 | 0.000 |
| spine4-spine5 | 0.296 | 0.000 |

正解は剛体骨格なのでCVがほぼ0だが、推定はspine系区間で21〜30%も変動しており、
**本来一定であるはずの骨長がフレームごとに大きく揺れている**（時間的制約なしの単眼3Dリフティングに
典型的なジッター）。
注意点: head-spine0だけ推定CVが正解と近いが、これはkp0(head)のx,y座標が全フレームで
厳密に(0,0)固定（MotionBertのroot相対正規化のアーティファクト）であることが原因で、
精度の高さを意味しない。

**(b) 姿勢の柔軟性（関節角度: 肩-肘-手首、股関節-膝-足首の3点角度）**

| 関節 | 推定可動範囲 | 正解可動範囲 | 角度相関 | 角度MAE |
|---|---|---|---|---|
| left_elbow | 71.4°〜131.0° | 86.7°〜127.4° | 0.896 | 8.2° |
| right_elbow | 78.3°〜145.4° | 73.3°〜146.4° | 0.129 | 23.8° |
| left_knee | 11.8°〜149.4° | 71.0°〜155.9° | 0.655 | 23.7° |
| right_knee | 21.8°〜141.7° | 47.5°〜159.5° | 0.899 | 19.5° |

left_elbow・right_kneeは動きのタイミングとの相関が高い(0.90前後)一方、left_kneeは
可動範囲が正解より大きく広がっており、生理的にあり得ない曲がり方をしている疑いがある。
評価軸の候補: ①可動範囲が生理的に妥当な範囲に収まっているか（GT不要）、②角度の相関係数、③角度MAE。

**(c) セルフオクルージョンに対する精度**

`correct_3d_anotation.csv`の`J{n}_visibility`(0/1=遮蔽,2=可視)をCamera1視点の遮蔽ラベルとして使用。

- right_elbow: 遮蔽時MAE=24.7°(n=94) vs 可視時MAE=11.1°(n=7) → 遮蔽で明確に悪化。
- right_knee: 遮蔽時MAE=15.0°(n=24) vs 可視時MAE=20.9°(n=77) → 逆の傾向。
  遮蔽区間(78〜101フレーム)が動作終盤の動きの少ない区間と重なっており、
  **遮蔽の影響と動作量(動きの大小)の影響が交絡している**可能性。
- 骨長の中央値からの相対ブレでも同様の傾向差を確認（right_shoulder-elbow, right_elbow-wristは
  遮蔽時に明確に悪化、right_knee-ankleは逆）。
- MotionBertの`keypoint_scores`は全フレーム・全関節で常に1.0固定で、自己申告の信頼度は使えなかった。
  信頼度ベースで遮蔽を検出したい場合は2D側(HRNet)のスコアを別途保存する必要がある。
- 評価軸の候補: GTのvisibilityで層別した誤差比較、骨長中央値からの相対乖離（GT不要）、
  動作量で揃えた上での遮蔽あり/なし比較（交絡除去）。

## 既知の未対応事項 / TODO

1. **MotionBert推論スクリプトのループバグ修正 → Camera2〜12を再実行**（最優先）
2. 再実行後、2Dと同様にCamera2〜12を含めた3D側のカメラ間比較・GT比較を再実施
3. 3D推定値(正規化・無次元)と正解値(実寸)のスケール差の扱い方針を検討
   （現状はCV・角度など無次元指標で代替。絶対誤差で評価したい場合はスケール推定/キャリブレーションが必要）
4. `for_test_HRNet`は元々`for_test`という名前だったが、作業中にユーザー側でリネームされた経緯あり。
   スクリプトのパス参照は現在のディレクトリ構成に追随済み。今後リネームする場合は
   各スクリプトの`BASE_DIR`基準の相対パスを要確認。

## 追記: 3D推定(Camera1)の考察3点 と 発表スライドへの反映（2026-08-28）

MotionBertのCamera2〜12重複バグを踏まえ、3D推定の考察は**Camera1のみ**を対象に実施した。
`correct_3d_anotation.csv`のvisibilityフラグを自己遮蔽ラベルとして活用している点がポイント。

### (a) 姿勢の恒常性（head-spine0-spine1-…-spine5の6区間）

区間長のフレーム間変動係数(CV=std/mean)を推定・正解で比較（単位に依存しないため直接比較可能）。

| 区間 | 推定CV | 正解CV |
|---|---|---|
| head-spine0 | 0.041 | 0.042 |
| spine0-spine1 | 0.214 | 0.000 |
| spine1-spine2 | 0.267 | 0.003 |
| spine2-spine3 | 0.251 | 0.000 |
| spine3-spine4 | 0.281 | 0.000 |
| spine4-spine5 | 0.296 | 0.000 |

正解(剛体骨格)はCVがほぼ0だが、推定はspine系区間で21〜30%も変動しており、骨長の恒常性に課題。
head-spine0のみCVが低いのは、kp0(head)のx,y座標が全フレーム(0,0)固定されるroot正規化の
アーティファクトであり、精度の高さを意味しない点に注意。

### (b) 姿勢の柔軟性（肩-肘-手首、股関節-膝-足首の3点角度）

| 関節 | 推定可動範囲 | 正解可動範囲 | 角度相関 | 角度MAE |
|---|---|---|---|---|
| left_elbow | 71.4°〜131.0° | 86.7°〜127.4° | 0.896 | 8.2° |
| right_elbow | 78.3°〜145.4° | 73.3°〜146.4° | 0.129 | 23.8° |
| left_knee | 11.8°〜149.4° | 71.0°〜155.9° | 0.655 | 23.7° |
| right_knee | 21.8°〜141.7° | 47.5°〜159.5° | 0.899 | 19.5° |

left_elbow・right_kneeは動きのタイミングとの相関が高い(0.9前後)一方、left_kneeは可動範囲が
正解より大きく広がっており、非現実的な曲がり方をしている疑いがある。

### (c) セルフオクルージョンに対する精度

正解データのvisibilityフラグ(0/1=遮蔽,2=可視)で層別し、遮蔽時/可視時の角度誤差(MAE)を比較。

| 関節 | 遮蔽時MAE | 可視時MAE |
|---|---|---|
| right_elbow | 24.7°(n=94) | 11.1°(n=7) |
| right_knee | 15.0°(n=24) | 20.9°(n=77) |

right_elbowは遮蔽時に明確に悪化（典型例）。right_kneeは逆の傾向で、遮蔽区間(78〜101フレーム)が
動作終盤の低動作量フェーズと重なっており、遮蔽の影響と動作量の影響が交絡している可能性がある。
MotionBertの`keypoint_scores`は全フレーム常に1.0固定で、自己申告の信頼度は使えなかった。

### 発表スライド(`A-14-2_noguchi.pptx`)への反映

作りかけだった発表スライドのスライド7〜11に、上記の考察を反映した。

- **スライド7〜9**（既存タイトルが「恒常性の評価」「柔軟性の評価」「セルフオクルージョンに対する
  精度の評価」で、内容が上記(a)(b)(c)と1:1対応していたため、それぞれに結果表と要点を追加）
- **スライド10（考察）**: 3項目を総合した考察と、Camera1単一視点による評価という限界の明記
- **スライド11（まとめ・今後の課題）**: まとめと、多視点検証・骨長制約導入・遮蔽データ拡充などの
  今後の課題を追加

デザインは最小限（白背景・黒文字・罫線のみの表、装飾なし）。反映には`zikkenn/build_slides.py`
（python-pptxで既存ボックスの下に表・箇条書きテキストボックスを追加するスクリプト）を使用した。
このスクリプトは`A-14-2_noguchi.pptx`を直接上書きするため、再実行する場合は事前にバックアップを
取ること。

**視覚確認の方法**: このPC(Windows, PowerPoint導入済み)ではLibreOffice(`soffice`)が未インストールで、
スキル付属の`scripts/office/soffice.py`もサンドボックス前提(`socket.AF_UNIX`参照)でWindows上では
動作しない。代わりにPowerShellからPowerPoint COM (`New-Object -ComObject PowerPoint.Application`)
を使い、`Slides.Item(n).Export(path, "PNG", w, h)`でスライドを画像化してQAした。
また`scripts/office/validate.py`は既定のロケール(cp932)でXMLを読もうとして失敗するため、
`PYTHONUTF8=1`を付けて実行する必要があった。
