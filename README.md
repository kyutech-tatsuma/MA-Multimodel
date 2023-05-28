# ma_multimodel
mp4データから音声と動画をそれぞれ抽出し、それらから得られた特徴量を学習させ回帰モデルを作るプログラムです。
## データの形式
このプログラムでは、以下のような形式のcsvファイルで作られたアノテーションデータが想定されています。

| video_path | target |
| ---------- | ------ |
| video1.mp4 |   0.4  |
| video2.mp4 |   1.4  |

## モデルの学習方法
### motion captureによる処理
まず、https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=0　
にアクセスし、body_pose_model.pthとhand_pose_model.pthをダウンロードします。ダウンロード完了後、detect_pose/model/にpthファイルを移動させます。

その後、detect_poseディレクトリに移動し、以下のコマンドを実行します。このコマンドを実行することで指定したディレクトリー内の全ての動画に対してモーションキャプチャー(openpose)で処理を行います。
```
pip install -r requirements.txt
python detect_pose.py --video_path <動画データが入っているディレクトリのパス>
```
### データの学習
モーションキャプチャーによる処理完了後、プロジェクト直下のディレクトリに移動し、以下のコマンドを実行します。
```
python train.py --data <csvデータのパス> --epochs <学習回数>
```
