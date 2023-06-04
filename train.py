# 必要なモジュールをインポートする
from torch import optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from model import VideoNet, AudioNet, CombinedNet
from dataset import VideoAudioDataset
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class EarlyStopping:
    """早期終了 (early stopping) を制御するクラス"""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    # callメソッドはインスタンスそのものが関数のように実行されたときに呼び出される関数
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''検証用ロスが改善した時にモデルを保存します'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# モデルの量子化
def quantize_model(model):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')# qnnpack
    torch.quantization.prepare(model, inplace=True)

    # ここでのモデルの評価は、オブザーバーがレイヤーの重みとバイアスを調査して、それぞれの最小値と最大値を見つけるために必要です。
    # 通常、このステップではモデルに対して一部のサンプルデータを入力します。

    # Calibrate with a few batches of data
    for data in train_loader: 
        video_data, audio_data, targets = data
        video_data = video_data.float()
        audio_data = audio_data.float()
        model(video_data, audio_data)

    torch.quantization.convert(model, inplace=True)

def train_multimodel(opt):
    # 引数からエポック数とデータセットのパスを取得
    epochs = opt.epochs
    dataset = opt.data
    # csvファイルからデータを読み込む
    data = pd.read_csv(dataset)

    # ビデオファイルのパスと目標値をリストとして取得
    video_files = data['video_path'].tolist()
    targets = data['target'].tolist()

    # データを訓練用と評価用に分割する
    video_files_train, video_files_val, targets_train, targets_val = train_test_split(video_files, targets, test_size=0.1, random_state=42)


    # 訓練用と評価用のデータセットを作成する
    train_dataset = VideoAudioDataset(video_files_train, targets_train)
    val_dataset = VideoAudioDataset(video_files_val, targets_val)

    # データをロードする
    '''DataLoaderはデータのロードとバッチ処理を効率的に行うためのクラス。
    データセットからデータを読み込み、訓練ループで使用できる形式に変換する

    具体的には以下のような機能を持つ
    1. バッチ処理： データセットから複数の要素を一度に取得し、それらを一つのバッチとして返す。これはニューラルネットワークの訓練において重要なステップで、一度に大量のデータを処理することで計算効率を高め、また学習過程を安定化させることができる。
    2. シャッフル： エポックごとにデータセットをシャッフルする。これにより、各エポックでデータがランダムな順序で提供され、モデルが特定のデータの順序に依存しないようにすることができる。
    3. マルチスレッドデータロード： num_workersパラメータを使って複数のスレッドでデータを同時にロードすることができる。これにより、大量のデータを効率的にロードすることが可能になる。
    4. 自動化されたメモリ管理： データローダーはGPUメモリ内でのデータ管理を自動化する。これにより、訓練中に必要なデータが必要なタイミングで利用可能になり、メモリの効率的な使用が可能になる。

    '''
    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # ネットワークをインスタンス化
    video_net = VideoNet()
    audio_net = AudioNet()

    model = CombinedNet(video_net, audio_net) # ビデオとオーディオのネットワークを結合したネットワーク

    # 損失関数と最適化手法を定義
    criterion = nn.MSELoss()  # 平均二乗誤差を損失関数として使用
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 訓練中の損失の記録用リスト
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=7, verbose=True)
    # モデルの訓練
    for epoch in range(epochs): 
        model.train() # 訓練モードに設定
        running_loss = 0.0
        total_loss = 0
        for i, data in enumerate(train_loader, 0):
            # データローダーからデータを取得し、型をfloatに変換
            video_data, audio_data, targets = data
            print("Batch", i+1)
            for j in range(len(video_data)):
                print("Data", j+1)
                print("Frames shape:", video_data[j].shape)
                print("Audio spectrogram shape:", audio_data[j].shape)
            video_data = video_data.float().requires_grad_()
            audio_data = audio_data.float().requires_grad_()
            targets = targets.float()
            targets = targets.unsqueeze(1) # 目標値の次元を調整
                    
            optimizer.zero_grad() # 勾配をリセット

            # フォワードプロパゲーション
            outputs = model(video_data, audio_data)
            # 損失の計算とバックプロパゲーション
            loss = criterion(outputs, targets)
            loss.backward()
            # loss.backward() と optimizer.step() の間で使用します
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            # パラメータの更新
            optimizer.step()

            running_loss += loss.item()
            # ロスの計算と記録
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                train_losses.append(running_loss / 100)
                running_loss = 0.0
        # -------train process----------
        # -------val process--------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():  # 勾配計算を無効化
            for data in val_loader:
                # データローダーからデータを取得し、型をfloatに変換
                video_data, audio_data, targets = data
                video_data = video_data.float()
                audio_data = audio_data.float()
                targets = targets.float()
                targets = targets.unsqueeze(1)  # 目標値の次元を調整
                # フォワードプロパゲーション
                outputs = model(video_data, audio_data)
                # 損失の計算
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # 平均バリデーションロスを計算
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        print('Epoch: {}, Training Loss: {}, Validation Loss: {}'.format(epoch + 1, running_loss / len(train_loader), val_loss))
        # 早期停止の判定
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print('Finished Training')

    # モデルの評価
    model.eval() # 評価モードに設定
    with torch.no_grad():  # 勾配計算を無効化
        total_loss = 0
        for i, data in enumerate(val_loader, 0):
            # データローダーからデータを取得し、型をfloatに変換
            video_data, audio_data, targets = data
            video_data = video_data.float()
            audio_data = audio_data.float()
            targets = targets.float()
            targets = targets.unsqueeze(1) # 目標値の次元を調整
            # フォワードプロパゲーション
            outputs = model(video_data, audio_data)
            # 損失の計算
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            # ロスの計算と記録
            val_losses.append(total_loss / (i+1))

    # 訓練と評価の損失のグラフの作成
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig('loss_graph.png')
    plt.show()

    # 量子化を適用
    quantize_model(video_net)
    quantize_model(audio_net)
    quantize_model(model)  # CombinedNet
    # モデルの保存
    torch.save(model.state_dict(), 'optimized_model.pt')
    
    print('Validation Loss: %.3f' % (total_loss / len(val_loader)))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str, required=True, help='datafile path')
    parser.add_argument('--epochs',type=int, default=50, help='epochs')
    opt = parser.parse_args()
    print(opt)
    print('-----biginning training-----')
    train_multimodel(opt)
    print('-----completing training-----')
