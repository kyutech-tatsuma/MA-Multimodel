# 必要なモジュールをインポートする
from torch import optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from model import VideoNet, AudioNet, CombinedNet
from dataset import VideoAudioDataset
import argparse

def train_multimodel(opt):
    epochs = opt.epochs
    dataset = opt.data
    # csvファイルの読み込み
    data = pd.read_csv(dataset)

    # ビデオのパスと動画に付与された数値を対応させる
    video_files = data['video_path'].tolist()
    targets = data['target'].tolist()

    # データを訓練用と評価用に分割する
    video_files_train, video_files_val, targets_train, targets_val = train_test_split(video_files, targets, test_size=0.2, random_state=42)

    # 訓練用と評価用のデータセットを作成する
    train_dataset = VideoAudioDataset(video_files_train)
    val_dataset = VideoAudioDataset(video_files_val)

    # データをロードする
    '''DataLoaderはデータのロードとバッチ処理を効率的に行うためのクラス。
    データセットからデータを読み込み、訓練ループで使用できる形式に変換する

    具体的には以下のような機能を持つ
    1. バッチ処理： データセットから複数の要素を一度に取得し、それらを一つのバッチとして返す。これはニューラルネットワークの訓練において重要なステップで、一度に大量のデータを処理することで計算効率を高め、また学習過程を安定化させることができる。
    2. シャッフル： エポックごとにデータセットをシャッフルする。これにより、各エポックでデータがランダムな順序で提供され、モデルが特定のデータの順序に依存しないようにすることができる。
    3. マルチスレッドデータロード： num_workersパラメータを使って複数のスレッドでデータを同時にロードすることができる。これにより、大量のデータを効率的にロードすることが可能になる。
    4. 自動化されたメモリ管理： データローダーはGPUメモリ内でのデータ管理を自動化する。これにより、訓練中に必要なデータが必要なタイミングで利用可能になり、メモリの効率的な使用が可能になる。

    '''
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # ネットワークをインスタンス化
    video_net = VideoNet()
    audio_net = AudioNet()

    # 二つのネットワークを統一するネットワークもインスタンス化
    model = CombinedNet(video_net, audio_net)

    # 損失関数と評価関数を定義
    criterion = nn.MSELoss()  # 損失関数に平均2乗誤差を使用
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 損失関数の変化を記録するための配列
    train_losses = []
    val_losses = []

    # ネットワークの学習
    for epoch in range(epochs): 
        # 一回の学習ごとに損失関数の値を初期化する
        running_loss = 0.0
        # データの中身をループさせる
        for i, data in enumerate(train_loader, 0):
            # train_loaderの要素からデータを取得する
            video_data, audio_data, targets = data

            '''既存の勾配をリセットする。
            PyTorchでは勾配を累積するという動作を持つため、非常に重要な関数。
            ニューラルネットワークの訓練では、通常誤差を計算し、その誤差を用いて各層のパラメータを更新する。この時、誤差は各パラメータに対する勾配として表される。
            PyTorchでは、勾配は各パラメータに対して累積される。これは、RNNのような再起的なネットワークの訓練では役に立つが、一般的なネットワークの訓練では、各更新ステップごとに勾配をリセットすることが必要。
            '''
            optimizer.zero_grad()

            # ネットワークに学習させ、損失関数、評価関数の計算を行う
            outputs = model(video_data, audio_data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # ロスの加算
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                train_losses.append(running_loss / 2000)
                running_loss = 0.0

    print('Finished Training')

    # モデルの保存
    torch.save(model.state_dict(), 'combined_model.pth')

    # モデルの評価を行う
    model.eval()
    with torch.no_grad():  # 評価用データセットを使って評価を行う
        total_loss = 0
        for i, data in enumerate(val_loader, 0):
            video_data, audio_data, targets = data
            outputs = model(video_data, audio_data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            val_losses.append(total_loss / (i+1))

    # 学習過程の可視化
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

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
