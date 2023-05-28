# 必要モジュールのインポート
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import cv2
from pydub import AudioSegment

import numpy as np

# 動画から情報を抽出するためのネットワークの定義
class VideoNet(nn.Module):
    def __init__(self):
        super(VideoNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16 * 16, 128)  # This needs to be adjusted based on input size and convolution/pooling operations
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 16 * 16 * 16)  # This needs to be adjusted based on input size and convolution/pooling operations
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 音声から情報を抽出するためのネットワークの定義
class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()
        self.fc1 = nn.Linear(40000, 120)  # adjust the input size to match the size of your audio features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 動画と音声の特徴量を組み合わせるためのネットワーク
class CombinedNet(nn.Module):
    def __init__(self, video_net, audio_net):
        super(CombinedNet, self).__init__()
        self.video_net = video_net
        self.audio_net = audio_net
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 1)  # adjust the output size to match your needs

    def forward(self, video_data, audio_data):
        video_output = self.video_net(video_data)
        audio_output = self.audio_net(audio_data)
        x = torch.cat((video_output, audio_output), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
