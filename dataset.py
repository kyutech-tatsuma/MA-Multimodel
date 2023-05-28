# 必要モジュールのインポート
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import cv2
from pydub import AudioSegment

import numpy as np

# データセットクラスの定義
class VideoAudioDataset(Dataset):
    def __init__(self, video_files, transform=None):
        self.video_files = video_files
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]

        # Load video and extract frames
        video = cv2.VideoCapture(video_file)
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        video.release()

        # Load audio and convert to spectrogram
        audio = AudioSegment.from_file(video_file)
        audio_samples = audio.get_array_of_samples()
        audio_spectrogram = np.abs(np.fft.rfft(audio_samples))

        return torch.tensor(frames), torch.tensor(audio_spectrogram)
