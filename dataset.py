# 必要モジュールのインポート
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from pydub import AudioSegment
import math
import numpy as np

# データセットクラスの定義
class VideoAudioDataset(Dataset):
    def __init__(self, video_files, targets, transform=None, resize_shape=(112, 112)):# 30 seconds at 44.1 kHz
        self.video_files = video_files
        self.targets = targets
        self.transform = transform
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        max_frames = 5000
        max_audio_length = 44100 * 30  # 30 seconds at 44.1 kHz

        target= self.targets[idx]

        video_file = self.video_files[idx]

        # Load video and extract frames
        video = cv2.VideoCapture(video_file)
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.resize_shape)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        video.release()

        # Downsample frames if necessary
        if len(frames) > max_frames:
            frames = frames[::len(frames)//max_frames][:max_frames]
        frames = np.array(frames)
        frames = np.transpose(frames, (3, 0, 1, 2))
        # Load audio and convert to spectrogram
        audio = AudioSegment.from_file(video_file)

        audio_samples = audio.get_array_of_samples()
        # Pad or trim audio samples
        if len(audio_samples) > max_audio_length:
            audio_samples = audio_samples[:max_audio_length]
        else:
            audio_samples += [0] * (max_audio_length - len(audio_samples))
        audio_spectrogram = np.abs(np.fft.rfft(audio_samples))
        
        print("Frames shape:", frames.shape)
        print("Audio spectrogram shape:", audio_spectrogram.shape)


        return torch.tensor(frames), torch.tensor(audio_spectrogram), target
