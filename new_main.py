!pip install audiomentations
import librosa
from IPython.display import Audio
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torchaudio
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import audiomentations
from sklearn.preprocessing import LabelEncoder
import random
import cv2

from config import CONFIG
from augmentation import audio_augmentation
from preprocess import crop_or_pad,efficientnet_base_norm
from utils import get_spectrogram,spec_augmentation,train_val_split,filter_data,upsample_data
from audio_dataset import AudioDataset
from model import build_model


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 固定随机数种子
fix_seed(CONFIG.random_seed)

def prepare_data(data_path):
    # Load metadata
    meta_train2023 = pd.read_csv(os.path.join(data_path,"train_metadata.csv"))

    meta_train2023['filepath'] = os.path.join(data_path,'train_audio/')+meta_train2023['filename']
    # Encode Labels
    le = LabelEncoder()
    meta_train2023['class'] = le.fit_transform(meta_train2023['primary_label'])
    meta_train2023['aug_type'] = 0

    return meta_train2023

data_path = '/data'
meta_train2023 = prepare_data(data_path)


# 打上 #split
meta_train2023 = train_val_split(meta_train2023)

# 打上 小类别标记
meta_train2023 = filter_data(meta_train2023)

# 上采样
up_df = upsample_data(meta_train2023,thr=20,seed=CONFIG.random_seed)

# 划分训练和验证集
train_df =up_df.query("fold!=4 | ~cv").reset_index(drop=True)
valid_df = up_df.query("fold==4 & cv").reset_index(drop=True)

# 对训练样本做增强
aug_train_df_li = [train_df]

for i in range(1,5): # 4 types of augmentations
    aug_train_df = train_df.copy()
    aug_train_df['aug_type'] = i
    aug_train_df_li.append(aug_train_df)

# 包括原始数据 + 4种不同类型的增强数据
train_df = pd.concat(aug_train_df_li).reset_index(drop=True)

training_data = AudioDataset(train_df,CONFIG,is_train=True)
valid_data = AudioDataset(valid_df,CONFIG,is_train=False)

train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=32, shuffle=False)


model =  build_model()

EPOCHS=30
LR = 5e-4
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


model_train(model,
            'pre-train-efficientnetb0',
            train_dataloader,
            valid_dataloader,
            EPOCHS,
            LR,
            DEVICE)








