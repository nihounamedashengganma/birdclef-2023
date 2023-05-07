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
import random
import cv2
from config import CONFIG
from augmentation import audio_augmentation
from preprocess import crop_or_pad,efficientnet_base_norm,downstream_data_norm
from utils import get_spectrogram,spec_augmentation,train_val_split,filter_data,upsample_data
from audio_dataset import AudioDataset_fin
from model import build_model
from train import model_eval,model_train

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 固定随机数种子
fix_seed(CONFIG.random_seed)



if __name__ == '__main__':

    # load locally
    train_save_path = '/data/train_data.pth'
    valid_save_path = '/data/valid_data.pth'

    local_train_data = torch.load(train_save_path)
    local_valid_data = torch.load(valid_save_path)

    training_data = AudioDataset_fin(local_train_data,CONFIG,is_train=True)
    valid_data = AudioDataset_fin(local_valid_data,CONFIG,is_train=False)

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








