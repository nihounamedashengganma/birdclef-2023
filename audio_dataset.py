import torch
from torch.utils.data import Dataset
import librosa
from augmentation import audio_augmentation
from preprocess import crop_or_pad,efficientnet_base_norm
from utils import get_spectrogram,spec_augmentation

class AudioDataset(Dataset):
    def __init__(self,audio_df,cfg,is_train=True):
        self.audio_df = audio_df
        self.cfg = cfg
        self.is_train = is_train
    
    
    def __len__(self):
        return len(self.audio_df)
    
    def __getitem__(self,idx):
        audio_path = self.audio_df['filepath'].iloc[idx]
        label = self.audio_df['class'].iloc[idx]
        aug_type = self.audio_df['aug_type'].iloc[idx]

        audio,sr = librosa.load(audio_path,sr=None)
        audio = torch.tensor(audio)
        audio = crop_or_pad(audio)
        audio= audio_augmentation(audio,aug_type,self.cfg)
     
        spec = get_spectrogram(audio)
        if aug_type==0 and self.is_train:
            spec = spec_augmentation(spec)
        spec = spec[0]

        spec_img = torch.stack([spec,spec,spec],dim=0)

        # do the normalization
        spec_img /= 255

        # normalizer = efficientnet_base_norm()
        # spec_img = normalizer(spec_img)
        
        return spec_img, label