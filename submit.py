import librosa
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import torchaudio
import random
from tqdm import tqdm
from glob import glob
from preprocess import crop_or_pad,efficientnet_base_norm,downstream_data_norm
from utils import get_spectrogram




# for test
class CONFIG:
    random_seed= 42
    
    sample_rate = 32000
    
    duration=5
    train_duration=10
    
    audio_len = train_duration*sample_rate
    target_len = train_duration*sample_rate
    
    img_size = [128,256]
    num_classes = 264
    batch_size = 32
    
    nfft = 2028
    window = 2048
    hop_length = audio_len // (img_size[1] - 1)
    fmin = 500
    fmax = 14000
    normalize = True

    prob_fm = 0.8

    min_pixel = -100.0
    max_pixel = 41.67259979248047



# some utils

def build_test_data(test_dir):
    
    test_path_format = os.path.join(test_dir,'*ogg')
    
    test_paths = glob(test_path_format)
    
    test_df = pd.DataFrame(test_paths, columns=['filepath'])
    test_df['filename'] = test_df.filepath.map(lambda x: x.split('/')[-1].replace('.ogg',''))
    
    return test_df

def load_audio(filepath, sr):
    audio, orig_sr = librosa.load(filepath, sr=None)
    
    if sr!=orig_sr:
        audio = librosa.resample(y, orig_sr, sr)
    
    return audio




class Test_AudioDataset(Dataset):
    def __init__(self,test_df,cfg):
        self.test_df = test_df
        self.cfg = cfg
    
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self,idx):
        audio_path = self.test_df['filepath'].iloc[idx]
        audio_name = self.test_df['filename'].iloc[idx]
        
        # 读取mixed audio
        mix_audios = load_audio(audio_path,self.cfg.sample_rate)
        
        # 存储每段独立audio的row_id 以及 对应的audio内容
        ids,audios = list(),list()
        
        # 每段独立audio的长度
        simple_audio_len = self.cfg.duration * self.cfg.sample_rate
        
        for idx,i in enumerate(range(0,len(mix_audios),simple_audio_len)):
            ids.append(f'{audio_name}_{(idx+1)*self.cfg.duration}')
            
            simple_audio_start = i
            simple_audio_end = i + simple_audio_len
            
            audios.append(mix_audios[start:end])
        
        audios = [torch.tensor(audio) for audio in audios]
        audios = [crop_or_pad(audio) for audio in audios] # padding
        
        specs = [get_spectrogram(audio)[0] for audio in audios] # get mel-spec
        
        spec_imgs = [torch.stack([spec,spec,spec],dim=0) for spec in specs]

        # scale to [0,1]
        sspec_imgs = [downstream_data_norm(spec_img,self.cfg.min_pixel,self.cfg.max_pixel) for spec_img in spec_imgs]

        normalizer = efficientnet_base_norm()
        
        # normalize base on efficientnet train resources
        spec_imgs = [normalizer(spec_img) for spec_img in spec_imgs]
        
        stacked_spec_imgs = torch.stack(spec_imgs,dim=0)
        
        return ids,stacked_spec_imgs  


