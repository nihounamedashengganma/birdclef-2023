!pip install audiomentations
import librosa
from IPython.display import Audio
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchaudio
from torch.utils.data import DataLoader
import audiomentations
from sklearn.preprocessing import LabelEncoder
import torchaudio
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift,SpecFrequencyMask
import torchvision
import cv2


# 固定随机数种子
def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class CONFIG:
    random_seed= 42
    
    sample_rate = 32000
    duration=10
    audio_len = duration*sample_rate
    target_len = duration*sample_rate
    
    img_size = [128,256]
    num_classes = 264
    batch_size = 32
    
    nfft = 2028
    window = 2048
    hop_length = audio_len // (img_size[1] - 1)
    fmin = 500
    fmax = 14000
    normalize = True

    prob_fm = 0.45


fix_seed(CONFIG.random_seed)

data_path = '/kaggle/input/birdclef-2023'

# Load metadata
meta_train2023 = pd.read_csv(os.path.join(data_path,"train_metadata.csv"))

meta_train2023['filepath'] = os.path.join(data_path,'train_audio/')+meta_train2023['filename']
# Encode Labels
le = LabelEncoder()
meta_train2023['class'] = le.fit_transform(meta_train2023['primary_label'])


def crop_or_pad(audio, target_len = CONFIG.target_len
                , pad_mode='constant', take_first=True):
    audio_len = audio.shape[0]
    diff_len = abs(target_len - audio_len)
    if audio_len < target_len:
        # padding audio tensor at a random start point
        pad1 = torch.randint(high=diff_len, size=[], dtype=torch.int32)
        pad2 = diff_len - pad1
        audio = torch.nn.functional.pad(audio, (pad1.item(), pad2.item()), mode=pad_mode)
    elif audio_len > target_len:
        if take_first:
            audio = audio[:target_len]
        else:
            idx = torch.randint(high=diff_len, size=[], dtype=torch.int32)
            audio = audio[idx.item(): (idx.item() + target_len)]
    return torch.reshape(audio, [target_len])


def audio_augment(audio):
    augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.35),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.45),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.45),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.45),
    ])
    audio = audio.numpy()
    aug_audio = augment(samples=audio, sample_rate=CONFIG.sample_rate)
    return torch.tensor(aug_audio)


def get_spectrogram(audio):
    audio = audio.numpy()
    spec = librosa.feature.melspectrogram(y=audio, 
                                   sr=CONFIG.sample_rate, 
                                   n_mels=CONFIG.img_size[0],
                                   n_fft=CONFIG.nfft,
                                   hop_length=CONFIG.hop_length,
                                   fmax=CONFIG.fmax,
                                   fmin=CONFIG.fmin,
                                   )
    spec = librosa.power_to_db(spec, ref=1.0)
    return torch.tensor(spec).reshape((1, spec.shape[0],spec.shape[1]))


def spec_augmentation(spec):
    # TimeFrequencyMask
    #  Add random factor to Frequency Masking
    rand_prob = float(torch.rand(1))
    transfom1 = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
    transfom2 = torchaudio.transforms.TimeMasking(time_mask_param=25,p=0.5)
    if rand_prob >= CONFIG.prob_fm:
        spec = transfom1(spec)
        spec = transfom2(spec)
    else:
        spec = transfom2(spec)
    return spec


def spec_norm(mel_spectrogram):
    mel_spectrogram = mel_spectrogram[0]
    mean = torch.mean(mel_spectrogram, dim=1)
    std = torch.std(mel_spectrogram, dim=1)
    # Expand the mean and standard deviation tensors to have the same shape as the original spectrogram
    mean = mean.unsqueeze(1).expand_as(mel_spectrogram)
    std = std.unsqueeze(1).expand_as(mel_spectrogram)

    # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation
    normalized_mel_spectrogram = (mel_spectrogram - mean) / std

    # You can also apply additional operations such as clipping or scaling if desired
    # For example, to clip the normalized spectrogram to a specific range, you can use torch.clamp:
    min_value = 0.0
    max_value = 1.0
    h = mel_spectrogram.shape[0]
    w = mel_spectrogram.shape[1]
    normalized_mel_spectrogram = torch.clamp(normalized_mel_spectrogram, min=min_value, max=max_value)
    return normalized_mel_spectrogram.reshape((1,h,w))


def normalize_transform(pretrained=True):
    from torchvision import transforms
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize


class AudioDataset(Dataset):
    def __init__(self,audio_df, transform=None, is_train=True):
        self.audio_df = audio_df
        self.transform = transform
        self.is_train = is_train
    
    
    def __len__(self):
        return len(self.audio_df)
    
    def __getitem__(self,idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        audio_path = self.audio_df['filepath'].iloc[idx]
        label = self.audio_df['class'].iloc[idx]
        audio,sr = librosa.load(audio_path,sr=None)
        audio = torch.tensor(audio)
        audio = crop_or_pad(audio)
#         label_1hot = one_hot(torch.tensor(label), num_classes=CONFIG.num_classes)
#         if self.is_train:
#             audio= audio_augment(audio)
        # Spectrumgram Augmentations
        spec = get_spectrogram(audio)
        if self.is_train:
            spec = spec_augmentation(spec).reshape((1, spec[0].shape[0], 
            spec[0].shape[1]))
        spec = spec[0]

        spec_img = torch.stack([spec,spec,spec],dim=0)
        
        # normalize
        normalizer = normalize_transform()
        spec_img = normalizer(spec_img)
        
#         sample  = {'img':spec_img,'label':label_1hot}
        
#         if self.transform:
#             sample = self.transform(sample)
        
        
        return spec_img, label


dataset  =  AudioDataset(meta_train2023)


from sklearn.model_selection import StratifiedKFold
seed = CONFIG.random_seed

# Initialize the StratifiedKFold object with 5 splits and shuffle the data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
# Reset the index of the dataframe
meta_train2023 = meta_train2023.reset_index(drop=True)

# Create a new column in the dataframe to store the fold number for each row
meta_train2023["fold"] = -1

# Iterate over the folds and assign the corresponding fold number to each row in the dataframe
for fold, (train_idx, val_idx) in enumerate(skf.split(meta_train2023, 
meta_train2023['primary_label'])):
    meta_train2023.loc[val_idx, 'fold'] = fold


def filter_data(df, thr=5):
    # Count the number of samples for each class
    counts = df.primary_label.value_counts()

    # Condition that selects classes with less than `thr` samples
    cond = df.primary_label.isin(counts[counts<thr].index.tolist())

    # Add a new column to select samples for cross validation
    df['cv'] = True

    # Set cv = False for those class where there is samples less than thr
    df.loc[cond, 'cv'] = False

    # Return the filtered dataframe
    return df


#### -oversample for minority class
def upsample_data(df, thr=20,seed=666):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df

meta_train2023 = filter_data(meta_train2023, thr=5)
up_df = upsample_data(meta_train2023, thr=50,seed=CONFIG.random_seed)


train_df =up_df.query("fold!=4 | ~cv").reset_index(drop=True)
valid_df = up_df.query("fold==4 &cv").reset_index(drop=True)

training_data = AudioDataset(train_df, is_train=True)
valid_data = AudioDataset(valid_df, is_train=False)

from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=32, shuffle=False)


import torchvision.models as models
import torch.nn as nn
def build_model(pretrained=True, fine_tune=True, num_classes=CONFIG.num_classes):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b0(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1280,out_features=CONFIG.num_classes)
    return model


model = build_model()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


def model_eval(model,val_dataloader,device):

    model.eval()

    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for step,(x,y) in enumerate(val_dataloader):
            x,y = x.to(device),y.to(device)
            
            logits = model(x)

            batch_loss = criterion(logits,y)

            total_loss += batch_loss
        
        total_loss /= (step+1)
    
    return total_loss


def model_train(model,model_name,train_dataloader,val_dataloader,epochs,lr,device):

    model = model.to(device)

    # define optimizer
    optimizer = Adam(model.parameters(),lr = lr)
    # define loss func
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_epoch,min_loss = 0,1e10

    for epoch in range(1,epochs+1):
        # train mode
        model.train()

        train_loss = 0.0

        for step,(x,y) in tqdm(enumerate(train_dataloader)):
            x,y = x.to(device),y.to(device) 
      
            logits = model(x)

            batch_loss = criterion(logits,y)

            optimizer.zero_grad() # 清除梯度
            batch_loss.backward() # 计算梯度

            optimizer.step() # 反向传播

            train_loss += batch_loss
        
        train_loss /= (step+1)

        # eval
        eval_loss = model_eval(model,val_dataloader,device)

        print('at epoch {} the training loss is {} the eval loss is {}'.format(epoch,train_loss,eval_loss))

        if eval_loss < min_loss:
            min_loss = eval_loss
            best_epoch = epoch

            torch.save(model.state_dict(),f'{model_name}.pkl')


EPOCHS=30
LR = 5e-4
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'


model_train(model,
            'pre-train-efficientnetb0',
            train_dataloader,
            valid_dataloader,
            EPOCHS,
            LR,
            DEVICE)
