import torch
import torchaudio
import librosa
from config import CONFIG
from sklearn.model_selection import StratifiedKFold


def get_spectrogram(audio):
    '''
    获得audio的梅尔谱图特征
    '''
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
    '''
    梅尔谱图增强
    '''
    # TimeFrequencyMask
    #  Add random factor to Frequency Masking
    rand_prob = float(torch.rand(1))
    transfom1 = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
    transfom2 = torchaudio.transforms.TimeMasking(time_mask_param=25,p=0.2)
    if rand_prob >= CONFIG.prob_fm:
        spec = transfom1(spec)
        spec = transfom2(spec)
    else:
        spec = transfom2(spec)
    return spec

def spec_augmentation_sep(spec,aug_type):
    '''
    单一维度的梅尔谱图增强
    '''
    augments = [torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
                torchaudio.transforms.TimeMasking(time_mask_param=25)]

    spec = augments[aug_type-5](spec)

    return spec


def train_val_split(df,n_splits=5):
    '''
    划分训练和验证数据
    '''
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CONFIG.random_seed)
    # Reset the index of the dataframe
    df = df.reset_index(drop=True)

    # Create a new column in the dataframe to store the fold number for each row
    df["fold"] = -1

    # Iterate over the folds and assign the corresponding fold number to each row in the dataframe
    for fold, (train_idx, val_idx) in enumerate(skf.split(df,df['primary_label'])):
        df.loc[val_idx, 'fold'] = fold
    
    return df

def filter_data(df, thr=5):
    '''
    筛选小类别数据，后续不放入验证集中
    '''
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


