import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from config import CONFIG
from augmentation import audio_augmentation
from preprocess import crop_or_pad,efficientnet_base_norm
from utils import get_spectrogram,spec_augmentation,train_val_split,filter_data,upsample_data
import torch
from torch.utils.data import Dataset,DataLoader
from audio_dataset import AudioDataset_tmp

# 加载原始数据集
def load_meta_data(data_path):
    '''
    加载原始数据集
    '''
    meta_train2023 = pd.read_csv(os.path.join(data_path,"train_metadata.csv"))

    meta_train2023['filepath'] = os.path.join(data_path,'train_audio/')+meta_train2023['filename']
    # Encode Labels
    le = LabelEncoder()
    meta_train2023['class'] = le.fit_transform(meta_train2023['primary_label'])
    meta_train2023['aug_type'] = 0

    return meta_train2023


# 对训练样本做增强
def class_data_augmentation(df,aug_dic,num_classes,thr,curr_aug_type,seed):
    '''
    对训练样本中每一类进行不同类型的数据增强（audio阶段 & spec阶段），对于小类别样本，先上采样后再进行增强。然后将数据保存到本地
    
    args:
        df:原始训练集
        aug_dic:存储增强方法以及对应的数据范围 e.g. [0,0.2] 表示对前20%的数据做增强
        num_classes:类别个数
        thr:样本下限个数 i.e.不足该下限则做上采样
        curr_aug_type: 当前已有的aug_type数量 （包含原始类型0）
        seed:随机种子
    '''
    
    # copy一份原始训练集
    df_cp = df.copy()

    # 存储最终结果
    aug_dfs = list()

    for i in tqdm(range(num_classes)):
        tmp_df = df_cp[df_cp['class']==i].reset_index(drop=True)

        if len(tmp_df) < thr:
            # 小类别样本做上采样
            num_up = thr - len(tmp_df)
            upsample_tmp_df = tmp_df.sample(n=num_up,replace=True,random_state=seed)
            tmp_df = pd.concat([tmp_df,upsample_tmp_df]).reset_index(drop=True)
        
        # 做一次类内shuffle
        tmp_df = tmp_df.sample(frac=1.0,random_state=seed)
        tmp_df = tmp_df.reset_index(drop=True)

        num_total_sample = len(tmp_df)
        
        # 做不同类型的增强
        for idx,(k,v) in enumerate(aug_dic.items()):
            if v[1]!=1.0:
                tmp_df.loc[int(v[0]*num_total_sample):int(v[1]*num_total_sample),'aug_type'] = idx+curr_aug_type
            else:
                tmp_df.loc[int(v[0]*num_total_sample):,'aug_type'] = idx+curr_aug_type

        aug_dfs.append(tmp_df)
    
    df_aug = pd.concat(aug_dfs).reset_index(drop=True)

    return df_aug

def local_save(dataloader,save_path):

    output = list()

    min_pixel,max_pixel = list(),list()

    for (x,y) in tqdm(dataloader):
        output.append((x[0],y[0]))

        min_pixel.append(torch.min(x[0]).item())
        max_pixel.append(torch.max(x[0]).item())
    
    fin_min = min(min_pixel)
    fin_max = max(max_pixel)

    print(f'min_pixel:{fin_min},max_pixel:{max_pixel}')
    
    torch.save(output,save_path)

    



if __name__ == '__main__':

    meta_data_path = '/data'
    meta_train2023 = load_meta_data(meta_data_path)
    # 打上 #split
    meta_train2023 = train_val_split(meta_train2023)
    # 打上 小类别标记
    meta_train2023 = filter_data(meta_train2023)
    # 上采样
    up_df = upsample_data(meta_train2023,thr=20,seed=CONFIG.random_seed)

    # 划分训练和验证集
    train_df =up_df.query("fold!=4 | ~cv").reset_index(drop=True)
    valid_df = up_df.query("fold==4 & cv").reset_index(drop=True)

    # 总共包括10种增强方式，对应的aug_type 依次为 1，2，3，4，5，6, 7, 8, 9, 10
    aug_dic1 = {'gaussian_noise':[0,0.1],
                'time_stretch':[0.1,0.2],
                'pitch_shift':[0.2,0.3],
                'shift':[0.3,0.4],
                'background_noise':[0.4,0.5],
                'gaussian_snr':[0.5,0.6],
                'gain':[0.6,0.7],
                'reverse':[0.7,0.8],
                'freq_masking':[0.8,0.9],
                'time_masking':[0.9,1.0]}

    # 构建增强数据集1
    train_df_aug1 = class_data_augmentation(train_df,aug_dic1,CONFIG.num_classes,50,1,CONFIG.random_seed)

    
    # 合并
    train_df =pd.concat([train_df,train_df_aug1]).reset_index(drop=True)

    # 分别创建train/valid dataset
    training_data = AudioDataset_tmp(train_df,CONFIG,is_train=True)
    valid_data = AudioDataset_tmp(valid_df,CONFIG,is_train=False)

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)


    train_save_path = '/data/train_data.pth'
    valid_save_path = '/data/valid_data.pth'

    local_save(train_dataloader,train_save_path)
    local_save(valid_dataloader,valid_save_path)
