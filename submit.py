import librosa
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import random
from tqdm import tqdm
from glob import glob


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

class2label = {0: 'abethr1',
 1: 'abhori1',
 2: 'abythr1',
 3: 'afbfly1',
 4: 'afdfly1',
 5: 'afecuc1',
 6: 'affeag1',
 7: 'afgfly1',
 8: 'afghor1',
 9: 'afmdov1',
 10: 'afpfly1',
 11: 'afpkin1',
 12: 'afpwag1',
 13: 'afrgos1',
 14: 'afrgrp1',
 15: 'afrjac1',
 16: 'afrthr1',
 17: 'amesun2',
 18: 'augbuz1',
 19: 'bagwea1',
 20: 'barswa',
 21: 'bawhor2',
 22: 'bawman1',
 23: 'bcbeat1',
 24: 'beasun2',
 25: 'bkctch1',
 26: 'bkfruw1',
 27: 'blacra1',
 28: 'blacuc1',
 29: 'blakit1',
 30: 'blaplo1',
 31: 'blbpuf2',
 32: 'blcapa2',
 33: 'blfbus1',
 34: 'blhgon1',
 35: 'blhher1',
 36: 'blksaw1',
 37: 'blnmou1',
 38: 'blnwea1',
 39: 'bltapa1',
 40: 'bltbar1',
 41: 'bltori1',
 42: 'blwlap1',
 43: 'brcale1',
 44: 'brcsta1',
 45: 'brctch1',
 46: 'brcwea1',
 47: 'brican1',
 48: 'brobab1',
 49: 'broman1',
 50: 'brosun1',
 51: 'brrwhe3',
 52: 'brtcha1',
 53: 'brubru1',
 54: 'brwwar1',
 55: 'bswdov1',
 56: 'btweye2',
 57: 'bubwar2',
 58: 'butapa1',
 59: 'cabgre1',
 60: 'carcha1',
 61: 'carwoo1',
 62: 'categr',
 63: 'ccbeat1',
 64: 'chespa1',
 65: 'chewea1',
 66: 'chibat1',
 67: 'chtapa3',
 68: 'chucis1',
 69: 'cibwar1',
 70: 'cohmar1',
 71: 'colsun2',
 72: 'combul2',
 73: 'combuz1',
 74: 'comsan',
 75: 'crefra2',
 76: 'crheag1',
 77: 'crohor1',
 78: 'darbar1',
 79: 'darter3',
 80: 'didcuc1',
 81: 'dotbar1',
 82: 'dutdov1',
 83: 'easmog1',
 84: 'eaywag1',
 85: 'edcsun3',
 86: 'egygoo',
 87: 'equaka1',
 88: 'eswdov1',
 89: 'eubeat1',
 90: 'fatrav1',
 91: 'fatwid1',
 92: 'fislov1',
 93: 'fotdro5',
 94: 'gabgos2',
 95: 'gargan',
 96: 'gbesta1',
 97: 'gnbcam2',
 98: 'gnhsun1',
 99: 'gobbun1',
 100: 'gobsta5',
 101: 'gobwea1',
 102: 'golher1',
 103: 'grbcam1',
 104: 'grccra1',
 105: 'grecor',
 106: 'greegr',
 107: 'grewoo2',
 108: 'grwpyt1',
 109: 'gryapa1',
 110: 'grywrw1',
 111: 'gybfis1',
 112: 'gycwar3',
 113: 'gyhbus1',
 114: 'gyhkin1',
 115: 'gyhneg1',
 116: 'gyhspa1',
 117: 'gytbar1',
 118: 'hadibi1',
 119: 'hamerk1',
 120: 'hartur1',
 121: 'helgui',
 122: 'hipbab1',
 123: 'hoopoe',
 124: 'huncis1',
 125: 'hunsun2',
 126: 'joygre1',
 127: 'kerspa2',
 128: 'klacuc1',
 129: 'kvbsun1',
 130: 'laudov1',
 131: 'lawgol',
 132: 'lesmaw1',
 133: 'lessts1',
 134: 'libeat1',
 135: 'litegr',
 136: 'litswi1',
 137: 'litwea1',
 138: 'loceag1',
 139: 'lotcor1',
 140: 'lotlap1',
 141: 'luebus1',
 142: 'mabeat1',
 143: 'macshr1',
 144: 'malkin1',
 145: 'marsto1',
 146: 'marsun2',
 147: 'mcptit1',
 148: 'meypar1',
 149: 'moccha1',
 150: 'mouwag1',
 151: 'ndcsun2',
 152: 'nobfly1',
 153: 'norbro1',
 154: 'norcro1',
 155: 'norfis1',
 156: 'norpuf1',
 157: 'nubwoo1',
 158: 'pabspa1',
 159: 'palfly2',
 160: 'palpri1',
 161: 'piecro1',
 162: 'piekin1',
 163: 'pitwhy',
 164: 'purgre2',
 165: 'pygbat1',
 166: 'quailf1',
 167: 'ratcis1',
 168: 'raybar1',
 169: 'rbsrob1',
 170: 'rebfir2',
 171: 'rebhor1',
 172: 'reboxp1',
 173: 'reccor',
 174: 'reccuc1',
 175: 'reedov1',
 176: 'refbar2',
 177: 'refcro1',
 178: 'reftin1',
 179: 'refwar2',
 180: 'rehblu1',
 181: 'rehwea1',
 182: 'reisee2',
 183: 'rerswa1',
 184: 'rewsta1',
 185: 'rindov',
 186: 'rocmar2',
 187: 'rostur1',
 188: 'ruegls1',
 189: 'rufcha2',
 190: 'sacibi2',
 191: 'sccsun2',
 192: 'scrcha1',
 193: 'scthon1',
 194: 'shesta1',
 195: 'sichor1',
 196: 'sincis1',
 197: 'slbgre1',
 198: 'slcbou1',
 199: 'sltnig1',
 200: 'sobfly1',
 201: 'somgre1',
 202: 'somtit4',
 203: 'soucit1',
 204: 'soufis1',
 205: 'spemou2',
 206: 'spepig1',
 207: 'spewea1',
 208: 'spfbar1',
 209: 'spfwea1',
 210: 'spmthr1',
 211: 'spwlap1',
 212: 'squher1',
 213: 'strher',
 214: 'strsee1',
 215: 'stusta1',
 216: 'subbus1',
 217: 'supsta1',
 218: 'tacsun1',
 219: 'tafpri1',
 220: 'tamdov1',
 221: 'thrnig1',
 222: 'trobou1',
 223: 'varsun2',
 224: 'vibsta2',
 225: 'vilwea1',
 226: 'vimwea1',
 227: 'walsta1',
 228: 'wbgbir1',
 229: 'wbrcha2',
 230: 'wbswea1',
 231: 'wfbeat1',
 232: 'whbcan1',
 233: 'whbcou1',
 234: 'whbcro2',
 235: 'whbtit5',
 236: 'whbwea1',
 237: 'whbwhe3',
 238: 'whcpri2',
 239: 'whctur2',
 240: 'wheslf1',
 241: 'whhsaw1',
 242: 'whihel1',
 243: 'whrshr1',
 244: 'witswa1',
 245: 'wlwwar',
 246: 'wookin1',
 247: 'woosan',
 248: 'wtbeat1',
 249: 'yebapa1',
 250: 'yebbar1',
 251: 'yebduc1',
 252: 'yebere1',
 253: 'yebgre1',
 254: 'yebsto1',
 255: 'yeccan1',
 256: 'yefcan',
 257: 'yelbis1',
 258: 'yenspu1',
 259: 'yertin1',
 260: 'yesbar1',
 261: 'yespet1',
 262: 'yetgre1',
 263: 'yewgre1'}


def crop_or_pad(audio, target_len = CONFIG.target_len
                , pad_mode='constant', take_first=True):
    '''
    audio 尺寸统一
    '''
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


def downstream_data_norm(data,min_v,max_v):
    '''
    基于训练样本对数据做min-max normalization
    '''
    return (data - min_v) / (max_v - min_v)


def efficientnet_base_norm(pretrained=True):
    '''
    基于efficientnet训练样本的图像特征（mean & std)对下游任务样本做归一化
    '''
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


def load_model(modelpath):
    import torchvision.models as models
    import torch.nn as nn

    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(in_features=1280,out_features=CONFIG.num_classes)
    
    model.load_state_dict(torch.load(modelpath,map_location='cpu'))
    
    return model


def predict(test_dataset,model):
    from collections import defaultdict
    
    model.eval()
    
    output = defaultdict(list)
    
    for i in tqdm(range(len(test_dataset))):
        ids,imgs = test_dataset[i]
        output['row_id'] += ids
        
        with torch.no_grad():
            logits = model(imgs)
            probs = F.softmax(logits,dim=-1).detach().numpy()
            
            for c in range(CONFIG.num_classes):
                output[class2label.get(c)] += probs[:,c].tolist()
    
    
    output_df = pd.DataFrame(output)
    
    return output_df


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


if __name__ == '__main__':
    
    test_dir = '/data/test_soundscapes'
    test_data = build_test_data(test_dir)

    test_dataset = Test_AudioDataset(test_df=test_data,
                                     cfg=CONFIG)
    
    modelpath = 'xxx'
    model = load_model(modelpath)

    output_df = predict(test_dataset,model)
    output_df.to_csv('submission.csv',index=False)


