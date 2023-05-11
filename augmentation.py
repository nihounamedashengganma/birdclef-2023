import os
import pandas as pd
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask, AddGaussianSNR, Gain, Reverse 


def audio_augmentation(audio,aug_type,cfg):
    if aug_type==0:
        return audio
    else:
        augments = [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
            PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0),
            TimeMask(p=1.0),
            AddGaussianSNR(p=1.0),
            Gain(p=1.0),
            Reverse(p=1.0)
        ]
        audio = audio.numpy()
        aug_audio = augments[aug_type-1](samples=audio, sample_rate=cfg.sample_rate)
    
        return torch.tensor(aug_audio)


