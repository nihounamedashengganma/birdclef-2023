import torch
from config import CONFIG
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift,SpecFrequencyMask


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