import torch
from config import CONFIG


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