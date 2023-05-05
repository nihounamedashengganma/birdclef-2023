

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

    prob_fm = 0.8
    
    min_pixel = -100.0
    max_pixel = 41.67259979248047