import torch
import numpy as np
import os
from collections import OrderedDict
import librosa
import random

def get_dataset_filelist(input_training_file):
    with open(input_training_file, 'r', encoding='utf-8') as fi:
        training_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]
    return training_indexes

class MPNet_Dataset(torch.utils.data.Dataset):
    def __init__(self, training_indexes, clean_wavs_dir, noisy_wavs_dir, segment_size, 
                sampling_rate, split=True, shuffle=True, n_cache_reuse=1, device=None):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        if self._cache_ref_count == 0:
            clean_audio, _ = librosa.load(os.path.join(self.clean_wavs_dir, filename + '.wav'), sr=self.sampling_rate)
            noisy_audio, _ = librosa.load(os.path.join(self.noisy_wavs_dir, filename + '.wav'), sr=self.sampling_rate)
            length = min(len(clean_audio), len(noisy_audio))
            clean_audio, noisy_audio = clean_audio[: length], noisy_audio[: length]
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1
        
        clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start: audio_start+self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start+self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)), 'constant')

        return (clean_audio.squeeze(), noisy_audio.squeeze())

    def __len__(self):
        return len(self.audio_indexes)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1]+(1e-10), stft_spec[:, :, :, 0]+(1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com

def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav

def sisdr(x, s, remove_dc=True):
    """
    Compute SI-SDR
    x: extracted signal
    s: reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    rename_map = {
        ".ccm.0.": ".norm.",
        ".ccm.2.": ".conv_in.",
        ".ccm.4.": ".depthwise_conv.",
        ".ccm.5.": ".bn.",
        ".ccm.7.": ".conv_out."
    }

    print("Loading '{}'".format(filepath))
    state_dict = torch.load(filepath, map_location=device)
    new_state_dict = {}
    odict = state_dict['state_dict']

    new_odict = OrderedDict()

    for k, v in odict.items():
        new_key = k        
        if "mask_decoder.dense_block" in k:
            parts = k.split(".")
            if parts[-2] in {"0", "1", "2"}:
                parts[-2] = str(int(parts[-2]) + 1)
            new_key = ".".join(parts)
        
        elif "phase_decoder.dense_block" in k:
            parts = k.split(".")
            if parts[-2] in {"0", "1", "2"}:
                parts[-2] = str(int(parts[-2]) + 1)
            new_key = ".".join(parts)
        
        elif "dense_encoder.dense_block" in k:
            parts = k.split(".")
            if parts[-2] in {"0", "1", "2"}:
                parts[-2] = str(int(parts[-2]) + 1)
            new_key = ".".join(parts)

        for old, new in rename_map.items():
            if old in k:
                new_key = k.replace(old, new)
                a=1
                break

        new_odict[new_key] = v

    new_state_dict['state_dict'] = new_odict
    print("Complete.")
    return new_state_dict