from random import choice

import numpy as np
from python_speech_features import fbank

from .model import NUM_FBANKS


def sample_from_mfcc(mfcc, max_length):
    if mfcc.shape[0] >= max_length:
        r = choice(range(0, len(mfcc) - max_length + 1))
        s = mfcc[r : r + max_length]
    else:
        s = pad_mfcc(mfcc, max_length)
    return np.expand_dims(s, axis=-1)


def calculate_nfft(samplerate, winlen):
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft


def read_mfcc(audio, sample_rate, win_length):
    energy = np.abs(audio)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    audio_voice_only = audio[offsets[0] : offsets[-1]]
    nfft = calculate_nfft(sample_rate, win_length / sample_rate)
    mfcc = mfcc_fbank(audio_voice_only, sample_rate, nfft)
    return mfcc


def pad_mfcc(mfcc, max_length):
    if len(mfcc) < max_length:
        mfcc = np.vstack(
            (mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1)))
        )
    return mfcc


def mfcc_fbank(signal: np.ndarray, sample_rate: int, nfft):
    filter_banks, energies = fbank(
        signal, samplerate=sample_rate, nfilt=NUM_FBANKS, nfft=nfft
    )
    return np.array(
        [(v - np.mean(v)) / max(np.std(v), 1e-12) for v in filter_banks],
        dtype=np.float32,
    )
