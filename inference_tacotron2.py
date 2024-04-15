import numpy as np
import torch
from hifigan import HiFiGAN, HiFiGANConfig, HiFiGANAudioConfig
from model import Tacotron2, Tacotron2Config, Tacotron2TrainState
from dataset import Tacotron2Dataset, Tacotron2AudioConfig, Tacotron2SpeakerDataset
from scipy.io.wavfile import write

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hifigan = HiFiGAN.load("./model_best_0.2960.pt", device=device)
tacotron2 = Tacotron2.load(
    "./output/checkpoints/t9/model_490000_0.2663.pt", device=device
)

speaker_dataset = Tacotron2SpeakerDataset("/mnt/workspace/AI/audio_output/kss")

mels, audios, attns = tacotron2(
    speakers=speaker_dataset.embeds(),
    inputs=["오늘은 십이월 이십 구일이다."],
    vocoder=hifigan,
    include_attention=True,
)

write("test.wav", 44100, (audios[0].cpu().numpy().squeeze() * 32768.0).astype(np.int16))
