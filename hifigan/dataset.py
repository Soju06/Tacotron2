import os
from pathlib import Path
import random
from typing import Iterable
import numpy as np
import torch
import librosa
from librosa.util import normalize
from torch.utils.data import Dataset

from .stft import STFT
from .config import HiFiGANAudioConfig


class HiFiGANDataset(Dataset):
    path: Path | None
    config: HiFiGANAudioConfig
    segment_length: int
    stft: STFT
    normalize: bool

    audio_files: list[Path] | None

    ignore_sampling_rate_error: bool = False

    def __init__(
        self,
        path: str | os.PathLike | None,
        config: HiFiGANAudioConfig,
        segment_length: int = 8192,
        pattern: str | Iterable[str] = "*.wav",
        normalize: bool = True,
        ignore_sampling_rate_error: bool = False,
    ):
        self.path = Path(path) if path else None
        self.config = config
        self.segment_length = segment_length
        self.stft = config.stft()
        self.normalize = normalize
        self.ignore_sampling_rate_error = ignore_sampling_rate_error

        if self.path:
            if isinstance(pattern, str):
                pattern = [pattern]

            self.audio_files = [
                file for p in pattern for file in self.path.glob(p) if file.is_file()
            ]

        else:
            self.audio_files = None

    def load_audio(self, path: str | os.PathLike) -> torch.FloatTensor:
        audio, sampling_rate = librosa.load(
            path,
            sr=self.config.sampling_rate if self.ignore_sampling_rate_error else None,
        )
        assert (
            sampling_rate == self.config.sampling_rate
        ), f"Sampling rate mismatch: {sampling_rate} != {self.config.sampling_rate}"

        audio = audio / 32768.0

        if self.normalize:
            audio = normalize(audio) * 0.95

        return torch.FloatTensor(audio)

    def __getitem__(
        self,
        index: int
        | str
        | os.PathLike
        | tuple[str | os.PathLike, str | os.PathLike | np.ndarray],
        without_slice: bool = False,
    ):
        assert self.audio_files, "Dataset path is not set."
        mel = None

        if isinstance(index, int):
            file = self.audio_files[index]
        elif isinstance(index, tuple):
            file, mel = index

            if not isinstance(mel, np.ndarray):
                mel = np.load(mel)

            mel = torch.FloatTensor(mel)
        else:
            file = Path(index)

        audio = self.load_audio(file).unsqueeze(0)

        if not without_slice:
            if audio.size(1) >= self.segment_length:
                max_audio_start = audio.size(1) - self.segment_length
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start : audio_start + self.segment_length]
            else:
                audio = torch.nn.functional.pad(
                    audio, (0, self.segment_length - audio.size(1)), "constant"
                )

        mel_loss = self.stft.mel(audio).squeeze()

        if not mel:
            mel = mel_loss.detach().clone()
        else:
            mel = mel.squeeze()

        return audio.squeeze(0), mel, mel_loss

    def __len__(self):
        assert self.audio_files, "Dataset path is not set."
        return len(self.audio_files) if self.path else 0
