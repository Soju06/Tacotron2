from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import librosa
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile  # type: ignore

from config import Tacotron2AudioConfig
from stft import TacotronSTFT
from tokenizer import TextTokenizer

if TYPE_CHECKING:
    from deepspeaker import DeepSpeakerEmbedder


class Tacotron2Sample(NamedTuple):
    audio: np.ndarray | None
    mel: np.ndarray
    embed: np.ndarray | None
    tokens: np.ndarray
    texts: np.ndarray

    def duration(self, hop_length: int, sampling_rate: int) -> float:
        return self.mel.shape[1] * hop_length / sampling_rate

    def decode(self, index: int = 0, encoding: str = "utf-8") -> str:
        return self.texts[index].tobytes().decode(encoding)

    def decode_all(self, encoding: str = "utf-8") -> list[str]:
        return [text.tobytes().decode(encoding) for text in self.texts]

    def save(
        self,
        path: str | PathLike,
        audio: bool = False,
        embed: bool = False,
        compress: bool = False,
    ):
        (np.savez_compressed if compress else np.savez)(
            Path(path),
            mel=self.mel,
            tokens=self.tokens,
            texts=self.texts,
            **({"audio": self.audio} if audio else {}),
            **({"embed": self.embed} if embed else {}),
        )

    def save_test(self, path: str | PathLike, sampling_rate: int):
        path = Path(path)

        # Plot mel
        matplotlib.use("Agg")
        fig, axe = plt.subplots()
        im1 = axe.imshow(self.mel.T, aspect="auto", origin="lower")
        axe.set_title("Audio Mel")
        fig.colorbar(im1, ax=axe)
        fig.tight_layout()
        fig.savefig(str(path.with_suffix(".png")))

        # Save audio
        if self.audio is not None:
            wavfile.write(
                str(path.with_suffix(".wav")),
                sampling_rate,
                (self.audio * 32767.0).astype(np.int16),
            )

    @staticmethod
    def load(path: str | PathLike):
        data = np.load(path, allow_pickle=True)

        return Tacotron2Sample(
            audio=data["audio"] if "audio" in data else None,
            mel=data["mel"],
            embed=data["embed"] if "embed" in data else None,
            tokens=data["tokens"],
            texts=data["texts"],
        )


class Tacotron2Preprocessor:
    config: Tacotron2AudioConfig
    tokenizer: TextTokenizer
    stft: TacotronSTFT
    embedder: "DeepSpeakerEmbedder"

    def __init__(
        self,
        tokenizer: TextTokenizer | str,
        config: Tacotron2AudioConfig | None = None,
        embedder: "DeepSpeakerEmbedder | str | PathLike | None" = None,
    ):
        from deepspeaker import DeepSpeakerEmbedder

        self.config = config or Tacotron2AudioConfig()
        self.tokenizer = TextTokenizer.load(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.stft = TacotronSTFT(
            filter_length=self.config.filter_length,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            sampling_rate=self.config.sampling_rate,
            mel_fmin=self.config.mel_fmin,
            mel_fmax=self.config.mel_fmax,
            n_mel_channels=80,
        )

        if isinstance(embedder, DeepSpeakerEmbedder):
            self.embedder = embedder
        else:
            self.embedder = DeepSpeakerEmbedder(
                embedder,
                sampling_rate=self.config.sampling_rate,
                win_length=self.config.win_length,
            )

        assert self.embedder.sampling_rate == self.config.sampling_rate, (
            "Audio sampling rate must match DeepSpeaker sampling rate "
            f"({self.embedder.sampling_rate} != {self.config.sampling_rate})"
        )
        assert self.embedder.win_length == self.config.win_length, (
            "Audio win length must match DeepSpeaker win length "
            f"({self.embedder.win_length} != {self.config.win_length})"
        )

    def process(
        self,
        labels: str | list[str],
        audio_path: str | PathLike,
        ensure_sampling_rate: bool = False,
    ):
        audio_path = Path(audio_path)
        assert audio_path.exists(), f"{audio_path} does not exist"

        # Tokenize label
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(labels, list):
            tokens = [np.array(self.tokenizer.tokenize(text)) for text in labels]

        tokens = np.array(tokens, dtype=object)
        audio, sampling_rate = librosa.load(
            audio_path, sr=self.config.sampling_rate if ensure_sampling_rate else None
        )

        assert sampling_rate == self.config.sampling_rate, (
            "Audio sampling rate must match Tacotron2Preprocessor sampling rate "
            f"({sampling_rate} != {self.config.sampling_rate})"
        )

        # Audio normalization (0.9999 instead of 1 to avoid clipping)
        audio = audio / np.abs(audio).max() * 0.9999

        # Trim silence
        audio = librosa.effects.trim(
            audio,
            top_db=self.config.trim_top_db,
            frame_length=self.config.filter_length,
            hop_length=self.config.hop_length,
        )[0]

        # Padding
        audio = np.append(audio, [0.0] * self.config.hop_length * self.config.silence_audio_size)
        audio = audio.astype(np.float32)

        # Mel spectrogram
        mel: np.ndarray = self.stft.wav2mel(audio).T

        # Text to array
        texts = np.array([np.frombuffer(label.encode("utf-8"), dtype=np.uint8) for label in labels], dtype=object)

        if texts[0].dtype != np.uint8:
            texts = texts.astype(np.uint8)

        # Speaker embedding
        embed = self.embedder(audio)

        return Tacotron2Sample(
            audio=audio,
            mel=mel,
            embed=embed,
            tokens=tokens,
            texts=texts,
        )
