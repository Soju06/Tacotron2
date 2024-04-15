from dataclasses import dataclass, field
from typing import Any, Literal

from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from .stft import STFT


@dataclass
class HiFiGANAudioConfig:
    sampling_rate: int = 22050

    mel_fmin: int = 0
    mel_fmax: int = 8000

    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024

    def stft(self) -> STFT:
        return STFT(
            num_mels=80,
            n_fft=self.filter_length,
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_length,
            win_size=self.win_length,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
        )


@dataclass
class HiFiGANTrainState:
    step: int = 0
    epoch: int = 0
    generator_loss: float = 0.0
    discriminator_loss: float = 0.0
    mel_loss: float = 0.0
    total_loss: float = 0.0
    learning_rate: float = 0.001
    learning_rate_decay: float = 0.999
    adam_betas: tuple[float, float] = field(default_factory=lambda: (0.5, 0.9))
    optim_gen: Any | None = None
    optim_dis: Any | None = None

    def optimizer(
        self, params: Any, restore_state: Literal["optim_gen", "optim_dis"]
    ) -> AdamW:
        optim = AdamW(
            params,
            lr=self.learning_rate,
            betas=self.adam_betas,
        )
        state = getattr(self, restore_state)

        if state:
            optim.load_state_dict(state)

        return optim

    def scheduler(self, optimizer: AdamW) -> ExponentialLR:
        return ExponentialLR(
            optimizer,
            gamma=self.learning_rate_decay,
            last_epoch=self.epoch - 1,
        )

    @property
    def exclude_optimizer(self) -> "HiFiGANTrainState":
        return HiFiGANTrainState(
            step=self.step,
            epoch=self.epoch,
            generator_loss=self.generator_loss,
            discriminator_loss=self.discriminator_loss,
            mel_loss=self.mel_loss,
            total_loss=self.total_loss,
            learning_rate=self.learning_rate,
            learning_rate_decay=self.learning_rate_decay,
            adam_betas=self.adam_betas,
        )


@dataclass
class HiFiGANModelState:
    gen: Any
    mpd: Any | None
    msd: Any | None
