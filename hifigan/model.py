import copy
from dataclasses import asdict, dataclass, field
from os import PathLike
from typing import Literal
import numpy as np
import torch
import torch.nn as nn

from .models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from .config import *
from .trainer import HiFiGANTrainer


@dataclass
class HiFiGANConfig:
    resblock: Literal["1", "2"] = "1"
    upsample_rates: list[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 4, 4])
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )

    mpd_periods: list[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    mpd_convs: list[int] = field(default_factory=lambda: [1, 32, 128, 512, 1024])
    mpd_kernel_size: int = 5
    mpd_stride: int = 3

    def generator(self) -> Generator:
        return Generator(
            num_mels=80,
            upsample_rates=self.upsample_rates,
            upsample_kernel_sizes=self.upsample_kernel_sizes,
            upsample_initial_channel=self.upsample_initial_channel,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            resblock_dilation_sizes=self.resblock_dilation_sizes,
        )

    def multi_period_discriminator(self):
        return MultiPeriodDiscriminator(
            periods=self.mpd_periods,
            convs=self.mpd_convs,
            kernel_size=self.mpd_kernel_size,
            stride=self.mpd_stride,
        )

    def multi_scale_discriminator(self):
        return MultiScaleDiscriminator()


@dataclass
class HiFiGANState:
    config: HiFiGANConfig
    model: HiFiGANModelState
    train: HiFiGANTrainState = field(default_factory=HiFiGANTrainState)
    mode: Literal["eval", "train"] = "train"

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = HiFiGANConfig(**self.config)
        if isinstance(self.model, dict):
            self.model = HiFiGANModelState(**self.model)  # type: ignore
        if isinstance(self.train, dict):
            self.train = HiFiGANTrainState(**self.train)

    def load_model(self) -> "HiFiGAN":
        return HiFiGAN(
            config=self.config,
            mode=self.mode,
            model_state=self.model,
            train_state=self.train,
        )


class HiFiGAN(nn.Module):
    gen: Generator
    mpd: MultiPeriodDiscriminator | None
    msd: MultiScaleDiscriminator | None

    config: HiFiGANConfig
    train_state: HiFiGANTrainState

    def __init__(
        self,
        config: HiFiGANConfig | None = None,
        mode: Literal["eval", "train"] = "train",
        model_state: HiFiGANModelState | None = None,
        train_state: HiFiGANTrainState | None = None,
    ):
        super(HiFiGAN, self).__init__()
        self.config = config or HiFiGANConfig()
        self.train_state = train_state or HiFiGANTrainState()
        self.gen = self.config.generator()
        self.mpd = None
        self.msd = None

        if mode == "train":
            self.mpd = self.config.multi_period_discriminator()
            self.msd = self.config.multi_scale_discriminator()

        if model_state:
            self.load_state(model_state)

        if mode == "eval":
            self.eval()
            self.gen.remove_weight_norm()

    def load_state(self, model_state: HiFiGANModelState):
        self.gen.load_state_dict(model_state.gen)
        if self.mpd and model_state.mpd:
            self.mpd.load_state_dict(model_state.mpd)
        if self.msd and model_state.msd:
            self.msd.load_state_dict(model_state.msd)

    @property
    def mode(self) -> Literal["eval", "train"]:
        return "train" if self.mpd else "eval"

    @property
    def device(self) -> torch.device:
        return next(self.gen.parameters()).device

    def to_eval(self) -> "HiFiGAN":
        """
        Fork from 'train' mode to 'eval' mode.
        """
        assert self.mode == "train", "Model is in eval mode."

        return HiFiGAN(
            config=copy.deepcopy(self.config),
            model_state=HiFiGANModelState(
                gen=self.gen.state_dict(),
                mpd=None,
                msd=None,
            ),
            mode="eval",
            train_state=self.train_state.exclude_optimizer,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> np.ndarray:
        single = len(x.shape) == 2
        if single:
            x = x.unsqueeze(0)  # [1, X, Y]
        if x.shape[1] != self.gen.num_mels:
            x = x.transpose(1, 2)  # [1, mels, X]

        if x.device != self.device:
            x = x.to(self.device)

        wavs = self.gen(x)

        if wavs.device != self.device:
            wavs = wavs.to(self.device)

        if single:
            wavs = wavs.squeeze(0)

        return wavs

    def trainer(
        self,
        audio: HiFiGANAudioConfig | None = None,
        train_state: HiFiGANTrainState | None = None,
    ) -> "HiFiGANTrainer":
        assert self.mode == "train", "Model is not in train mode."
        return HiFiGANTrainer(
            model=self,
            state=train_state or self.train_state,
            audio=audio or HiFiGANAudioConfig(),
        )

    def __iter__(
        self,
    ):
        return iter([self.gen, self.mpd, self.msd])

    def __repr__(self) -> str:
        return f"HiFiGAN(mode={self.mode}, config={self.config}, train_state={self.train_state})"

    def save(
        self,
        path: str | PathLike,
        mode: Literal["eval", "train"] = "eval",
        train_state: HiFiGANTrainState | None = None,
    ):
        assert (
            self.mode == "train" or mode == self.mode
        ), "Model is in eval mode, but you are trying to save it in train mode."
        train_state = train_state or self.train_state

        torch.save(
            asdict(
                HiFiGANState(
                    config=self.config,
                    model=HiFiGANModelState(
                        gen=self.gen.state_dict(),
                        mpd=self.mpd.state_dict() if mode == "train" else None,  # type: ignore
                        msd=self.msd.state_dict()
                        if mode == "train" and self.msd
                        else None,
                    ),
                    train=train_state
                    if mode == "train"
                    else train_state.exclude_optimizer,
                    mode=mode,
                )
            ),
            path,
        )

    @staticmethod
    def load(
        path: str | PathLike,
        mode: Literal["eval", "train"] = "eval",
        device: torch.device | str | None = None,
    ) -> "HiFiGAN":
        state = HiFiGANState(**torch.load(path))
        assert (
            state.mode == "train" or mode == state.mode
        ), "Model is saved in eval mode, but you are trying to load it in train mode."

        model = state.load_model()

        if device:
            model = model.to(device)

        return model
