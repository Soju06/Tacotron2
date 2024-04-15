from dataclasses import asdict, dataclass
from os import PathLike
from typing import Any
import numpy as np

import torch
import torch.nn as nn

from config import Tacotron2Config, Tacotron2TrainState
from models import Tacotron2Model
from tokenizer import TextTokenizer
from trainer import Tacotron2Trainer


@dataclass
class Tacotron2State:
    config: Tacotron2Config
    train: Tacotron2TrainState
    model: Any

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = Tacotron2Config(**self.config)
        if isinstance(self.train, dict):
            self.train = Tacotron2TrainState(**self.train)

    def load_model(self) -> "Tacotron2":
        return Tacotron2(
            config=self.config,
            model_state=self.model,
            train_state=self.train,
        )


class Tacotron2(nn.Module):
    model: Tacotron2Model
    config: Tacotron2Config
    tokenizer: TextTokenizer
    train_state: Tacotron2TrainState

    def __init__(
        self,
        config: Tacotron2Config | None = None,
        model_state: Any = None,
        train_state: Tacotron2TrainState | None = None,
    ):
        super(Tacotron2, self).__init__()
        self.config = config or Tacotron2Config()
        self.train_state = train_state or Tacotron2TrainState()
        self.tokenizer = self.config.get_tokenizer()
        self.model = self.config.model(tokenizer=self.tokenizer)

        if model_state:
            self.model.load_state_dict(model_state)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        speakers: list[np.ndarray] | list[torch.Tensor] | np.ndarray | torch.Tensor,
        inputs: str | list[str] | np.ndarray | torch.Tensor,
        vocoder: nn.Module | None = None,
        include_attention: bool = False,
    ):
        self.model.eval()
        batch = not isinstance(inputs, str)

        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(inputs, list):
            inputs = np.array([self.tokenizer.tokenize(input) for input in inputs])
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).long().to(self.device)

        if isinstance(speakers, list):
            speakers = np.array(speakers)
        if isinstance(speakers, np.ndarray):
            speakers = torch.from_numpy(speakers).float().to(self.device)

        input_lengths = np.array([len(input) for input in inputs])
        _, mels_postnet, _, alignments = self.model.inference(
            texts=inputs,
            max_src_len=max(input_lengths),
            spker_embeds=speakers,
        )

        attns = (
            [
                alignments[
                    i,
                    : mel.shape[0],
                    :token_length,
                ]
                for i, (mel, token_length) in enumerate(zip(mels_postnet, input_lengths))
            ]
            if include_attention
            else None
        )

        audios = vocoder(mels_postnet) if vocoder else None

        if not batch:
            mels_postnet = mels_postnet[0]
            attns = attns[0] if attns else None

        return mels_postnet, audios, attns

    def save(
        self,
        path: str | PathLike,
        train_state: Tacotron2TrainState | None = None,
    ):
        train_state = train_state or self.train_state
        torch.save(
            asdict(
                Tacotron2State(
                    config=self.config,
                    train=train_state,
                    model=self.model.state_dict(),
                )
            ),
            path,
        )

    def trainer(
        self,
        state: Tacotron2TrainState | None = None,
    ):
        state = state or self.train_state
        return Tacotron2Trainer(self, state)

    @staticmethod
    def load(
        path: str | PathLike,
        device: torch.device | str | None = None,
    ) -> "Tacotron2":
        state = Tacotron2State(**torch.load(path))
        model = state.load_model()

        if device:
            model = model.to(device)

        return model
