from dataclasses import dataclass, field
from typing import Any

from torch.optim import Adam

from models import Tacotron2Loss, Tacotron2Model
from tokenizer import TextTokenizer


@dataclass
class Tacotron2AudioConfig:
    sampling_rate: int = 22050

    trim_top_db: int = 15
    filter_length: int = 1024
    hop_length: int = 256
    silence_audio_size: int = 5

    win_length: int = 1024
    mel_fmin: int = 0
    mel_fmax: int = 8000


@dataclass
class Tacotron2Config:
    tokenizer: str | TextTokenizer = "english"
    mask_padding: bool = True

    speaker_embeddint_dim: int = 512
    symbols_embedding_dim: int = 512

    attention_rnn_dim: int = 1024
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31
    attention_dropout: float = 0.1

    encoder_embedding_dim: int = 512
    encoder_n_convolutions: int = 3
    encoder_kernel_size: int = 5

    decoder_rnn_dim: int = 1024
    decoder_dropout: float = 0.1
    max_decoder_steps: int = 1000

    prenet_dim: int = 256
    postnet_embedding_dim: int = 512
    postnet_kernel_size: int = 5
    postnet_n_convolutions: int = 5

    frames_per_step: int = 2  # any of reduction factor is supported
    gate_threshold: float = 0.5

    def get_tokenizer(self) -> TextTokenizer:
        if isinstance(self.tokenizer, str):
            return TextTokenizer.load(self.tokenizer)

        return self.tokenizer

    def model(self, tokenizer: TextTokenizer | None = None) -> Tacotron2Model:
        if tokenizer is None:
            tokenizer = self.get_tokenizer()

        return Tacotron2Model(
            n_mel_channels=80,
            n_symbols=len(tokenizer),
            mask_padding=self.mask_padding,
            symbols_embedding_dim=self.symbols_embedding_dim,
            encoder_n_convolutions=self.encoder_n_convolutions,
            encoder_embedding_dim=self.encoder_embedding_dim,
            encoder_kernel_size=self.encoder_kernel_size,
            n_frames_per_step=self.frames_per_step,
            attention_rnn_dim=self.attention_rnn_dim,
            decoder_rnn_dim=self.decoder_rnn_dim,
            prenet_dim=self.prenet_dim,
            max_decoder_steps=self.max_decoder_steps,
            gate_threshold=self.gate_threshold,
            attention_dropout=self.attention_dropout,
            decoder_dropout=self.decoder_dropout,
            attention_dim=self.attention_dim,
            attention_location_n_filters=self.attention_location_n_filters,
            attention_location_kernel_size=self.attention_location_kernel_size,
            postnet_embedding_dim=self.postnet_embedding_dim,
            postnet_kernel_size=self.postnet_kernel_size,
            postnet_n_convolutions=self.postnet_n_convolutions,
            speaker_embeddint_dim=self.speaker_embeddint_dim,
        )


@dataclass
class Tacotron2TrainState:
    step: int = 0
    epoch: int = 0

    # loss
    guided_sigma: float = 0.4
    guided_lambda: float = 1.0

    total_loss: float = 0.0
    mel_loss: float = 0.0
    gate_loss: float = 0.0
    attn_loss: float = 0.0

    # optimizer
    adam_betas: tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))
    eps: float = 0.00000001
    weight_decay: float = 0.000001
    learning_rate: float = 0.001

    optimizer_state: Any | None = None

    def loss(self, frames_per_step: int) -> Tacotron2Loss:
        return Tacotron2Loss(
            n_frames_per_step=frames_per_step,
            guided_sigma=self.guided_sigma,
            guided_lambda=self.guided_lambda,
        )

    def optimizer(self, params: Any) -> Adam:
        optim = Adam(
            params,
            lr=self.learning_rate,
            betas=self.adam_betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        for param_group in optim.param_groups:
            param_group["lr"] = self.learning_rate

        if self.optimizer_state:
            optim.load_state_dict(self.optimizer_state)

        return optim

    @property
    def exclude_optimizer(self) -> "Tacotron2TrainState":
        return Tacotron2TrainState(
            step=self.step,
            epoch=self.epoch,
            guided_sigma=self.guided_sigma,
            guided_lambda=self.guided_lambda,
            total_loss=self.total_loss,
            mel_loss=self.mel_loss,
            gate_loss=self.gate_loss,
            attn_loss=self.attn_loss,
            adam_betas=self.adam_betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
        )
