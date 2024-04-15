from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from dataset import Tacotron2Batch, Tacotron2Dataset
from models import Tacotron2Loss

if TYPE_CHECKING:
    from model import Tacotron2, Tacotron2TrainState


class Tacotron2Trainer:
    model: "Tacotron2"
    vocoder: nn.Module | None = None
    state: "Tacotron2TrainState"
    train_model: nn.DataParallel
    loss: Tacotron2Loss
    optimizer: Adam

    logger: SummaryWriter | None
    verbose: bool = True

    log_interval: int = 50
    checkpoint_interval: int = 1000
    validation_interval: int = 1000

    checkpoint_path: Path | None = None
    checkpoint_format: str = "Tacotron2_{step}_{total_loss:.4f}.pt"
    best_checkpoint_format: str = "Tacotron2_best_{validation_loss:.4f}.pt"
    remove_old_best_checkpoint: bool = True
    last_best_checkpoint: Path | None = None

    sample_count: int = 4
    save_best: bool = True
    best_loss: float = float("inf")

    dataset: Tacotron2Dataset | None = None
    train_loader: DataLoader | None = None
    validation_loader: DataLoader | None = None

    def __init__(self, model: "Tacotron2", state: "Tacotron2TrainState"):
        self.model = model
        self.train_model = nn.DataParallel(model.model)
        self.loss = state.loss(model.config.frames_per_step)
        self.optimizer = state.optimizer(model.model.parameters())
        self.state = state.exclude_optimizer

    def set_summary_writer(self, logger: SummaryWriter | str | PathLike | None):
        if not logger:
            self.logger = None
        elif isinstance(logger, SummaryWriter):
            self.logger = logger
        else:
            self.logger = SummaryWriter(logger)
        return self

    def set_log_interval(self, log_interval: int):
        self.log_interval = log_interval
        return self

    def set_checkpoint_interval(self, checkpoint_interval: int):
        self.checkpoint_interval = checkpoint_interval
        return self

    def set_validation_interval(self, validation_interval: int):
        self.validation_interval = validation_interval
        return self

    def set_verbose(self, verbose: bool):
        self.verbose = verbose
        return self

    def set_vocoder(self, vocoder: nn.Module | None):
        self.vocoder = vocoder
        return self

    def set_checkpoint_path(
        self,
        checkpoint_path: str | PathLike | None,
        checkpoint_format: str | None = None,
        best_checkpoint_format: str | None = None,
        remove_old_best_checkpoint: bool | None = None,
    ):
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        if checkpoint_format:
            self.checkpoint_format = checkpoint_format
        if best_checkpoint_format:
            self.best_checkpoint_format = best_checkpoint_format
        if remove_old_best_checkpoint is not None:
            self.remove_old_best_checkpoint = remove_old_best_checkpoint
        return self

    def set_sample_count(self, sample_count: int):
        self.sample_count = sample_count
        return self

    def set_save_best(self, save_best: bool):
        self.save_best = save_best
        return self

    def __plot_mel(
        self, mel: np.ndarray, mel_gen: np.ndarray, attn: np.ndarray
    ) -> Figure:
        matplotlib.use("Agg")

        fig, axs = plt.subplots(3, 1)
        im1 = axs[0].imshow(mel, aspect="auto", origin="lower")
        axs[0].set_title("Original Mel")
        fig.colorbar(im1, ax=axs[0])

        im2 = axs[1].imshow(mel_gen, aspect="auto", origin="lower")
        axs[1].set_title("Generated Mel")
        fig.colorbar(im2, ax=axs[1])

        im3 = axs[2].imshow(attn, aspect="auto", origin="lower")
        axs[2].set_title("Attention")
        axs[2].set_xlabel("Audio timestep")
        axs[2].set_ylabel("Text timestep")
        axs[2].set_xlim(0, attn.shape[1])
        axs[2].set_ylim(0, attn.shape[0])
        fig.colorbar(im3, ax=axs[2])

        fig.tight_layout()

        return fig

    def __format_checkpoint(self, format: str, **kwargs) -> str:
        return format.format(**self.state.__dict__, **kwargs)

    def save_checkpoint(
        self,
        path: str | PathLike | None = None,
        state: "Tacotron2TrainState | None" = None,
    ):
        state = state or self.state.exclude_optimizer  # like a copy
        state.optimizer_state = self.optimizer.state_dict()

        if not path:
            assert self.checkpoint_path, "Checkpoint path is not set."
            path = self.checkpoint_path / self.__format_checkpoint(
                self.checkpoint_format
            )

        path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore
        self.model.save(
            path,
            train_state=state,
        )

    def set_dataset(
        self,
        dataset: Tacotron2Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        validation_split: float | int = 0.05,
    ):
        self.dataset = dataset
        self.train_loader, self.validation_loader = dataset.loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            validation_split=validation_split,
        )
        return self

    def set_dataset_from_path(
        self,
        path: str | PathLike,
        batch_size: int,
        shuffle: bool = True,
        exclude: set[str] | None = None,
        skip_incomplete: bool = True,
        num_workers: int = 4,
        validation_split: float = 0.05,
    ):
        self.set_dataset(
            Tacotron2Dataset(
                path=path,
                exclude=exclude,
                skip_incomplete=skip_incomplete,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            validation_split=validation_split,
        )

    def set_train_loader(self, train_loader: DataLoader):
        self.train_loader = train_loader
        return self

    def set_validation_loader(self, validation_loader: DataLoader):
        self.validation_loader = validation_loader
        return self

    @torch.no_grad()
    def validate(self, dataloader: DataLoader | None = None):
        dataloader = dataloader or self.validation_loader
        assert dataloader, "Validation loader is not set."
        assert self.dataset, "Dataset is not set."
        sampling_rate = self.dataset.config.sampling_rate
        device = self.model.device
        vocoder_device = (
            next(self.vocoder.parameters()).device if self.vocoder else device
        )
        mini_batches = len(dataloader)
        batch_size = dataloader.batch_size or 1
        model, step = self.train_model, self.state.step
        model.eval()

        if self.vocoder:
            self.vocoder.eval()

        index = 0
        if self.logger:
            avg = (mini_batches * batch_size) / (self.sample_count or 1)
            samples = np.array(
                [int(avg * i + avg / 2) for i in range(self.sample_count)]
            )
        else:
            samples = np.array([])

        losses = np.zeros(3)  # total, mel, gate, attn

        for batch in tqdm(
            dataloader,
            total=mini_batches,
            desc=f"[VALID] Epoch {self.state.epoch}",
            disable=not self.verbose,
            leave=False,
        ):
            batch: Tacotron2Batch
            batch = batch.to(device)

            mels, mels_postnet, gates, alignments = model(
                batch.tokens,
                batch.token_lengths,
                batch.mels,
                batch.mel_lengths,
                batch.embeds,
            )

            losses += np.array(
                [
                    x.item()
                    for x in self.loss(
                        batch.mels,
                        batch.gates,
                        batch.token_lengths,
                        batch.mel_lengths,
                        batch.mel_pad,
                        mels,
                        mels_postnet,
                        gates,
                        alignments,
                    )
                ]
            )

            if self.logger:
                for abs_index in samples[
                    (samples >= index) & (samples < index + len(batch.mels))
                ]:
                    i = abs_index - index
                    token_length = batch.token_lengths[i].item()
                    mel_length = batch.mel_lengths[i].item()
                    mel = batch.mels[i, :mel_length]
                    gen_mel = mels_postnet[i, :mel_length]
                    attn = alignments[
                        i,
                        : mel_length // self.model.config.frames_per_step,
                        :token_length,
                    ]

                    self.logger.add_figure(
                        f"validation/{abs_index}",
                        self.__plot_mel(
                            mel.cpu().numpy().T,
                            gen_mel.cpu().numpy().T,
                            attn.cpu().numpy().T,
                        ),
                        step,
                    )

                    if self.vocoder:
                        self.logger.add_audio(
                            f"validation/{abs_index}/reconstruction",
                            self.vocoder(mel.to(vocoder_device)),
                            0,
                            sample_rate=sampling_rate,
                        )

                        self.logger.add_audio(
                            f"validation/{abs_index}/synthesis",
                            self.vocoder(gen_mel.to(vocoder_device)),
                            step,
                            sample_rate=sampling_rate,
                        )

                index += len(batch.mels)

        losses /= mini_batches  # type: ignore
        loss_mel, loss_gate, loss_attn = losses
        loss_total = loss_mel + loss_gate + (loss_attn * 2)

        if self.logger:
            self.logger.add_scalar("validation/loss/total", loss_total, step)
            self.logger.add_scalar("validation/loss/mel", loss_mel, step)
            self.logger.add_scalar("validation/loss/gate", loss_gate, step)
            self.logger.add_scalar("validation/loss/attention", loss_attn, step)

        if self.best_loss <= loss_total:
            return

        self.best_loss = loss_total

        if not self.save_best or not self.checkpoint_path:
            return

        if self.last_best_checkpoint and self.remove_old_best_checkpoint:
            self.last_best_checkpoint.unlink()

        self.last_best_checkpoint = self.checkpoint_path / self.__format_checkpoint(
            self.best_checkpoint_format, validation_loss=loss_total
        )

        self.save_checkpoint(path=self.last_best_checkpoint)

    def train_batch(self, batch: Tacotron2Batch, bar: tqdm | None = None):
        model = self.train_model
        model.train()
        batch = batch.to(self.model.device)

        mels, mels_postnet, gates, alignments = model(
            batch.tokens,
            batch.token_lengths,
            batch.mels,
            batch.mel_lengths,
            batch.embeds,
        )

        mel_loss, gate_loss, attn_loss = self.loss(
            batch.mels,
            batch.gates,
            batch.token_lengths,
            batch.mel_lengths,
            batch.mel_pad,
            mels,
            mels_postnet,
            gates,
            alignments,
        )

        total_loss = gate_loss + attn_loss + (mel_loss * 2)
        total_loss.backward()
        
        grad_norm = nn.utils.clip_grad_norm_(  # type: ignore
            model.parameters(), 1.0
        )

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update state
        self.state.step += 1
        step = self.state.step
        total_loss = self.state.total_loss = total_loss.item()
        mel_loss = self.state.mel_loss = mel_loss.item()
        gate_loss = self.state.gate_loss = gate_loss.item()
        attn_loss = self.state.attn_loss = attn_loss.item()

        if step % self.log_interval == 0:
            if bar:
                bar.set_description(
                    f"Step: {step: 8d}, "
                    f"Total Loss: {total_loss:.4f}, "
                    f"Mel Loss: {mel_loss:.4f}, "
                    f"Gate Loss: {gate_loss:.4f}, "
                    f"Attn Loss: {attn_loss:.4f}"
                )

            if self.logger:

                self.logger.add_scalar("train/loss/total", total_loss, step)
                self.logger.add_scalar("train/loss/mel", mel_loss, step)
                self.logger.add_scalar("train/loss/gate", gate_loss, step)
                self.logger.add_scalar("train/loss/attention", attn_loss, step)
                self.logger.add_scalar("train/grad_norm", grad_norm, step)

        # Save checkpoint
        if step % self.checkpoint_interval == 0 and self.checkpoint_path:
            self.save_checkpoint()

        # Validation
        if step % self.validation_interval == 0 and self.validation_loader:
            self.validate()

        return total_loss, mel_loss, gate_loss, attn_loss

    def train_epoch(
        self,
        train_loader: DataLoader | None = None,
        outside_bar: tqdm | None = None,
    ):
        train_loader = train_loader or self.train_loader
        assert train_loader, "Train loader is not set."
        mini_batches = len(train_loader)
        losses = np.zeros(4)  # total, mel, gate, attn

        # Update state
        self.state.epoch += 1

        for batch in tqdm(
            train_loader,
            total=mini_batches,
            desc=f"[TRAIN] Epoch {self.state.epoch}",
            disable=not self.verbose,
            leave=False,
        ):
            batch: Tacotron2Batch
            losses += np.array(self.train_batch(batch, bar=outside_bar))

            if outside_bar:
                outside_bar.update()

        return losses / mini_batches

    def train(self, epochs: int, early_stop: int = 100):
        assert self.train_loader, "Train loader is not set."
        assert self.validation_loader, "Validation loader is not set."

        best_loss = float("inf")
        best_epoch = self.state.epoch

        with tqdm(
            range(self.state.epoch, self.state.epoch + epochs),
            total=epochs * len(self.train_loader),
            desc="Step",
            position=0,
            leave=False,
            disable=not self.verbose,
        ) as outer_bar:
            outer_bar.n = self.state.step

            while self.state.epoch <= epochs:
                total_loss, mel_loss, gate_loss, attn_loss = self.train_epoch(
                    outside_bar=outer_bar
                )

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_epoch = self.state.epoch
                    continue

                if self.state.epoch - best_epoch > early_stop:
                    if self.verbose:
                        print(
                            f"Loss did not improve for {early_stop} epochs. Stopping."
                        )

                    break
