from itertools import chain
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .dataset import HiFiGANDataset

if TYPE_CHECKING:
    from .model import HiFiGAN, HiFiGANTrainState, HiFiGANAudioConfig

from .models import discriminator_loss, feature_loss, generator_loss
from .stft import STFT


class HiFiGANTrainer:
    model: "HiFiGAN"
    state: "HiFiGANTrainState"
    audio: "HiFiGANAudioConfig"
    optimizer_gen: AdamW
    optimizer_dis: AdamW
    scheduler_gen: ExponentialLR
    scheduler_dis: ExponentialLR
    stft: STFT

    logger: SummaryWriter | None
    verbose: bool = True

    log_interval: int = 100
    checkpoint_interval: int = 1000
    validation_interval: int = 1000

    checkpoint_path: Path | None = None
    checkpoint_format: str = "HiFiGAN_{step}_{mel_loss:.4f}.pt"
    best_checkpoint_format: str = "HiFiGAN_best_{validation_loss:.4f}.pt"
    remove_old_best_checkpoint: bool = True
    last_best_checkpoint: Path | None = None

    sample_count: int = 4
    save_best: bool = True
    best_loss: float = float("inf")

    dataset: Dataset | None = None
    train_loader: DataLoader | None = None
    validation_loader: DataLoader | None = None

    def __init__(self, model: "HiFiGAN", state: "HiFiGANTrainState", audio: "HiFiGANAudioConfig"):
        assert (
            model.mode == "train" and model.msd and model.mpd
        ), "Model must be in train mode and must have discriminator."
        self.model = model
        self.optimizer_gen = state.optimizer(model.gen.parameters(), "optim_gen")
        self.optimizer_dis = state.optimizer(
            chain(model.mpd.parameters(), model.msd.parameters()), "optim_dis"
        )
        self.scheduler_gen = state.scheduler(self.optimizer_gen)
        self.scheduler_dis = state.scheduler(self.optimizer_dis)
        self.audio = audio
        self.stft = audio.stft().to(model.device)
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

    def set_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        validation_split: float | int = 0.05,
    ):
        assert hasattr(dataset, "__len__"), "Dataset must have __len__ method."
        self.dataset = dataset
        total_size = len(dataset)  # type: ignore
        validation_size = (
            int(total_size * validation_split) if validation_split < 1 else validation_split
        )
        train_dataset, validation_dataset = random_split(
            dataset, [total_size - validation_size, validation_size]
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return self

    def set_dataset_from_path(
        self,
        path: str | PathLike,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        validation_split: float | int = 0.05,
        segment_length: int = 8192,
        pattern: str = "*.wav",
        normalize: bool = True,
        ignore_sampling_rate_error: bool = False,
    ):
        self.set_dataset(
            HiFiGANDataset(
                path=path,
                config=self.audio,
                segment_length=segment_length,
                pattern=pattern,
                normalize=normalize,
                ignore_sampling_rate_error=ignore_sampling_rate_error,
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

    def __format_checkpoint(self, format: str, **kwargs) -> str:
        return format.format(**self.state.__dict__, **kwargs)

    def save_checkpoint(
        self,
        path: str | PathLike | None = None,
        state: "HiFiGANTrainState | None" = None,
    ):
        state = state or self.state.exclude_optimizer  # like a copy
        state.optim_gen = self.optimizer_gen.state_dict()
        state.optim_dis = self.optimizer_dis.state_dict()

        if not path:
            assert self.checkpoint_path, "Checkpoint path is not set."
            path = self.checkpoint_path / self.__format_checkpoint(self.checkpoint_format)

        path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore
        self.model.save(
            path,
            mode="train",
            train_state=state,
        )

    @torch.no_grad()
    def validate(self, dataloader: DataLoader | None = None):
        gen, step, stft = self.model.gen, self.state.step, self.stft
        dataloader = dataloader or self.validation_loader
        assert dataloader, "Validation loader is not set."
        device = self.model.device
        gen.eval()

        dataset = dataloader.dataset

        while isinstance(dataset, Subset):
            dataset = dataset.dataset

        sample_fn = dataset.__getitem__
        if "without_slice" in sample_fn.__code__.co_varnames:
            sample_fn = lambda x: dataset.__getitem__(x, without_slice=True)  # type: ignore

        assert hasattr(dataset, "__len__"), "Dataset must have __len__ method."
        total_size = len(dataset)  # type: ignore

        loss_total = 0
        for _, mel, original_mel in tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"[VALID] Epoch {self.state.epoch}",
            disable=not self.verbose,
            leave=False,
        ):
            original_mel = Variable(original_mel.to(device, non_blocking=True))
            gen_audio = gen(mel.to(device))
            mel = stft.mel(gen_audio.squeeze(1))
            loss_total += functional.l1_loss(original_mel, mel).item()

        loss_total /= len(dataloader)

        if self.logger:
            self.logger.add_scalar("validation/loss/mel", loss_total, step)

            if self.sample_count:
                avg = total_size / self.sample_count

                for i in (int(avg * i + avg / 2) for i in range(self.sample_count)):
                    audio, mel, mel_loss = sample_fn(i)
                    gen_audio = gen(mel.to(device))
                    gen_mel = stft.mel(gen_audio).squeeze()

                    self.logger.add_audio(
                        f"validation/{i}",
                        audio,
                        0,
                        self.audio.sampling_rate,
                    )
                    self.logger.add_figure(
                        f"validation/{i}/mel",
                        stft.plot_mel(mel_loss),
                        0,
                    )

                    self.logger.add_audio(
                        f"validation/{i}/gen",
                        gen_audio[0],
                        step,
                        self.audio.sampling_rate,
                    )
                    self.logger.add_figure(
                        f"validation/{i}/mel/gen",
                        stft.plot_mel(gen_mel),
                        step,
                    )

        if self.best_loss <= loss_total:
            return

        self.best_loss = loss_total

        if not self.save_best or not self.checkpoint_path:
            return

        if self.last_best_checkpoint and self.remove_old_best_checkpoint:
            self.last_best_checkpoint.unlink(missing_ok=True)

        self.last_best_checkpoint = self.checkpoint_path / self.__format_checkpoint(
            self.best_checkpoint_format,
            validation_loss=loss_total,
        )
        self.save_checkpoint(
            path=self.last_best_checkpoint,
        )

    def train_batch(
        self,
        audio: torch.Tensor,
        mel: torch.Tensor,
        original_mel: torch.Tensor,
        bar: tqdm | None = None,
    ) -> tuple[float, float, float]:
        device = self.model.device
        gen, mpd, msd = self.model
        assert gen and mpd and msd, "Model must have generator and discriminator."
        optim_gen = self.optimizer_gen
        optim_dis = self.optimizer_dis
        gen.train()
        mpd.train()
        msd.train()

        mel, audio, original_mel = (
            Variable(mel.to(device, non_blocking=True)),
            Variable(audio.to(device, non_blocking=True)).unsqueeze(1),
            Variable(original_mel.to(device, non_blocking=True)),
        )
        gen_audio = gen(mel)

        # Discriminator
        optim_dis.zero_grad()
        dis_loss_f = discriminator_loss(*mpd(audio, gen_audio.detach())[:2])  # r, g
        dis_loss_s = discriminator_loss(*msd(audio, gen_audio.detach())[:2])  # r, g
        dis_loss = dis_loss_f + dis_loss_s
        dis_loss.backward()
        optim_dis.step()

        # Generator
        optim_gen.zero_grad()
        _, gen_f, gen_fr, gen_fg = mpd(audio, gen_audio)
        _, gen_s, gen_sr, gen_sg = msd(audio, gen_audio)

        mel_loss = functional.l1_loss(original_mel, self.stft.mel(gen_audio.squeeze(1)))
        gen_loss = (
            generator_loss(gen_f)
            + generator_loss(gen_s)
            + feature_loss(gen_fr, gen_fg)
            + feature_loss(gen_sr, gen_sg)
            + (mel_loss * 45)
        )
        gen_loss.backward()
        optim_gen.step()

        # Update state
        self.state.step += 1
        step = self.state.step
        gen_loss = self.state.generator_loss = gen_loss.item()
        dis_loss = self.state.discriminator_loss = dis_loss.item()
        mel_loss = self.state.mel_loss = mel_loss.item()

        # Log
        if step % self.log_interval == 0:
            if bar:
                bar.set_description(
                    f"Step: {step: 8d}, "
                    f"Dis Loss: {dis_loss:.5f}, "
                    f"Gen Loss: {gen_loss:.5f}, "
                    f"Mel Loss: {mel_loss:.5f}"
                )

            if self.logger:
                self.logger.add_scalar("train/loss/discriminator", dis_loss, step)
                self.logger.add_scalar("train/loss/generator", gen_loss, step)
                self.logger.add_scalar("train/loss/mel", mel_loss, step)

        # Save checkpoint
        if step % self.checkpoint_interval == 0 and self.checkpoint_path:
            self.save_checkpoint()

        # Validation
        if step % self.validation_interval == 0 and self.validation_loader:
            self.validate()

            if self.logger:
                self.logger.flush()

        return dis_loss, gen_loss, mel_loss

    def train_epoch(
        self,
        train_loader: DataLoader | None = None,
        outside_bar: tqdm | None = None,
    ):
        train_loader = train_loader or self.train_loader
        assert train_loader, "Train loader is not set."
        loss_total = np.array([0.0, 0.0, 0.0])

        # Update state
        self.state.epoch += 1

        for batch in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"[TRAIN] Epoch {self.state.epoch}",
            disable=not self.verbose,
            leave=False,
        ):
            loss_total += np.array(self.train_batch(*batch, bar=outside_bar))

            if outside_bar:
                outside_bar.update()

        self.scheduler_gen.step()
        self.scheduler_dis.step()

        return (loss_total / len(train_loader)).tolist()

    def train(self, epochs: int, early_stop: int = 100):
        assert self.train_loader, "Train loader is not set."
        assert self.validation_loader, "Validation loader is not set."

        if self.state.epoch > epochs:
            epochs += self.state.epoch

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
                dis_loss, gen_loss, mel_loss = self.train_epoch(outside_bar=outer_bar)

                if gen_loss < best_loss:
                    best_loss = gen_loss
                    best_epoch = self.state.epoch
                    continue

                if self.state.epoch - best_epoch > early_stop:
                    if self.verbose:
                        print(f"Loss did not improve for {early_stop} epochs. Stopping.")

                    break
