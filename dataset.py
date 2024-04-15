import random
import shutil
import string
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from config import Tacotron2AudioConfig
from preprocess import Tacotron2Preprocessor, Tacotron2Sample

RANDOM_STRINGS = string.ascii_lowercase + string.digits


@dataclass
class Tacotron2Batch:
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    texts: list[str]
    mels: torch.Tensor
    mel_lengths: torch.Tensor
    mel_max_length: int
    mel_pad: int
    gates: torch.Tensor
    embeds: torch.Tensor

    def to(self, device: torch.device | str) -> "Tacotron2Batch":
        self.tokens = self.tokens.to(device)
        self.token_lengths = self.token_lengths.to(device)
        self.mels = self.mels.to(device)
        self.mel_lengths = self.mel_lengths.to(device)
        self.gates = self.gates.to(device)
        self.embeds = self.embeds.to(device)

        return self

    @classmethod
    def make_batch(
        cls,
        items: Generator[tuple[np.ndarray, np.ndarray, str, np.ndarray], None, None],
        frames_per_step: int = 2,
        mel_stats: tuple[float, float] | None = None,
    ):
        mels: list[np.ndarray]
        tokens: list[np.ndarray]
        texts: list[str]
        embeds: list[np.ndarray]
        mels, tokens, texts, embeds = zip(*items)  # type: ignore

        token_lengths = np.array([len(token) for token in tokens])
        mel_lengths = np.array([mel.shape[0] for mel in mels])
        token_max_length = int(max(token_lengths))
        mel_max_length = int(max(mel_lengths))

        mel_pad = mel_max_length % frames_per_step

        if mel_pad != 0:
            mel_max_length += frames_per_step - mel_pad
            assert mel_max_length % frames_per_step == 0

        gates = np.zeros([mel_lengths.shape[0], mel_max_length])

        for i, mel_len in enumerate(mel_lengths):
            gates[i, mel_len - 1 :] = 1

        mels = torch.from_numpy(
            np.stack(
                [
                    np.pad(  # type: ignore
                        x,
                        (0, mel_max_length - np.shape(x)[0]),  # type: ignore
                        mode="constant",
                        constant_values=0,
                    )[
                        :, : np.shape(x)[1]  # type: ignore
                    ]
                    for x in mels
                ]
            )
        ).float()

        if mel_stats:
            mel_min, mel_max = mel_stats
            mels = 2 * (mels - mel_min) / (mel_max - mel_min) - 1  # type: ignore

        return Tacotron2Batch(
            tokens=torch.from_numpy(
                np.stack(
                    [
                        np.pad(  # type: ignore
                            x,
                            (0, token_max_length - x.shape[0]),
                            mode="constant",
                            constant_values=0,
                        )
                        for x in tokens
                    ]
                )
            ).long(),
            token_lengths=torch.from_numpy(token_lengths),
            texts=texts,
            mels=mels,  # type: ignore
            mel_lengths=torch.from_numpy(mel_lengths),
            mel_max_length=mel_max_length,
            mel_pad=mel_pad,
            gates=torch.from_numpy(gates).float(),
            embeds=torch.from_numpy(np.concatenate(embeds, axis=0)).float(),
        )


@dataclass
class Tacotron2SpeakerEmbeds:
    count: int = 0
    embed: np.ndarray | None = None

    def __post_init__(self):
        self.count = int(self.count)

    def update(self, embed: np.ndarray):
        if self.embed is None:
            self.embed = embed
        else:
            assert (
                self.embed.shape == embed.shape
            ), f"Embed shape must match ({self.embed.shape} != {embed.shape})"

            self.count += 1
            self.embed += (embed - self.embed) / (self.count)

    def __call__(self) -> np.ndarray:
        assert self.embed is not None, "No embeds"
        return self.embed


class Tacotron2SpeakerDataset:
    path: Path
    config: Tacotron2AudioConfig | None
    audios: list[str]
    embeds: Tacotron2SpeakerEmbeds

    def __init__(
        self,
        path: str | PathLike,
        config: Tacotron2AudioConfig | None = None,
        ensure_config: bool = False,
    ):
        self.path = Path(path)
        embeds_path = self.embeds_path
        config_path = self.config_path

        if ensure_config:
            assert embeds_path.exists(), f"{embeds_path} does not exist"
            assert config_path.exists(), f"{config_path} does not exist"

        self.audios = list(f.name for f in self.path.glob("*.npz") if f.resolve() != embeds_path)
        self.embeds = Tacotron2SpeakerEmbeds(**(np.load(embeds_path) if embeds_path.exists() else {}))  # type: ignore

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = Tacotron2AudioConfig(**yaml.safe_load(f))

            if config:
                assert self.config == config, f"Config must match ({self.config} != {config})"
        else:
            self.config = config

    @property
    def embeds_path(self):
        return (self.path / "embeds.npz").resolve()

    @property
    def config_path(self):
        return (self.path / "data.yaml").resolve()

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index: int, text_index: int | None = None):
        sample = Tacotron2Sample.load(self.path / self.audios[index])
        text_index = text_index or random.randrange(len(sample.tokens))

        return (
            sample.mel,
            sample.tokens[text_index].astype(np.int64),
            sample.decode(text_index),
            self.embeds(),
        )

    def save(self):
        assert self.config is not None, "No config"
        np.savez(self.embeds_path, **asdict(self.embeds))

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self.config), f)

    def extend(
        self,
        preprocessor: Tacotron2Preprocessor,
        audios: str | PathLike | list[str | PathLike],
        labels: str
        | Path
        | list[str | Path | None]
        | Callable[[Path], str | list[str] | None]
        | None = None,
        *,
        max_workers: int | None = None,
        compress: bool = False,
        test_count: int = 2,
        audio_pattern: str = "*.wav",
        label_suffix: str = ".txt",
        encoding: str | None = "utf-8",
        ensure_sampling_rate: bool = False,
        skip_not_found: bool = False,
        speaker_name: str | None = "",
        verbose: bool = True,
    ) -> float:
        if self.config:
            assert self.config == preprocessor.config, (
                "Preprocessor config must match dataset config "
                f"({self.config} != {preprocessor.config})"
            )
        else:
            self.config = preprocessor.config

        tests_path = self.path / "tests"
        embeds_path = self.embeds_path.resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        tests_path.mkdir(parents=True, exist_ok=True)

        def process(args: tuple[int, str | PathLike]):
            i, audio_path = args
            audio_path = Path(audio_path)

            if not audio_path.exists():
                assert skip_not_found, f"{audio_path} does not exist"
                if verbose:
                    print(f"Skipping {audio_path}: Does not exist")
                return None

            # Get label
            if callable(labels):
                label = labels(audio_path)
            elif isinstance(labels, (list, tuple)):
                label = labels[i]
            else:
                label = labels

                if not label:
                    label = audio_path.with_suffix(label_suffix)

            if isinstance(label, Path):
                if label.exists():
                    label = label.read_text(encoding=encoding)
                else:
                    assert skip_not_found, f"{label} does not exist"
                    label = None

            if not label:
                if verbose:
                    print(f"Skipping {audio_path}: No label")
                return None

            # Process
            sample = preprocessor.process(
                label,
                audio_path,
                ensure_sampling_rate=ensure_sampling_rate,
            )

            # Save
            file_path = self.path / f"{audio_path.stem}.npz"

            if embeds_path == file_path.resolve():
                file_path = self.path / f"{audio_path.stem}_.npz"

            while file_path.exists():
                file_path = (
                    self.path / f"{audio_path.stem}_{''.join(random.sample(RANDOM_STRINGS, 6))}.npz"
                )

            sample.save(file_path, compress=compress)

            # Test
            if i in tests:
                sample.save_test(tests_path / audio_path.stem, preprocessor.config.sampling_rate)

            return (
                file_path,
                sample.embed,
                sample.duration(preprocessor.config.hop_length, preprocessor.config.sampling_rate),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if not isinstance(audios, list):
                audios = list(Path(audios).glob(audio_pattern))

            audio_list = []
            avg = len(audios) // (test_count or 1)
            tests = set(int(avg * i + avg / 2) for i in range(test_count))
            total_time = 0

            for result in tqdm(
                executor.map(process, enumerate(audios)),
                desc=f"Speaker {speaker_name}",
                disable=not verbose,
                leave=False,
                total=len(audios),
            ):
                if not result:
                    continue

                sample, embed, duration = result
                assert embed is not None, "No embed"
                total_time += duration
                self.embeds.update(embed)
                audio_list.append(sample.name)

        self.audios = list(set(self.audios) | set(audio_list))
        self.save()

        print(f"\nSpeaker {speaker_name}: {len(audios)} audios, {total_time / 60 / 60:.2f} hours")

        return total_time


class Tacotron2Dataset(Dataset):
    path: Path
    freezn: bool = False
    speakers: dict[str, Tacotron2SpeakerDataset]

    cached_length: int | None = None
    cached_indices: dict[str, int] | None = None

    _config: Tacotron2AudioConfig | None

    def __init__(
        self,
        path: str | PathLike,
        config: Tacotron2AudioConfig | None = None,
        exclude: set[str] | None = None,
        ensure_config: bool = True,
        skip_incomplete: bool = True,
        delete_incomplete: bool = False,
    ):
        self.path = Path(path)
        self.speakers = {}

        if self.path.exists():
            for speaker_path in self.path.iterdir():
                if not speaker_path.is_dir() or (exclude and speaker_path.name in exclude):
                    continue

                try:
                    dataset = self.speakers[speaker_path.name] = Tacotron2SpeakerDataset(
                        speaker_path,
                        ensure_config=ensure_config or skip_incomplete or delete_incomplete,
                    )
                except AssertionError as e:
                    if delete_incomplete:
                        print(f"Deleting incomplete dataset {speaker_path}: {e}")
                        shutil.rmtree(speaker_path)
                        continue
                    if skip_incomplete:
                        print(f"Skipping incomplete dataset {speaker_path}: {e}")
                        continue
                    else:
                        raise e

                if config:
                    assert dataset.config == config, (
                        "Dataset config must match config " f"({dataset.config} != {config})"
                    )

                config = dataset.config

        self._config = config

    @property
    def config(self):
        assert self._config is not None, "Config is not set"
        return self._config

    def freeze(self):
        self.freezn = True
        self.cached_length = None
        self.cached_length = self.__len__()

        indices = {}
        total = 0
        for key, speaker in self.speakers.items():
            indices[key] = total
            total += len(speaker)

        self.cached_indices = indices

    def unfreeze(self):
        self.freezn = False
        self.cached_length = None
        self.cached_indices = None

    def remove(self, speaker_name: str, delete: bool = False):
        dataset = self.speakers.pop(speaker_name, None)
        assert dataset is not None, f"Speaker {speaker_name} does not exist"

        if delete:
            shutil.rmtree(dataset.path)

    def clear(self, delete: bool = False):
        for speaker_name in list(self.speakers.keys()):
            self.remove(speaker_name, delete=delete)

    def append(
        self,
        preprocessor: Tacotron2Preprocessor,
        audios: str | PathLike | list[str | PathLike],
        labels: str
        | Path
        | list[str | Path | None]
        | Callable[[Path], str | list[str] | None]
        | None = None,
        speaker_name: str | None = None,
        *,
        max_workers: int | None = None,
        compress: bool = False,
        test_count: int = 2,
        audio_pattern: str = "*.wav",
        label_suffix: str = ".txt",
        encoding: str | None = "utf-8",
        ensure_sampling_rate: bool = False,
        skip_not_found: bool = False,
        verbose: bool = True,
    ) -> float:
        assert not self.freezn, "Dataset is frozen"
        if self._config:
            assert self._config == preprocessor.config, (
                "Preprocessor config must match dataset config "
                f"({self._config} != {preprocessor.config})"
            )
        else:
            self._config = preprocessor.config

        if isinstance(audios, list):
            assert speaker_name is not None, "Speaker name must be provided"
        elif speaker_name is None:
            speaker_name = Path(audios).name

        speaker_path = self.path / speaker_name
        dataset = Tacotron2SpeakerDataset(speaker_path, config=self._config)
        total_time = dataset.extend(
            preprocessor,
            audios,
            labels,
            max_workers=max_workers,
            compress=compress,
            test_count=test_count,
            audio_pattern=audio_pattern,
            label_suffix=label_suffix,
            encoding=encoding,
            ensure_sampling_rate=ensure_sampling_rate,
            skip_not_found=skip_not_found,
            speaker_name=speaker_name,
            verbose=verbose,
        )
        self.speakers[speaker_name] = dataset
        return total_time

    def extend(
        self,
        preprocessor: Tacotron2Preprocessor,
        speakers: str
        | PathLike
        | list[str | PathLike]
        | dict[str, str | PathLike | list[str | PathLike]],
        labels: str
        | Path
        | list[str | Path | None]
        | Callable[[Path], str | list[str] | None]
        | dict[
            str | None,
            str | Path | list[str | Path | None] | Callable[[Path], str | list[str] | None] | None,
        ]
        | None = None,
        *,
        max_workers: int | None = None,
        compress: bool = False,
        test_count: int = 2,
        audio_pattern: str = "*.wav",
        label_suffix: str = ".txt",
        encoding: str | None = "utf-8",
        ensure_sampling_rate: bool = False,
        skip_not_found: bool = False,
        verbose: bool = True,
    ) -> float:
        if isinstance(speakers, list):
            speakers = {speaker.name: speaker for speaker in map(Path, speakers) if speaker.is_dir()}
        elif not isinstance(speakers, dict):
            speakers = {
                speaker.name: speaker for speaker in Path(speakers).iterdir() if speaker.is_dir()
            }

        if isinstance(labels, list):
            labels = {speaker: labels[i] for i, speaker in enumerate(speakers.keys())}
        elif callable(labels):
            labels = {speaker: labels for speaker in speakers.keys()}
        elif not isinstance(labels, dict) and labels is not None:
            labels = Path(labels)
            labels = {speaker: labels / speaker for speaker in speakers.keys()}

        total_time = 0
        for result in tqdm(
            (
                self.append(
                    preprocessor,
                    speaker_name=name,
                    audios=audios,
                    labels=labels.get(name, labels[None]) if labels else None,  # type: ignore
                    max_workers=max_workers,
                    compress=compress,
                    test_count=test_count,
                    audio_pattern=audio_pattern,
                    label_suffix=label_suffix,
                    encoding=encoding,
                    ensure_sampling_rate=ensure_sampling_rate,
                    skip_not_found=skip_not_found,
                    verbose=verbose,
                )
                for name, audios in speakers.items()
            ),
            desc="Speakers",
            disable=not verbose,
            leave=False,
            total=len(speakers),
        ):
            total_time += result

        if verbose:
            print(f"Total time: {total_time / 60 / 60:.2f} hours")

        return total_time

    def __len__(self):
        if self.freezn and self.cached_length is not None:
            return self.cached_length

        length = sum(len(s) for s in self.speakers.values())

        if self.freezn:
            self.cached_length = length

        return length

    def __getitem__(self, index: int):
        if self.freezn and self.cached_indices is not None:
            for key, start_index in self.cached_indices.items():
                if start_index <= index < start_index + len(self.speakers[key]):
                    return self.speakers[key][index - start_index]
        else:
            for speaker in self.speakers.values():
                if index < len(speaker):
                    return speaker[index]

                index -= len(speaker)

        raise IndexError(f"Index {index} out of range")

    def loaders(
        self,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        validation_split: float | int = 0.05,
    ) -> tuple[DataLoader, DataLoader]:
        total_size = len(self)
        assert total_size > 0, "Dataset is empty"
        validation_size = (
            int(total_size * validation_split) if validation_split < 1 else int(validation_split)
        )

        assert (
            batch_size < validation_size and batch_size < total_size - validation_size
        ), "Batch size is too large"

        train_dataset, validation_dataset = random_split(
            self, [total_size - validation_size, validation_size]
        )

        return (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=_collate_fn,
                num_workers=num_workers,
            ),
            DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_collate_fn,
                num_workers=num_workers,
            ),
        )


def _collate_fn(batch: list[tuple[np.ndarray, np.ndarray, str, np.ndarray]]):
    lengths = np.array([i[1].shape[0] for i in batch])  # token length
    indexs = np.argsort(-lengths)  # sort by token length

    return Tacotron2Batch.make_batch(
        (batch[i] for i in indexs),  # type: ignore
    )
