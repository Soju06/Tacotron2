from os import PathLike
from pathlib import Path
from threading import Lock
import numpy as np
from . import gdown
from .utils import read_mfcc, sample_from_mfcc
from .model import DeepSpeakerModel, NUM_FRAMES, NUM_FBANKS
import tensorflow as tf

TRAINED_MODELS: dict[str, str] = {
    "ResCNN_triplet_training_checkpoint_265.h5": "1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP",
}


class DeepSpeakerEmbedder:
    sampling_rate: int
    win_length: int

    def __init__(
        self,
        model_path: str | PathLike | None = None,
        sampling_rate: int = 22050,
        win_length: int = 1024,
        batch_input_shape: tuple[int | None, int, int, int] = (
            None,
            NUM_FRAMES,
            NUM_FBANKS,
            1,
        ),
        device: str | None = None,
        lock: bool = True,
    ):
        if not model_path:
            model_path = (
                Path(__file__).parent
                / "pretrained_models"
                / next(iter(TRAINED_MODELS.keys()))
            )

        model_path = Path(model_path)
        if not model_path.exists():
            if model_path.name not in TRAINED_MODELS:
                raise ValueError(
                    f"Model {model_path} does not exist and is not a trained model"
                )

            print(f"Downloading DeepSpeaker model {model_path.name}...")
            gdown.download(
                TRAINED_MODELS[model_path.name], model_path, desc="Downloading model"
            )

        if not device:
            device = next(
                iter(tf.config.experimental.list_physical_devices("GPU")), None
            )
            device = "/gpu:0" if device else "/cpu:0"

        with tf.device(device):
            self.model = DeepSpeakerModel(batch_input_shape=batch_input_shape)
            self.model.m.load_weights(model_path, by_name=True)
        self.sampling_rate = sampling_rate
        self.win_length = win_length
        self.max_length = batch_input_shape[1]
        self.lock = Lock() if lock else None

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        win_length: int | None = None,
    ) -> np.ndarray:
        sample_rate = sample_rate or self.sampling_rate
        win_length = win_length or self.win_length
        mfcc = sample_from_mfcc(
            read_mfcc(audio, sample_rate, win_length), self.max_length
        )
        mfcc = np.expand_dims(mfcc, axis=0)

        if self.lock:
            self.lock.acquire()
        try:
            return self.model.m.predict(mfcc)
        finally:
            if self.lock:
                self.lock.release()

    def __call__(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        win_length: int | None = None,
    ) -> np.ndarray:
        return self.predict(audio, sample_rate, win_length)
