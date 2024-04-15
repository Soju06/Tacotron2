import re
import sys
import torch
from hifigan import HiFiGAN, HiFiGANConfig, HiFiGANAudioConfig
from model import Tacotron2, Tacotron2Config, Tacotron2TrainState
from dataset import Tacotron2Dataset, Tacotron2AudioConfig


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = "korean"
    dataset_path = "/mnt/ramdisk/Audios"
    dataset = Tacotron2Dataset(
        "/mnt/ramdisk",
        config=Tacotron2AudioConfig(
            sampling_rate=44100,
            mel_fmin=0,
            mel_fmax=16000,
            filter_length=2048,
            hop_length=256,
            win_length=2048,
        ),
        delete_incomplete=True,
    )

    if not len(dataset):
        dataset.clear(delete=True)
        print("Preprocessing...")

        import json
        from pathlib import Path
        from preprocess import Tacotron2Preprocessor
        from g2pk import G2p
        
        g2p = G2p()

        kss_labels = [
            line.split("|")
            for line in (Path(dataset_path) / "transcript.v.1.4.txt").read_text("utf-8").splitlines()
            if line
        ]
        kss_labels = {Path(label[0]).name: label[2] for label in kss_labels}

        def label_augmentation(label: str) -> list[str]:
            return [
                label,
                # 특수문자 제거
                re.sub(r"[^a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣\s]", "", label),
                # 띄어쓰기 제거
                re.sub(r"\s+", "", label),
                # 음소 변환
                g2p(label),
            ]

        def read_label(audio: Path):
            label_path = audio.with_suffix(".json")

            if not label_path.exists():
                return None

            with open(label_path, "r", encoding="utf-8") as f:
                return label_augmentation(json.load(f)["전사정보"]["TransLabelText"])

        def read_kss_label(audio: Path):
            label = kss_labels.get(audio.name)
            return label_augmentation(label) if label else None

        preprocessor = Tacotron2Preprocessor(
            tokenizer=tokenizer,
            config=dataset.config,
        )

        dataset.extend(
            preprocessor,
            dataset_path,
            {None: read_label, "kss": read_kss_label},
            ensure_sampling_rate=True,
        )

        # TF Overhead
        sys.exit()

    # TODO: fix time calc bug

    print("Training...")
    hifigan = HiFiGAN.load("../hifigan_model_1200000_0.3101.pt", device=device)
    tacotron2 = Tacotron2(
        config=Tacotron2Config(
            tokenizer=tokenizer,
            encoder_n_convolutions=5,
            prenet_dim=512,
            attention_rnn_dim=2048,
            attention_dim=256,
            postnet_embedding_dim=1024,
            max_decoder_steps=10000,
        )
    )
    # tacotron2 = Tacotron2.load("./output/checkpoints/t7/model_450000_0.2514.pt", device=device)

    trainer = tacotron2.trainer(
        Tacotron2TrainState() # ----------------------------------- reset
    )
    trainer.set_validation_interval(2500)
    trainer.set_checkpoint_interval(10000)
    trainer.set_checkpoint_path(
        "output/checkpoints/t9",
        "model_{step}_{mel_loss:.4f}.pt",
        "model_best_{validation_loss:.4f}.pt",
    )
    trainer.set_summary_writer("output/logs/t9")
    trainer.set_sample_count(14)
    trainer.set_dataset(
        dataset,
        batch_size=14,
        validation_split=0.2,
    )
    trainer.set_vocoder(hifigan)

    trainer.train(100, early_stop=50)

    trainer
