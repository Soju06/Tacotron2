import torch
from hifigan import HiFiGAN, HiFiGANConfig, HiFiGANAudioConfig, HiFiGANTrainState


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HiFiGAN(HiFiGANConfig()).to(device)
    # model = HiFiGAN.load("checkpoints/14/model_270000_0.2830.pt", device=device)

    trainer = model.trainer(
        audio=HiFiGANAudioConfig(
            sampling_rate=44100,
            mel_fmin=0,
            mel_fmax=16000,
            filter_length=2048,
            hop_length=256,
            win_length=2048,
        ),
        train_state=HiFiGANTrainState(
            # learning_rate=0.0015,
        ),
    )
    trainer.set_log_interval(100)
    trainer.set_validation_interval(100)
    trainer.set_checkpoint_interval(5000)
    trainer.set_checkpoint_path(
        "checkpoints/15",
        "model_{step}_{mel_loss:.4f}.pt",
        "model_best_{validation_loss:.4f}.pt",
    )
    trainer.set_summary_writer("logs/15")
    trainer.set_sample_count(8)
    trainer.set_dataset_from_path(
        "A:\\Audios",
        batch_size=8,
        pattern="**/*.wav",
        validation_split=0.1,
        # segment_length=12288,
        ignore_sampling_rate_error=True,
    )

    trainer.train(1000)

    trainer
