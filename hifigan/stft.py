import matplotlib
from matplotlib.figure import Figure
import torch
from librosa.filters import mel as librosa_mel_fn
from matplotlib import pyplot as plt
from torch.nn import functional


class STFT(torch.nn.Module):
    mel_basis: torch.Tensor
    hann_window: torch.Tensor

    def __init__(
        self,
        n_fft: int,
        num_mels: int,
        sampling_rate: int,
        hop_size: int,
        win_size: int,
        fmin: int,
        fmax: int,
    ):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.register_buffer(
            "mel_basis",
            torch.from_numpy(
                librosa_mel_fn(
                    sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
                )
            ).float(),
        )
        self.register_buffer("hann_window", torch.hann_window(win_size))

    def mel(self, y):
        y = functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        ).squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(self.mel_basis, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5))

        return spec

    def plot_mel(self, mel: torch.Tensor) -> Figure:
        mel = mel.cpu().numpy()
        matplotlib.use("Agg")
        fig, ax = plt.subplots()
        plt.colorbar(
            ax.imshow(mel, aspect="auto", origin="lower", interpolation="none"), ax=ax
        )
        fig.canvas.draw()
        plt.close()
        return fig
