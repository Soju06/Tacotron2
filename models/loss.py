import torch
import torch.nn as nn


class GuidedAttentionLoss(nn.Module):
    def __init__(
        self,
        sigma: float = 0.4,
        alpha: float = 1.0,
        reset_always: bool = True,
    ):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(
                att_ws.device
            )

        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)

        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))

        if self.reset_always:
            self._reset_masks()

        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))

        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma
            )

        return guided_attn_masks

    def _make_masks(self, ilens, olens):
        in_masks = self.make_non_pad_mask(ilens)
        out_masks = self.make_non_pad_mask(olens)

        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)

    def make_non_pad_mask(self, lengths, xs=None, length_dim=-1):
        return ~self.make_pad_mask(lengths, xs, length_dim)

    def make_pad_mask(self, lengths, xs=None, length_dim=-1):
        if length_dim == 0:
            raise ValueError("length_dim cannot be 0: {}".format(length_dim))

        if not isinstance(lengths, list):
            lengths = lengths.tolist()

        bs = int(len(lengths))
        maxlen = int(max(lengths)) if xs is None else xs.size(length_dim)
        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand

        if xs is not None:
            assert xs.size(0) == bs, (xs.size(0), bs)

            if length_dim < 0:
                length_dim = xs.dim() + length_dim

            ind = tuple(
                slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
            )
            mask = mask[ind].expand_as(xs).to(xs.device)

        return mask

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        grid_x, grid_y = torch.meshgrid(
            torch.arange(olen), torch.arange(ilen), indexing="ij"
        )
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)

        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma**2))
        )


class Tacotron2Loss(nn.Module):
    def __init__(
        self,
        n_frames_per_step: int,
        guided_sigma: float,
        guided_lambda: float,
    ):
        super(Tacotron2Loss, self).__init__()
        self.n_frames_per_step = n_frames_per_step

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.guided_attn_loss = GuidedAttentionLoss(
            sigma=guided_sigma,
            alpha=guided_lambda,
        )

    def forward(
        self,
        mels: torch.Tensor,
        gates: torch.Tensor,
        tokens_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
        mel_pad: int,
        mels_out: torch.Tensor,
        mels_out_postnet: torch.Tensor,
        gates_out: torch.Tensor,
        alignments: torch.Tensor,
    ):
        mels.requires_grad = False
        gates.requires_grad = False
        gates = gates.view(-1, 1)

        gates_out = gates_out.view(-1, 1)
        mel_loss = self.mse_loss(mels_out, mels) + self.mse_loss(mels_out_postnet, mels)
        gate_loss = self.bce_loss(gates_out, gates)

        attn_loss = self.guided_attn_loss(
            alignments,
            tokens_lengths,
            (mel_lengths + mel_pad) // self.n_frames_per_step,
        )

        return mel_loss, gate_loss, attn_loss
