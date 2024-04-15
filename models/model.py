from math import sqrt

import torch.nn as nn

from .modules import *


class Tacotron2Model(nn.Module):
    def __init__(
        self,
        n_symbols: int,
        mask_padding: bool,
        n_mel_channels: int,
        symbols_embedding_dim: int,
        encoder_n_convolutions: int,
        encoder_embedding_dim: int,
        encoder_kernel_size: int,
        n_frames_per_step: int,
        attention_rnn_dim: int,
        decoder_rnn_dim: int,
        prenet_dim: int,
        max_decoder_steps: int,
        gate_threshold: float,
        attention_dropout: float,
        decoder_dropout: float,
        attention_dim: int,
        attention_location_n_filters: int,
        attention_location_kernel_size: int,
        postnet_embedding_dim: int,
        postnet_kernel_size: int,
        postnet_n_convolutions: int,
        speaker_embeddint_dim: int,
    ):
        super(Tacotron2Model, self).__init__()
        n_symbols = n_symbols + 1

        self.mask_padding = mask_padding
        self.n_mel_channels = n_mel_channels
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(
            encoder_n_convolutions=encoder_n_convolutions,
            encoder_embedding_dim=encoder_embedding_dim,
            encoder_kernel_size=encoder_kernel_size,
        )
        self.decoder = Decoder(
            n_mel_channels=n_mel_channels,
            n_frames_per_step=n_frames_per_step,
            encoder_embedding_dim=encoder_embedding_dim,
            attention_rnn_dim=attention_rnn_dim,
            decoder_rnn_dim=decoder_rnn_dim,
            prenet_dim=prenet_dim,
            max_decoder_steps=max_decoder_steps,
            gate_threshold=gate_threshold,
            attention_dropout=attention_dropout,
            decoder_dropout=decoder_dropout,
            attention_dim=attention_dim,
            attention_location_n_filters=attention_location_n_filters,
            attention_location_kernel_size=attention_location_kernel_size,
        )
        self.postnet = Postnet(
            n_mel_channels=n_mel_channels,
            postnet_embedding_dim=postnet_embedding_dim,
            postnet_kernel_size=postnet_kernel_size,
            postnet_n_convolutions=postnet_n_convolutions,
        )

        self.speaker_emb = LinearNorm(
            speaker_embeddint_dim,
            encoder_embedding_dim,
        )

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, outputs[0].size(2))
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        outputs[0] = outputs[0].transpose(-2, -1)
        outputs[1] = outputs[1].transpose(-2, -1)
        return outputs

    def forward(
        self,
        texts,
        text_lens,
        mels,
        mel_lens,
        spker_embeds=None,
    ):
        embedded_inputs = self.embedding(texts).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lens)
        encoder_outputs = encoder_outputs + self.speaker_emb(spker_embeds).unsqueeze(
            1
        ).expand(-1, max(text_lens), -1)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels.transpose(-2, -1), memory_lengths=text_lens
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments], mel_lens
        )

    def inference(
        self,
        texts,
        max_src_len=None,
        spker_embeds=None,
    ):
        embedded_inputs = self.embedding(texts).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        encoder_outputs = encoder_outputs + self.speaker_emb(spker_embeds).unsqueeze(
            1
        ).expand(-1, max_src_len, -1)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        )
