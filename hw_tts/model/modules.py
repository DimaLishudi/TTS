import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from . import layers

class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()
        
        len_max_seq=model_config['max_seq_len']
        n_position = len_max_seq + 1
        n_layers = model_config['encoder_n_layer']
        self.pad = model_config['PAD']

        self.src_word_emb = nn.Embedding(
            model_config['vocab_size'],
            model_config['encoder_dim'],
            padding_idx=model_config['PAD']
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config['encoder_dim'],
            padding_idx=model_config['PAD']
        )

        self.layer_stack = nn.ModuleList([layers.FFTBlock(
            model_config,
            model_config['encoder_dim'],
            model_config['encoder_conv1d_filter_size'],
            model_config['encoder_head'],
            model_config['encoder_dim'] // model_config['encoder_head'],
            model_config['encoder_dim'] // model_config['encoder_head'],
            dropout=model_config['dropout']
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = utils.get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad=self.pad)
        non_pad_mask = utils.get_non_pad_mask(src_seq, self.pad)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):

        super(Decoder, self).__init__()

        len_max_seq=model_config['max_seq_len']
        n_position = len_max_seq + 1
        n_layers = model_config['decoder_n_layer']
        self.pad = model_config['PAD']

        self.position_enc = nn.Embedding(
            n_position,
            model_config['encoder_dim'],
            padding_idx=model_config['PAD'],
        )

        self.layer_stack = nn.ModuleList([layers.FFTBlock(
            model_config,
            model_config['encoder_dim'],
            model_config['encoder_conv1d_filter_size'],
            model_config['encoder_head'],
            model_config['encoder_dim'] // model_config['encoder_head'],
            model_config['encoder_dim'] // model_config['encoder_head'],
            dropout=model_config['dropout']
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = utils.get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, pad=self.pad)
        non_pad_mask = utils.get_non_pad_mask(enc_pos, self.pad)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = layers.VarianceAdaptorPredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = utils.create_alignment(alignment,
                                           duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_rounded = ((torch.exp(duration_predictor_output) - 1) * alpha + 0.5).int().clamp(min=0)
            output = self.LR(x, duration_rounded, mel_max_length)
            mel_pos = torch.arange(output.size(1)).unsqueeze(0).to(x.device) + 1
            return output, mel_pos