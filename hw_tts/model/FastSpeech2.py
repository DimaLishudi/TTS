import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import modules
from . import utils


class FastSpeech2(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config):
        super(FastSpeech2, self).__init__()
        n_mels = 80

        self.encoder = modules.Encoder(model_config)
        self.VarianceAdaptor = modules.VarianceAdaptor(model_config)
        self.decoder = modules.Decoder(model_config)

        self.mel_linear = nn.Linear(model_config['decoder_dim'], n_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~utils.get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0, **kwargs):
        encoder_out, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            lr_output, duration_predictor_output = self.length_regulator(encoder_out, alpha, length_target, mel_max_length)
            output = self.decoder(lr_output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, duration_predictor_output
        
        lr_output, mel_pos = self.length_regulator(encoder_out, alpha, length_target, mel_max_length)
        output = self.decoder(lr_output, mel_pos)
        output = self.mel_linear(output)
        return output

