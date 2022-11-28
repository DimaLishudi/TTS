import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers
from . import utils



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
        ### Your code here
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_rounded = ((torch.exp(duration_predictor_output) - 1) * alpha + 0.5).int().clamp(min=0)
            output = self.LR(x, duration_rounded, mel_max_length)
            mel_pos = torch.arange(output.size(1)).unsqueeze(0).to(x.device) + 1
            return output, mel_pos


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config):
        super(FastSpeech, self).__init__()
        n_mels = 80

        self.encoder = layers.Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.decoder = layers.Decoder(model_config)

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

