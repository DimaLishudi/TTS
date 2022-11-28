import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers
from . import modules
from . import utils


class VarianceAdaptor(nn.Module):
    """ VarianceAdaptor """

    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()
        self.length_regulator = modules.LengthRegulator(model_config)
        self.pitch_predictor  = layers.VarianceAdaptorPredictor(model_config)
        self.energy_predictor = layers.VarianceAdaptorPredictor(model_config)

        n_bins = model_config['n_bins']
        # in case we don't have stats we still may load model checkpoints with needed bins
        # we'll replace min and max with adequate values
        if 'pitch_stats_path' in model_config:
            stats_arr = np.load(model_config['pitch_stats_path'])
            pitch_min = stats_arr[2]
            pitch_max = stats_arr[3]
        else:
            pitch_min = -100
            pitch_max = 100
        if 'energy_stats_path' in model_config:
            stats_arr = np.load(model_config['energy_stats_path'])
            energy_min = stats_arr[2]
            energy_max = stats_arr[3]
        else:
            energy_min = -100
            energy_max = 100

        self.register_buffer('pitch_bins', torch.linspace(pitch_min, pitch_max, n_bins - 1))
        self.register_buffer('energy_bins', torch.linspace(energy_min, energy_max, n_bins - 1))

        self.pitch_embedding = nn.Embedding(n_bins, model_config["encoder_dim"])
        self.energy_embedding = nn.Embedding(n_bins, model_config["encoder_dim"])


    def forward(self, x,
                duration_coef=1.0, pitch_coef=1.0, energy_coef=1.0,
                target_duration=None, target_pitch=None, target_energy=None,
                mel_max_length=None):
        output, dur = self.length_regulator(x, duration_coef, target_duration, mel_max_length)

        pitch_pred = self.pitch_predictor(output)
        energy_pred = self.energy_predictor(output)
        if target_pitch is not None:
            pitch = target_pitch
        else:
            pitch = pitch_pred * pitch_coef
        if target_energy is not None:
            energy = target_energy
        else:
            energy = energy_pred * energy_coef
        pitch = self.pitch_embedding(torch.bucketize(pitch, self.pitch_bins))
        energy = self.pitch_embedding(torch.bucketize(energy, self.energy_bins))
        output = output + pitch + energy
        return output, dur, pitch_pred, energy_pred

    

class FastSpeech2(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config):
        super(FastSpeech2, self).__init__()
        n_mels = 80

        self.encoder = modules.Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config)
        self.decoder = modules.Decoder(model_config)

        self.mel_linear = nn.Linear(model_config['decoder_dim'], n_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~utils.get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self,
                src_seq, src_pos, mel_pos=None, mel_max_length=None,
                length_target=None, pitch_target=None, energy_target=None,
                dur_coef=1.0, pitch_coef=1.0, energy_coef=1.0, **kwargs):
        encoder_out, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            va_output, duration_pred, pitch_pred, energy_pred = self.variance_adaptor(
                encoder_out, dur_coef, pitch_coef, energy_coef,
                length_target, pitch_target, energy_target, mel_max_length
            )
            output = self.decoder(va_output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, duration_pred, pitch_pred, energy_pred
        
        va_output, mel_pos, _, _ = self.variance_adaptor(
            encoder_out, dur_coef, pitch_coef, energy_coef,
            length_target, pitch_target, energy_target, mel_max_length
        )
        output = self.decoder(va_output, mel_pos)
        output = self.mel_linear(output)
        return output
