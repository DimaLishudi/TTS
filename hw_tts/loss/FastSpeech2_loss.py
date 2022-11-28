import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(
            self,
            mel, log_duration_predicted, log_pitch_predicted, energy_predicted,\
            mel_target, duration_predictor_target, log_pitch_target, energy_target,\
            *args, **kwargs):
        mel_loss = self.l1_loss(mel, mel_target)

        duration_loss = self.mse_loss(
                log_duration_predicted,
                torch.log(duration_predictor_target.float() + 1)
        )
                                                
        pitch_loss = self.mse_loss(log_pitch_predicted, log_pitch_target)
        energy_loss = self.mse_loss(energy_predicted, energy_target)
        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

        return total_loss, mel_loss, duration_loss, pitch_loss, energy_loss
