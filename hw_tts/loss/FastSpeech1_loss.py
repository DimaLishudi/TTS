import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, log_duration_predicted, mel_target, duration_predictor_target, *args, **kwargs):
        mel_loss = self.l1_loss(mel, mel_target)

        duration_predictor_loss = self.mse_loss(log_duration_predicted,
                                                torch.log(duration_predictor_target.float() + 1))
        
        total_loss = mel_loss + duration_predictor_loss

        return total_loss, mel_loss, duration_predictor_loss, None, None
