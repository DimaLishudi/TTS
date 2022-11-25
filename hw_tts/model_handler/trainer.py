import torch
from torch.optim.lr_scheduler  import OneCycleLR
import hw_tts.loss
from tqdm.auto import tqdm
import os
from generator import TTSGenerator

class TTSTrainer(TTSGenerator):
    """
        Class to generate voice by text and train FastSpeech 1/2 TTS model
        holds FastSpeech 1/2 TTS model and WaveGlow vocoder
    """
    def __init__(self, config):
        super().__init__(config, log=True)

        self.Loss = getattr(hw_tts.loss, config['loss'])()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['trainer']['learning_rate'],
            weight_decay=config['optimizer']['weight_decay'],
            betas=config['optimizer']['betas'],
            eps=1e-9
        )
        scheduler_kwargs = dict(config['scheduler']) | {
            "steps_per_epoch": len(self.dataloader) * config['dataset']['batch_expand_size'],
            "epochs": config['trainer']['epochs'],
            "max_lr": config['trainer']['learning_rate'],
        }
        self.scheduler = OneCycleLR(self.optimizer, **scheduler_kwargs)


    def train_loop(self):
        epochs = self.config['trainer']['epochs']
        tqdm_bar = tqdm(epochs * len(self.dataloader) * self.config['dataset']['batch_expand_size']- self.current_step)
        device = self.device

        for epoch in range(epochs):
            for i, batchs in enumerate(self.dataloader):
                # real batch start here
                for j, db in enumerate(batchs):
                    self.current_step += 1
                    tqdm_bar.update(1)
                    
                    # logger.set_step(current_step)

                    # Get Data
                    character = db["text"].long().to(device)
                    mel_target = db["mel_target"].float().to(device)
                    duration = db["duration"].int().to(device)
                    mel_pos = db["mel_pos"].long().to(device)
                    src_pos = db["src_pos"].long().to(device)
                    max_mel_len = db["mel_max_len"]

                    # Forward
                    mel_output, duration_predictor_output = self.model(character,
                                                                src_pos,
                                                                mel_pos=mel_pos,
                                                                mel_max_length=max_mel_len,
                                                                length_target=duration)

                    # Calc Loss
                    total_loss, mel_loss, duration_loss = self.Loss(mel_output,
                                                        duration_predictor_output,
                                                        mel_target,
                                                        duration)

                    # Logger
                    t_l = total_loss.detach().cpu().numpy()
                    m_l = mel_loss.detach().cpu().numpy()
                    d_l = duration_loss.detach().cpu().numpy()

                    self.logger.add_scalar("duration_loss", d_l)
                    self.logger.add_scalar("mel_loss", m_l)
                    self.logger.add_scalar("total_loss", t_l)
                    # Backward
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()

                    # Clipping gradients to avoid gradient explosion
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config['trainer']['grad_clip_thresh'])
                    
                    self.optimizer.step()
                    self.scheduler.step()

                    if self.current_step % self.config['trainer']['save_step'] == 0:
                        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(
                        )}, os.path.join(self.config['trainer']['save_dir'], 'checkpoint_%d.pth.tar' % self.current_step))
                        print("save model at step %d ..." % self.current_step)

                    if self.current_step % self.config['trainer']['validate_step'] == 0:
                        self.generate()