import torch
from torch.optim.lr_scheduler  import OneCycleLR
import hw_tts.loss
from tqdm.auto import tqdm
import os
from .generator import TTSGenerator

class TTSTrainer(TTSGenerator):
    """
        Class to generate voice by text and train FastSpeech 1/2 TTS model
        holds FastSpeech 1/2 TTS model and WaveGlow vocoder
    """
    def __init__(self, config, checkpoint_path=None):
        super().__init__(config, log=True, checkpoint_path=checkpoint_path)

        self.Loss = getattr(hw_tts.loss, config['loss_type'])()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['trainer']['learning_rate'],
            weight_decay=config['optimizer']['weight_decay'],
            betas=config['optimizer']['betas'],
            eps=1e-9
        )
        scheduler_kwargs = config['lr_scheduler'] | {
            "steps_per_epoch": len(self.dataloader) * config['dataset']['batch_expand_size'],
            "epochs": config['trainer']['epochs'],
            "max_lr": config['trainer']['learning_rate'],
        }
        self.scheduler = OneCycleLR(self.optimizer, **scheduler_kwargs)
        os.makedirs(self.config['trainer']['save_dir'], exists_ok=True)


    def train_loop(self):
        epochs = self.config['trainer']['epochs']
        tqdm_bar = tqdm(total=epochs * len(self.dataloader) * self.config['dataset']['batch_expand_size']- self.current_step)
        device = self.device

        for epoch in range(epochs):
            self.logger.add_scalar("epoch", epoch+1)
            for i, batchs in enumerate(self.dataloader):
                self.logger.add_scalar(
                    "learning rate", self.scheduler.get_last_lr()[0]
                )

                # real batch start here
                for j, db in enumerate(batchs):
                    self.current_step += 1
                    tqdm_bar.update(1)
                    
                    self.logger.set_step(self.current_step)

                    # Get Data
                    character = db["text"].long().to(device)
                    mel_target = db["mel_target"].float().to(device)
                    duration = db["duration"].int().to(device)
                    mel_pos = db["mel_pos"].long().to(device)
                    src_pos = db["src_pos"].long().to(device)
                    max_mel_len = db["mel_max_len"]

                    # Forward
                    mel_output, duration_predictor_output = self.model(
                            character,
                            src_pos,
                            mel_pos=mel_pos,
                            mel_max_length=max_mel_len,
                            length_target=duration
                    )

                    # Calc Loss: total, mel, duration, pitch, energy
                    total_loss, m_loss, d_loss, p_loss, e_loss = self.Loss(
                            mel_output,
                            duration_predictor_output,
                            mel_target,
                            duration
                    )

                    # Logger
                    t_loss = total_loss.detach().cpu().numpy()
                    m_loss = m_loss.detach().cpu().numpy()
                    d_loss = d_loss.detach().cpu().numpy()
                    if p_loss is not None:
                        p_loss = p_loss.detach().cpu().numpy()
                    if e_loss is not None:
                        e_loss = e_loss.detach().cpu().numpy()

                    self.logger.add_scalar("duration_loss", d_loss)
                    self.logger.add_scalar("pitch_loss"   , p_loss)
                    self.logger.add_scalar("energy_loss"  , e_loss)
                    self.logger.add_scalar("mel_loss"     , m_loss)
                    self.logger.add_scalar("total_loss"   , t_loss)
                    # Backward
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()

                    # Clipping gradients to avoid gradient explosion
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config['trainer']['grad_clip_thresh'])
                    
                    self.optimizer.step()
                    self.scheduler.step()

                    if (self.current_step+1) % self.config['trainer']['save_step'] == 0:
                        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(
                        )}, os.path.join(self.config['trainer']['save_dir'], 'checkpoint_%d.pth.tar' % self.current_step))
                        print("save model at step %d ..." % self.current_step)

                    if (self.current_step+1) % self.config['trainer']['val_step'] == 0:
                        self.generate()
                        self.model.train()

        
        if (self.current_step+1) % self.config['trainer']['save_step'] == 0:
            torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(
            )}, os.path.join(self.config['trainer']['save_dir'], 'checkpoint_final.pth.tar'))
            print("save final model")