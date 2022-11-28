from datetime import datetime
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import io
from torchvision.transforms import ToTensor
from PIL import Image


class WanDBWriter:
    def __init__(self, config):
        self.writer = None
        self.selected_module = ""

        wandb.login()

        if "wandb_project" not in config["logger"]:
            raise ValueError("please specify project name for wandb")
        project = config["logger"]["wandb_project"]
        # TODO: all wandb run resuming
        # if 'resume' in config and config['resume']:
        #     wandb.init(
        #         id=config.id,
        #         resume='must',
        #         project=project,
        #         config=config
        #     )
        # else: 
        self.id = wandb.util.generate_id()
        wandb.init(
            id=self.id,
            resume='allow',
            project=project,
            config=config
        )
        self.wandb = wandb

        self.step = 0
        # self.timer = datetime.now()

    def set_step(self, step):
        self.step = step
        # if step == 0:
        #     self.timer = datetime.now()
        # else:
        #     duration = datetime.now() - self.timer
        #     self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
        #     self.timer = datetime.now()

    def scalar_name(self, scalar_name):
        return f"{scalar_name}"

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log({
            self.scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_scalars(self, tag, scalars):
        self.wandb.log({
            **{f"{scalar_name}_{tag}": scalar for scalar_name, scalar in scalars.items()}
        }, step=self.step)

    def add_image(self, scalar_name, image, caption=None):
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Image(image, caption=caption)
        }, step=self.step)

    def add_audio(self, scalar_name, audio, sample_rate=None, caption=None):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy().T
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Audio(audio, sample_rate=sample_rate, caption=caption)
        }, step=self.step)

    def add_text(self, scalar_name, text):
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Html(text)
        }, step=self.step)

    def add_histogram(self, scalar_name, hist, bins=None):
        hist = hist.detach().cpu().numpy()
        np_hist = np.histogram(hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(hist, bins=512)

        hist = self.wandb.Histogram(
            np_histogram=np_hist
        )

        self.wandb.log({
            self.scalar_name(scalar_name): hist
        }, step=self.step)

    def add_spectrogram(self, scalar_name, spec, caption=None):
        plt.figure(figsize=(20, 5))
        plt.imshow(spec.detach().cpu().squeeze().numpy())
        plt.title(scalar_name)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.add_image(scalar_name, ToTensor()(Image.open(buf)), caption=caption)

    def add_images(self, scalar_name, images):
        raise NotImplementedError()

    def add_pr_curve(self, scalar_name, scalar):
        raise NotImplementedError()

    def add_embedding(self, scalar_name, scalar):
        raise NotImplementedError()