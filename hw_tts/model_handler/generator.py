import numpy as np
import torch
import hw_tts.dataset, hw_tts.model
from logger import WanDBWriter
from tqdm.auto import tqdm
import os

# TODO:
# import utils, text, waveglow


class TTSGenerator():
    """
        Base Class to generate voice by text
        holds FastSpeech 1/2 TTS model and WaveGlow vocoder
    """
    def __init__(self, config, log=False, checkpoint_path=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataloader = hw_tts.dataset.get_LJSpeech_dataloader(config['dataset'])

        # TODO: reserve checkpoint in config
        self.model = getattr(hw_tts.model, config['model']['type'])(**dict(config['model']['args']))
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))

        self.model = self.model.to(self.device)

        if config.has_key('generator'):
            self.waveglow_model = utils.get_WaveGlow().to(self.device)
            self.waveglow_model.eval()
        
        self.current_step = 0
        self.logger = WanDBWriter() if log else None


    def synthesis(self, input, alpha=1.0, pitch=None, energy=None):
        input = np.array(input)
        input = np.stack([input])
        src_pos = np.array([i+1 for i in range(input.shape[1])])
        src_pos = np.stack([src_pos])
        sequence = torch.from_numpy(input).long().to(self.config['device'])
        src_pos = torch.from_numpy(src_pos).long().to(self.config['device'])
        
        with torch.no_grad():
            mel = self.model.forward(sequence, src_pos, alpha=alpha)
        return mel.contiguous().transpose(1, 2)


    def generate(self):
        if not self.config.has_key('generator'):
            return
        self.model.eval()

        if hasattr(self.config['generator'], 'results_dir'):
            res_dir = self.config['generator']['results_dir']
        else:
            res_dir = './results'
        os.makedirs(res_dir, exist_ok=True)

        data_list = list(self.config['generator']['texts'])

        if hasattr(self.config['generator'], 'alphas'):
            alphas_list = torch.asaray(self.config['generator']['alphas'])
        else:
            alphas_list = torch.ones(len(data_list), dtype=torch.long)


        for i, (input, alpha) in tqdm(enumerate(zip(data_list, alphas_list))):
            seq = text.text_to_sequence(input, self.config['text_cleaners'])
            mel = self.synthesis(seq, alpha)

            
            waveglow.inference.inference(
                mel, self.waveglow_model,
                res_dir + f"/out_{i}_{alpha}.wav"
            )

            self.logger.add_spectrogram(input, mel)

