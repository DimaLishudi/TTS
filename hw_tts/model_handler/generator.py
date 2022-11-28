import numpy as np
import torch
import hw_tts.dataset, hw_tts.model
from hw_tts.logger import WanDBWriter
from tqdm.auto import tqdm
import os

from ..FS_utils import text, utils, waveglow


class TTSGenerator():
    """
        Base Class to generate voice by text
        holds FastSpeech 1/2 TTS model and WaveGlow vocoder
    """
    def __init__(self, config, log=False, checkpoint_path=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataloader = hw_tts.dataset.get_LJSpeech_dataloader(config['dataset'])
        # prepare fastspeech model, vocoder ================================================
        # TODO: alternative checkpoint in config
        self.model = getattr(hw_tts.model, config['model']['type'])(config['model']['args'])
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))

        self.model = self.model.to(self.device)

        if 'generator' in config:
            self.waveglow_model = utils.get_WaveGlow().to(self.device)
            self.waveglow_model.eval()
        
        self.current_step = 0
        self.logger = WanDBWriter(config) if log else None

        # prepare cleaners ==================================================================

        if 'dataset' in self.config:
            self.text_cleaners = self.config['dataset']['text_cleaners']
        else:
            self.text_cleaners = self.config['text_cleaners']

        # prepare inputs ====================================================================

        if 'results_dir' in self.config['generator']:
            self.res_dir = self.config['generator']['results_dir']
            os.makedirs(self.res_dir, exist_ok=True)

        self.text_list = list(self.config['generator']['texts'])

        # get alpha/pitch/energy for synthesis, else set to 1
        if 'alphas_list' in self.config['generator']:
            self.alphas_list = np.asarray(self.config['generator']['alphas_list'])
        else:
            self.alphas_list = np.ones(len(self.text_list))
        if 'pitches_list' in self.config['generator']:
            self.pitches_list = np.asarray(self.config['generator']['pitches_list'])
        else:
            self.pitches_list = np.ones(len(self.text_list))
        if 'energies_list' in self.config['generator']:
            self.energies_list = np.asarray(self.config['generator']['energies_list'])
        else:
            self.energies_list = np.ones(len(self.text_list))

        self.input_zip = zip(self.text_list, self.alphas_list, self.pitches_list, self.energies_list)


    def synthesis(self, input, alpha=1.0, pitch=1.0, energy=1.0):
        input = np.array(input)
        input = np.stack([input])
        src_pos = np.array([i+1 for i in range(input.shape[1])])
        src_pos = np.stack([src_pos])
        sequence = torch.from_numpy(input).long().to(self.device)
        src_pos = torch.from_numpy(src_pos).long().to(self.device)
        
        with torch.no_grad():
            mel = self.model.forward(sequence, src_pos, alpha=alpha, pitch=pitch, energy=energy)
        return mel.contiguous().transpose(1, 2)


    @torch.inference_mode()
    def generate(self):
        if 'generator' not in self.config:
            return
        self.model.eval()

        for i, (input, alpha, pitch, energy) in tqdm(enumerate(self.input_zip)):
            alpha = round(alpha, 2)
            pitch = round(pitch, 2)
            energy = round(energy, 2)
    
            seq = text.text_to_sequence(input, self.text_cleaners)
            mel = self.synthesis(seq, alpha, pitch, energy)

            if 'results_dir' in self.config['generator']:
                res_path = self.res_dir + f"/out_{i}_d={alpha}_p={pitch}_e={energy}.wav"
            else:
                './tmp.wav'
            waveglow.inference.inference(
                mel, self.waveglow_model,
                res_path
            )
            name = f'input_{i}_d={alpha}_p={pitch}_e={energy}'
            caption = input + '\t' + f'd={alpha}_p={pitch}_e={energy}'
            self.logger.add_audio('audio ' + name, res_path, caption=caption)
            self.logger.add_spectrogram('spec ' + name, mel, caption=caption)
        
        if 'results_dir' not in self.config['generator']:
            os.remove('./tmp.wav')

