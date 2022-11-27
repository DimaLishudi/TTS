import torch
import torchaudio
import numpy as np
import os
import pyworld as pw
from tqdm.auto import trange

wav_dir_path = './data/LJSpeech-1.1/wavs/'
wav_paths = sorted(os.listdir(wav_dir_path))

# these shouldn't really be changed, so they aren't in config
n_mels = 80
n_fft = 1024
win_length = 1024
hop_length = 256
sample_rate = 22050
f_min = 0.0
f_max = 8000.0

spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=1)
mel_transform = torchaudio.transforms.MelScale(n_mels=n_mels, n_stft=n_fft//2 + 1, sample_rate=sample_rate, f_min=f_min, f_max=f_max, norm='slaney', mel_scale='slaney')

@torch.inference_mode()
def preprocess():
    os.makedirs('./data/mels', exist_ok=True)
    os.makedirs('./data/energies', exist_ok=True)
    os.makedirs('./data/pitches', exist_ok=True)

    energy_max = 0
    energy_min = np.inf

    pitch_max = 0
    pitch_min = np.inf


    for i in trange(len(wav_paths)):
        # mels and energy
        wav, sr = torchaudio.load(wav_dir_path + wav_paths[i])
        wav = wav.squeeze()
        assert sr == sample_rate, f"sample rate expected to be {sample_rate}"
        spec = spec_transform(wav)
        mel = mel_transform(spec).clamp(min=1e-5).log()
        
        energy = np.asarray(torch.norm(spec, p=2, dim=-2))


        # calculate pitch
        # borrowed from https://github.com/ming024/FastSpeech2/blob/master/preprocessor/preprocessor.py

        wav = np.asarray(wav)
        pitch, t = pw.dio(
            wav.astype(np.float64),
            sample_rate,
            frame_period=hop_length / sample_rate * 1000, # 1000 ms in sec
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sample_rate)

        # linearly interpolate zeroes in pitch and take log

        zero_idx = pitch==0
        pitch[zero_idx] = np.interp(x=np.argwhere(zero_idx).squeeze(), xp=np.argwhere(~zero_idx).squeeze(), fp=pitch[~zero_idx])
        pitch = np.log(pitch)

        # calculate  energy and pitch min/max for bins later
        energy_max = max(energy_max, energy.max())
        energy_min = min(energy_min, energy.min())
        pitch_max = max(pitch_max, pitch.max())
        pitch_min = min(pitch_min, pitch.min())

        # save results
        np.save(f"./data/mels/ljspeech-mel-{i:05}.npy", np.asarray(mel).T)
        np.save(f"./data/energies/ljspeech-energy-{i:05}.npy", energy)
        np.save(f"./data/pitches/ljspeech-pitch-{i:05}.npy", pitch)
    np.save("./data/energy_max_min.npy", np.array([energy_min, energy_max]))
    np.save("./data/pitch_max_min.npy", np.array([pitch_min, pitch_max]))


if __name__ == "__main__":
    preprocess()