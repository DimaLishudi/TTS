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

    energy_mean = 0
    energy_second_moment = 0
    energy_max = 0
    energy_min = np.inf

    pitch_mean = 0
    pitch_second_moment = 0
    pitch_max = 0
    pitch_min = np.inf

    total_len = 0 # total number of frames in dataset

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

        # update energy and pitch mean/std/min/max
        cur_len = len(energy)

        energy_mean = (energy_mean * total_len + np.sum(energy)) / (total_len + cur_len)
        energy_second_moment = (energy_second_moment * total_len + np.sum(energy**2)) / (total_len + cur_len)
        energy_max = max(energy_max, energy.max())
        energy_min = min(energy_min, energy.min())
        pitch_mean = (pitch_mean * total_len + np.sum(pitch)) / (total_len + cur_len)
        pitch_second_moment = (pitch_second_moment * total_len + np.sum(pitch**2)) / (total_len + cur_len)
        pitch_max = max(pitch_max, pitch.max())
        pitch_min = min(pitch_min, pitch.min())

        total_len += cur_len

        # save mels
        np.save(f"./data/mels/ljspeech-mel-{i:05}.npy", np.asarray(mel).T)

    # calc std and save statistics
    # mean and std are not used in model, as enegries and pitches are normalized, but we save them just in case
    energy_std = np.sqrt(energy_second_moment - energy_mean**2)
    pitch_std = np.sqrt(pitch_second_moment - pitch_mean**2)

    energy_min = (energy_min - energy_mean) / energy_std
    energy_max = (energy_max - energy_mean) / energy_std
    pitch_min = (pitch_min - pitch_mean) / pitch_std
    pitch_max = (pitch_max - pitch_mean) / pitch_std
    np.save("./data/energy_mean_std_min_max.npy", np.array([energy_mean, energy_std, energy_min, energy_max]))
    np.save("./data/pitch_mean_std_min_max.npy" ,  np.array([pitch_mean, pitch_std, pitch_min, pitch_max]))

    # another cycle to save normalized energies and pitches
    for i in trange(len(wav_paths)):
        # mels and energy
        wav, sr = torchaudio.load(wav_dir_path + wav_paths[i])
        wav = wav.squeeze()
        assert sr == sample_rate, f"sample rate expected to be {sample_rate}"
        spec = spec_transform(wav)
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

        # normalize

        pitch = (pitch - pitch_mean) / pitch_std
        energy = (energy - energy_mean) / energy_std

        # save results
        np.save(f"./data/energies/ljspeech-energy-{i:05}.npy", energy)
        np.save(f"./data/pitches/ljspeech-pitch-{i:05}.npy", pitch)


if __name__ == "__main__":
    preprocess()