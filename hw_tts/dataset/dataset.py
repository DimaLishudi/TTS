from hw_tts.FS_utils.text import text_to_sequence
import numpy as np
import torch
import torch.nn.functional as F
from time import perf_counter
from tqdm.auto import tqdm

import os

from torch.utils.data import Dataset, DataLoader


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_data_to_buffer(dataset_config):
    buffer = list()
    text = process_text(dataset_config['data_path'])

    start = perf_counter()
    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            dataset_config['mel_ground_truth'], "ljspeech-mel-%05d.npy" % (i))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            dataset_config['alignment_path'], str(i)+".npy"))
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, dataset_config['text_cleaners']))

        pitch_gt_name = os.path.join(
            dataset_config['pitch_ground_truth'], "ljspeech-pitch-%05d.npy" % (i))
        pitch = np.load(pitch_gt_name)
        energy_gt_name = os.path.join(
            dataset_config['energy_ground_truth'], "ljspeech-energy-%05d.npy" % (i))
        energy = np.load(energy_gt_name)

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        pitch = torch.from_numpy(pitch)
        energy = torch.from_numpy(energy)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        buffer.append({"text": character,
                       "duration": duration,
                       "mel_target": mel_gt_target,
                       "pitch" : pitch,
                       "energy" : energy    
        })

    end = perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    pitches = [batch[ind]["pitch"] for ind in cut_list]
    energies = [batch[ind]["energy"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    pitches = pad_1D_tensor(pitches)
    energies= pad_1D_tensor(energies)
    mel_targets = pad_2D_tensor(mel_targets)

    out = {"text": texts,
           "mel_target" : mel_targets,
           "duration"   : durations,
           "energy"     : energies,
           "pitch"      : pitches,
           "mel_pos"    : mel_pos,
           "src_pos"    : src_pos,
           "mel_max_len": max_mel_len}

    return out


def get_collator(batch_expand_size):
    def collate_fn_tensor(batch):
        len_arr = np.array([d["text"].size(0) for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = batchsize // batch_expand_size

        cut_list = list()
        for i in range(batch_expand_size):
            cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

        output = list()
        for i in range(batch_expand_size):
            output.append(reprocess_tensor(batch, cut_list[i]))

        return output
    return collate_fn_tensor


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def get_LJSpeech_dataloader(dataset_config):
    buffer = get_data_to_buffer(dataset_config)
    dataset = BufferDataset(buffer)
    return DataLoader(
        dataset,
        batch_size=dataset_config['batch_expand_size'] * dataset_config['batch_size'],
        shuffle=True,
        collate_fn = get_collator(dataset_config['batch_expand_size']),
        drop_last=True,
        num_workers=dataset_config['num_workers']
    )