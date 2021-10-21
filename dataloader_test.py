import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import time
import logging

import torch.utils.data
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


from utils.data_generator_old  import MaestroDataset, Augmentor, Sampler, collate_fn
from pytorch_utils import move_data_to_device
from utils import config

aim_recording = False
workspace = "./workspaces"
batch_size=1

sample_rate = config.sample_rate
segment_seconds = config.segment_seconds
hop_seconds = config.hop_seconds
segment_samples = int(segment_seconds * sample_rate)
frames_per_second = config.frames_per_second
classes_num = config.classes_num
num_workers = 1
augmentation = 'none'
mini_data = False
max_note_shift = 0
device = 'cpu'


sample_rate = 16000
window_size = 2048
hop_size = sample_rate // 100
mel_bins = 229
fmin = 30
fmax = sample_rate // 2

window = 'hann'
center = True
pad_mode = 'reflect'
ref = 1.0
amin = 1e-10
top_db = None

midfeat = 1792
momentum = 0.01


if augmentation == 'none':
    augmentor = None
elif augmentation == 'aug':
    augmentor = Augmentor()
else:
    raise Exception('Incorrect argumentation!')

hdf5s_dir = os.path.join(workspace, 'hdf5s')

train_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir,
                               segment_seconds=segment_seconds, frames_per_second=frames_per_second,
                               max_note_shift=max_note_shift, augmentor=augmentor)

print(type(train_dataset))


train_sampler = Sampler(hdf5s_dir=hdf5s_dir, split='train',
                        segment_seconds=segment_seconds, hop_seconds=hop_seconds,
                        batch_size=batch_size, mini_data=mini_data)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_sampler=train_sampler, collate_fn=collate_fn,
                                           num_workers=num_workers, pin_memory=True)

spectrogram_extractor = Spectrogram(n_fft=window_size,
                                    hop_length=hop_size, win_length=window_size, window=window,
                                    center=center, pad_mode=pad_mode, freeze_parameters=True)

logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                         n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref,
                                         amin=amin, top_db=top_db, freeze_parameters=True)


print(type(train_loader))

# optimizer = optim.Adam(model.parameters(), lr=learning_rate,
#                        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

iteration = 0

for batch_data_dict in train_loader:
    # Evaluation
    if iteration % 5000 == 0:  # and iteration > 0:
        logging.info('------------------------------------')
        logging.info('Iteration: {}'.format(iteration))

        train_fin_time = time.time()

    # Move data to device
    print("batch_data_dict.keys()", batch_data_dict.keys())
    for key in batch_data_dict.keys():
        batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        print(batch_data_dict['waveform'].shape)
        print("_________________________")
        x = spectrogram_extractor.forward(audio)   # (batch_size, 1, time_steps, freq_bins)
        x = logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
