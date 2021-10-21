import os
import argparse

import numpy as np

import torch
import torch.nn as nn
from models.model import Encoder, Decoder, EncoderDecoder

from utilities import get_filename, create_folder
from utils.data_generator import MaestroDataset, Sampler, collate_fn

from utils import config
from pytorch_utils import move_data_to_device
from transformers.optimization import Adafactor
import time
from tqdm import tqdm


def train(args):
    # Arugments & parameters
    workspace = args.workspace
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    NOTE_CHECKPOINT_PATH = args.checkpoint

    filename = args.filename

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 1
    learning_rate = 0.001


    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename)
    create_folder(checkpoints_dir)

    # Dataset
    train_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir,
        segment_seconds=segment_seconds, frames_per_second=frames_per_second,
        max_note_shift=0, augmentor=None)

    # Sampler for training
    train_sampler = Sampler(hdf5s_dir=hdf5s_dir, split='train',
        segment_seconds=segment_seconds, hop_seconds=hop_seconds,
        batch_size=batch_size, mini_data=None)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_sampler=train_sampler, collate_fn=collate_fn,
        num_workers=1, pin_memory=True)

    # Dataset
    valid_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir,
        segment_seconds=segment_seconds, frames_per_second=frames_per_second,
        max_note_shift=0, augmentor=None)

    # Sampler for validation
    valid_sampler = Sampler(hdf5s_dir=hdf5s_dir, split='validation',
        segment_seconds=segment_seconds, hop_seconds=hop_seconds,
        batch_size=batch_size, mini_data=None)

    # Dataloader
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
        batch_sampler=valid_sampler, collate_fn=collate_fn,
        num_workers=1, pin_memory=True)

    model = EncoderDecoder(feature_size=229, hidden_size=512, decoder_input_size=512, vocab_size=train_dataset.vocab_size)

    clip = 1
    criterion = nn.CrossEntropyLoss()

    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    N_EPOCHS = 2

    for epoch in range(N_EPOCHS):
        ################## train #####################
        epoch_loss = 0
        counter = 1
        model.train()
        for batch_data_dict in train_loader:

            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            outputs = model(batch_data_dict)
            targets = batch_data_dict['note_events_ids']

            # print("output dim", outputs.shape)
            # print("targets dim", targets.shape)

            # reshape
            outputs = torch.reshape(outputs, (-1, outputs.shape[2]))
            targets = torch.reshape(targets, (-1,))

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()
            counter += 1
            print(loss.item())

        print("Train mean loss", epoch_loss / counter)


        ################## valid #####################   
        model.eval()
        epoch_loss_valid = 0
        counter_valid = 1

        with torch.no_grad():

            for batch_data_dict_valid in valid_loader:

                for key in batch_data_dict_valid.keys():
                    batch_data_dict_valid[key] = move_data_to_device(batch_data_dict_valid[key], device)

                outputs = model(batch_data_dict_valid)
                targets = batch_data_dict_valid['note_events_ids']

                loss = criterion(outputs, targets)

                epoch_loss_valid += loss.item()
                counter_valid += 1

            print("Train mean loss", epoch_loss_valid / counter_valid)

def test(args):
    ...

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--checkpoint', type=str, required=False)
    # parser_train.add_argument('--max_note_shift', type=int, required=True)
    parser_train.add_argument('--batch_size', type=int, required=False, default=1)
    parser_train.add_argument('--learning_rate', type=float, required=False, default=10e-3)
    parser_train.add_argument('--reduce_iteration', type=int, required=False)
    parser_train.add_argument('--resume_iteration', type=int, required=False)
    parser_train.add_argument('--early_stop', type=int, required=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--model_path', type=str, required=True)
    parser_test.add_argument('--batch_size', type=int, required=True)
    parser_test.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise Exception('Error argument!')
