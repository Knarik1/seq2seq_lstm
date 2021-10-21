import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from transformers.optimization import Adafactor
from utils import config

from utilities import get_filename, create_folder
from utils.data_generator import MaestroDataset, Sampler, collate_fn

from models.model import EncoderDecoder
import argparse

def train(args):
	# Arugments & parameters
	workspace = args.workspace
	batch_size = args.batch_size
	device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

	clip = 1
	criterion = nn.CrossEntropyLoss()
	filename = args.filename
	sample_rate = config.sample_rate
	segment_seconds = config.segment_seconds
	hop_seconds = config.hop_seconds
	frames_per_second = config.frames_per_second

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

	# model
	model = LitEncoderDecoder(
		feature_size=229,
		hidden_size=512,
		decoder_input_size=512,
		vocab_size=train_dataset.vocab_size
	)

	# training
	trainer = pl.Trainer()
	trainer.fit(model, train_loader, valid_loader)


class LitEncoderDecoder(pl.LightningModule):
	def __init__(self, feature_size, hidden_size, decoder_input_size, vocab_size):
		super().__init__()
		self.model = EncoderDecoder(feature_size=feature_size, hidden_size=hidden_size, decoder_input_size=decoder_input_size, vocab_size=vocab_size)

	def forward(self, x):
		outputs = self.model(x)
		return outputs

	def configure_optimizers(self):
		optimizer = Adafactor(
			self.parameters(),
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

		return optimizer

	def training_step(self, train_batch, batch_idx):
		outputs = self(train_batch)

		targets = train_batch['note_events_ids']

		# print("output dim", outputs.shape)
		# print("targets dim", targets.shape)

		# reshape
		outputs = torch.reshape(outputs, (-1, outputs.shape[2]))
		targets = torch.reshape(targets, (-1,))

		loss = criterion(outputs, targets)

		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		outputs = self(val_batch)
		targets = val_batch['note_events_ids']

		# print("output dim", outputs.shape)
		# print("targets dim", targets.shape)

		# reshape
		outputs = torch.reshape(outputs, (-1, outputs.shape[2]))
		targets = torch.reshape(targets, (-1,))

		loss = criterion(outputs, targets)

		self.log('val_loss', loss)

if __name__ == "__main__":

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




