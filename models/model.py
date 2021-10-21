import torch
import torch.nn as nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

from tqdm import tqdm
import config

class Encoder(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int):
        super(Encoder, self).__init__()

        self.feature_size = feature_size
        self.hidden_size = hidden_size

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=config.window_size,
            hop_length=config.hop_size, win_length=config.window_size, window=config.window,
            center=config.center, pad_mode=config.pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=config.sample_rate,
            n_fft=config.window_size, n_mels=feature_size, fmin=config.fmin, fmax=config.fmax, ref=config.ref,
            amin=config.amin, top_db=config.top_db, freeze_parameters=True)

        # Sequence model
        self.rnn = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # shape (bs, segment_samples) -> (bs, 1, seq_len, freq_bins)
        x = self.spectrogram_extractor(waveform)

        # shape (bs, 1, seq_len, feature_size)
        x = self.logmel_extractor(x)

        # shape (bs, seq_len, feature_size)
        x = x.squeeze(1)

        # output shape (bs, time_steps, hidden_size)
        # hidden shape (bs, 1, hidden_size)
        output, (hidden, cell) = self.rnn(x)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size: int, vocab_size: int, hidden_size: int):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # (bs, seq_len, vocab_size) -> (bs, seq_len, hidden_size)
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)

        # Sequence model
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )

        # shape (bs, seq_len, vocab_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_indexes: torch.Tensor, hidden_state: torch.Tensor, cell_state: torch.Tensor) -> torch.Tensor:
        # embeddings shape(bs, hidden_size)
        embeddings = self.dropout(self.embedding_layer(input_indexes))

        # output (bs, 1, hidden_size)
        output, (hidden, cell) = self.rnn(embeddings.unsqueeze(1), (hidden_state, cell_state))

        # output (bs, hidden_size)
        output = output.squeeze(1)

        # shape (bs, vocab_size)
        z = self.fc(output)

        return z,hidden, cell


class EncoderDecoder(nn.Module):
    def __init__(self, feature_size, hidden_size, decoder_input_size, vocab_size):
        super(EncoderDecoder, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.decoder_input_size = decoder_input_size
        self.vocab_size = vocab_size

        self.encoder = Encoder(feature_size=feature_size, hidden_size=hidden_size)
        self.decoder = Decoder(input_size=decoder_input_size, vocab_size=vocab_size, hidden_size=hidden_size)

    def forward(self, data):
        z_hidden, z_cell = self.encoder(data['waveform'])

        # shape (bs, hidden_size)
        input_indexes = data['note_events_ids'][:, 0]

        # shape (bs, seq_len, vocab_size)
        outputs = torch.empty((data['note_events_ids'].shape[0], data['note_events_ids'].shape[1], self.vocab_size))

        for i in tqdm(range(data['note_events_ids'].shape[1])):
            # output shape (bs, vocab_size)
            # z_hidden shape (bs, 1, hidden_size)
            output, z_hidden, z_cell = self.decoder(input_indexes, z_hidden, z_cell)

            if i == 0:
                outputs = output.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, output.unsqueeze(1)), 1)

            # autoregressive trainig
            # input_indexes = output.argmax(1)

            # teacher-forcing
            input_indexes = data['note_events_ids'][:, i]

        return outputs
