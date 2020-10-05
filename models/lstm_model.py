import torch
import torch.nn as nn
from torch.nn import Parameter

import utils.config as cfg

BATCH_DIM, CHANNEL_DIM, LENGTH_DIM = 0, 1, 2


class LSTMCell(nn.Module):
    def __init__(self, in_size=cfg.num_units):
        super(LSTMCell, self).__init__()
        self.states_t0 = (Parameter(torch.randn(cfg.num_units)), Parameter(torch.randn(cfg.num_units)))

        self.line_forget = nn.Linear(in_size + cfg.num_units, cfg.num_units)
        self.line_input = nn.Linear(in_size + cfg.num_units, cfg.num_units)
        self.line_update = nn.Linear(in_size + cfg.num_units, cfg.num_units)
        self.line_output = nn.Linear(in_size + cfg.num_units, cfg.num_units)

    def forward(self, inputs, states=None):
        h_tm1 = self.states_t0[0].to(inputs.device).expand_as(inputs) if states is None else states[0]
        c_tm1 = self.states_t0[1].to(inputs.device).expand_as(inputs) if states is None else states[1]

        x = torch.cat((h_tm1, inputs), axis=CHANNEL_DIM)
        x_i = torch.sigmoid(self.line_input(x))
        x_f = torch.sigmoid(self.line_forget(x))
        x_u = torch.tanh(self.line_update(x))
        x_o = torch.sigmoid(self.line_output(x))

        c_t = x_f * c_tm1 + x_i * x_u
        h_t = x_o * torch.tanh(c_t)

        return h_t, (h_t, c_t)


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        self.dens_prep = nn.Conv1d(cfg.num_bits_per_vector + 1, cfg.num_units, 1)
        self.lstm_cell = LSTMCell()
        self.dens_post = nn.Conv1d(cfg.num_units, cfg.num_bits_per_vector, 1)

    def forward(self, input):
        x = self.dens_prep(input)

        seq, h = [None] * x.size(LENGTH_DIM), None
        for ii in range(x.size(LENGTH_DIM)):
            seq[ii], h = self.lstm_cell(x[:, :, ii], h)
        x = torch.stack(seq, dim=LENGTH_DIM)
        x = torch.sigmoid(self.dens_post(x))
        return x

    @property
    def device(self):
        return self.dens_prep.bias.device
