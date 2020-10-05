from random import randint

import torch
import utils.config as cfg
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')


class CopyDataset(Dataset):
    def __init__(self, seq_len: int = cfg.max_seq_len):
        self.max_seq_len = cfg.max_seq_len
        self.seq_len = seq_len
        self.curriculum = cfg.curriculum
        self.curriculum_point = seq_len

    def __len__(self):
        return 250

    def __getitem__(self, idx):
        seq_len = self.max_seq_len
        if self.curriculum == "uniform":
            seq_len = randint(1, self.max_seq_len)
        elif self.curriculum == "naive":
            seq_len = self.curriculum_point
        x_sos = torch.FloatTensor([0] * cfg.num_bits_per_vector + [1, 0]).unsqueeze(1)
        x_seq = torch.cat(
            ((torch.rand((cfg.num_bits_per_vector, seq_len)) > 0.5).float(),
             torch.zeros((2, seq_len))), dim=0)
        x_eos = torch.FloatTensor([0] * (cfg.num_bits_per_vector + 1) + [1]).unsqueeze(1)
        x_out = torch.zeros_like(x_seq)

        x = torch.cat((x_sos, x_seq, x_eos, x_out), dim=1)
        y = torch.cat((torch.zeros_like(x_sos),
                       torch.zeros_like(x_seq),
                       torch.zeros_like(x_eos),
                       x_seq), dim=1)[:cfg.num_bits_per_vector]
        m = torch.zeros_like(y)
        m[:cfg.num_bits_per_vector, -x_out.size(1):] = torch.ones_like(x_out[:cfg.num_bits_per_vector, :])

        x_ou = torch.zeros(cfg.num_bits_per_vector + 2, self.max_seq_len * 2 + 2)
        y_ou = torch.zeros(cfg.num_bits_per_vector, self.max_seq_len * 2 + 2)
        m_ou = torch.zeros(cfg.num_bits_per_vector, self.max_seq_len * 2 + 2)

        x_ou[:, :x.size(1)] = x
        y_ou[:, :y.size(1)] = y
        m_ou[:, :m.size(1)] = m
        return x_ou, y_ou, m_ou
