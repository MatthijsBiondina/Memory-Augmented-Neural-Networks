import torch
from random import randint

from torch.utils.data import Dataset

import utils.config as cfg


class AssociativeRecallDataset(Dataset):
    def __init__(self, seq_len: int = cfg.max_seq_len):
        self.max_seq_len = cfg.max_seq_len
        self.max_items = cfg.max_items
        self.seq_len = seq_len
        self.curriculum = cfg.curriculum
        self.curriculum_point = seq_len

    def __len__(self):
        return cfg.batch_size

    def __getitem__(self, idx):
        n_items = self.max_items
        if self.curriculum == "uniform":
            n_items = randint(2, self.max_items)

        x_sos = torch.FloatTensor([0] * cfg.num_bits_per_vector + [1, 0]).unsqueeze(1)
        items = [torch.cat(((torch.rand((cfg.num_bits_per_vector, 3)) > 0.5).float(),
                            torch.zeros((2, 3))), dim=0) for _ in range(n_items)]
        x_eos = torch.FloatTensor([0] * (cfg.num_bits_per_vector + 1) + [1]).unsqueeze(1)
        x_out = torch.zeros(cfg.num_bits_per_vector + 2, 3)
        x = []
        for ii in range(n_items):
            x.append(x_sos)
            x.append(items[ii])
        query_ii = randint(0, n_items - 2)
        x.append(x_eos)
        x.append(items[query_ii])
        x.append(x_eos)

        x = torch.cat(x, dim=1)
        y = torch.cat((torch.zeros_like(x), items[query_ii + 1]), dim=1)[:cfg.num_bits_per_vector]
        m = torch.cat((torch.zeros_like(x), torch.ones_like(x_out)), dim=1)[:cfg.num_bits_per_vector]
        x = torch.cat((x, x_out), dim=1)

        seq_len = 4 * (self.max_items + 2)
        x_ou = torch.zeros(cfg.num_bits_per_vector + 2, seq_len)
        y_ou = torch.zeros(cfg.num_bits_per_vector, seq_len)
        m_ou = torch.zeros(cfg.num_bits_per_vector, seq_len)

        x_ou[:, :x.size(1)] = x
        y_ou[:y.size(0), :y.size(1)] = y
        m_ou[:, :m.size(1)] = m

        return x_ou, y_ou, m_ou
