from random import randint, random
import torch

import utils.config as cfg


class CopyRepeatDataset:
    def __init__(self, max_seq_len: int = cfg.max_seq_len, max_repeats: int = cfg.max_repeats):
        self.max_seq_len = max_seq_len
        self.max_repeats = max_repeats
        self.curriculum = cfg.curriculum
        self.curriculum_point = max_seq_len

    def __len__(self):
        return 250

    def __getitem__(self, idx):
        seq_len, repeats = self.max_seq_len, self.max_repeats
        if self.curriculum == "uniform":
            seq_len = randint(1, self.max_seq_len)
            repeats = randint(1, self.max_repeats)
        elif self.curriculum == 'none':
            seq_len = self.max_seq_len
            repeats = self.max_repeats
        elif self.curriculum in ("naive", "prediction_gain_bandit", "prediction_gain_teacher"):
            seq_len, repeats = max(1, self.curriculum_point)
        elif self.curriculum == 'look_back':
            seq_len = self.curriculum_point if random() < 0.9 else randint(1, self.curriculum_point)
            repeats = self.curriculum_point if random() < 0.9 else randint(1, self.curriculum_point)
        elif self.curriculum == 'look_back_and_forward':
            seq_len = self.curriculum_point if random() < 0.8 else randint(1, self.max_seq_len)
            repeats = self.curriculum_point if random() < 0.8 else randint(1, self.max_seq_len)

        # TODO: only show sequence once
        x_sos = torch.FloatTensor([0] * cfg.num_bits_per_vector + [1, 0]).unsqueeze(1)
        x_seq = torch.cat(((torch.rand((cfg.num_bits_per_vector, seq_len)) > 0.5).float(),), dim=1)
        x_seq = torch.cat((x_seq, torch.zeros((2, seq_len))), dim=0)
        x_eos = torch.FloatTensor([0.] * (cfg.num_bits_per_vector + 1) +
                                  [repeats / self.max_repeats]).unsqueeze(1)
        x_out = torch.zeros_like(x_seq)

        x = torch.cat((x_sos, x_seq, x_eos, x_out), dim=1)
        y_eos = torch.FloatTensor([0.] * cfg.num_bits_per_vector + [1., 0.]).unsqueeze(1)
        y = torch.cat((torch.zeros_like(x_sos),
                       torch.zeros_like(x_seq),
                       torch.zeros_like(x_eos)) +
                      (x_seq,) * repeats + (y_eos,), dim=1)[: cfg.num_bits_per_vector + 1]

        m = torch.cat((torch.zeros(cfg.num_bits_per_vector + 1, seq_len + 2),
                       torch.ones((cfg.num_bits_per_vector + 1, seq_len * repeats + 1))),
                      dim=1)

        x_ou = torch.zeros(cfg.num_bits_per_vector + 2, (self.max_repeats * 2) * self.max_seq_len + 2)
        y_ou = torch.zeros(cfg.num_bits_per_vector + 1, (self.max_repeats * 2) * self.max_seq_len + 2)
        m_ou = torch.zeros(cfg.num_bits_per_vector + 1, (self.max_repeats * 2) * self.max_seq_len + 2)

        x_ou[:, :x.size(1)] = x
        y_ou[:, :y.size(1)] = y
        m_ou[:, :m.size(1)] = m
        return x_ou, y_ou, m_ou
