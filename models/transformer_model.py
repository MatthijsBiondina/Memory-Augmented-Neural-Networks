import torch
from torch import nn, Tensor
from torch.nn import functional as F
import utils.config as cfg

BATCH_DIM, CHANNEL_DIM, LENGTH_DIM = 0, 1, 2


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()

        self.line_prep = nn.Linear(cfg.input_size, cfg.num_units)
        self.transformer = nn.Transformer(d_model=cfg.num_units, nhead=10, num_encoder_layers=12)
        self.line_post = nn.Linear(cfg.num_units, cfg.output_size)

    def forward(self, src: Tensor, mask=None, return_sequence=False):
        src = src.transpose(1, 2).transpose(0, 1)
        src = self.line_prep(src)
        tgt = torch.zeros_like(src)
        tgt = self.transformer(src, tgt)
        out = cfg.output_func(self.line_post(tgt))
        out = out.transpose(0, 1).transpose(1, 2)
        return out

    @property
    def device(self):
        return self.line_prep.bias.device
