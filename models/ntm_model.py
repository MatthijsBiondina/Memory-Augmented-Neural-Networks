import torch
from torch import nn, Tensor
from torch.nn import Parameter, ParameterList
from torch.nn import functional as F

import utils.config as cfg
from models.lstm_model import LSTMCell
from utils.tools import nan

BATCH_DIM, CHANNEL_DIM, LENGTH_DIM = 0, 1, 2


def addressing(k, beta: Tensor, g, s, gamma, m_tm1, a_tm1):
    # Cosine Similarity
    k = torch.unsqueeze(k, dim=2)
    inner_product = m_tm1.matmul(k)
    k_norm = torch.sqrt(torch.sum(torch.square(k), dim=1, keepdim=True))
    m_norm = torch.sqrt(torch.sum(torch.square(m_tm1), dim=2, keepdim=True))
    K = torch.squeeze(inner_product / (k_norm * m_norm + 1e-8), dim=-1)

    # Sharpen
    K_exp = torch.exp(beta.unsqueeze(dim=CHANNEL_DIM) * K)
    w_c = K_exp / torch.sum(K_exp, dim=CHANNEL_DIM, keepdim=True)

    # Interpolation
    g = torch.unsqueeze(g, dim=CHANNEL_DIM)
    w_g = g * w_c + (1. - g) * a_tm1

    # Convolutional shift
    s = torch.cat((s[:, :cfg.conv_shift_range + 1],
                   torch.zeros(s.size(0), cfg.num_memory_locations - (2 * cfg.conv_shift_range + 1)).to(s.device),
                   s[:, -cfg.conv_shift_range:]), dim=CHANNEL_DIM)
    t = torch.cat((torch.flip(s, dims=(CHANNEL_DIM,)), torch.flip(s, dims=(CHANNEL_DIM,))), dim=CHANNEL_DIM)
    s_matrix = torch.stack(
        [t[:, cfg.num_memory_locations - ii - 1:cfg.num_memory_locations * 2 - ii - 1]
         for ii in range(cfg.num_memory_locations)], dim=1)
    w_conv = torch.sum(torch.unsqueeze(w_g, dim=1) * s_matrix, dim=2)

    # Sharpen
    w_sharp = w_conv.pow(gamma.unsqueeze(dim=1))
    w = w_sharp / torch.sum(w_sharp, dim=1, keepdim=True)

    nan(w)
    return w


class NTMCell(nn.Module):
    def __init__(self):
        super(NTMCell, self).__init__()

        x = nn.init.xavier_uniform(torch.empty(1, cfg.num_memory_locations))
        a = Parameter(F.softmax(x, 1))

        self.h_t0 = Parameter(nn.init.xavier_uniform(torch.empty(1, cfg.num_units)))
        self.c_t0 = Parameter(nn.init.xavier_uniform(torch.empty(1, cfg.num_units)))
        self.m_t0 = nn.init.constant(torch.empty(1, cfg.num_memory_locations, cfg.memory_size), 1e-6)
        self.R_t0 = ParameterList([Parameter(nn.init.xavier_uniform(torch.empty(1, cfg.memory_size)))
                                   for _ in range(cfg.num_read_heads)])
        self.A_t0 = ParameterList(
            [Parameter(nn.init.xavier_uniform(torch.empty(1, cfg.num_memory_locations)))
             for _ in range(cfg.num_read_heads + cfg.num_write_heads)])

        self.line_in = nn.Linear(cfg.memory_size * cfg.num_read_heads, cfg.num_units)
        self.controller = LSTMCell(in_size=cfg.num_units * 2)
        self.r_heads = nn.Linear(cfg.num_units,
                                 cfg.num_read_heads * (cfg.memory_size + 1 + 1 + (2 * cfg.conv_shift_range + 1) + 1))
        self.w_heads = nn.Linear(cfg.num_units,
                                 cfg.num_write_heads * (cfg.memory_size + 1 + 1 + (2 * cfg.conv_shift_range + 1) + 1 +
                                                        2 * cfg.memory_size))
        self.line_ou = nn.Linear(cfg.num_units + cfg.num_read_heads * cfg.memory_size, cfg.num_units)

    def forward(self, X, H_tm1=None):
        h_tm1 = self.h_t0.to(X.device).expand(X.size(0), -1) if H_tm1 is None else H_tm1[0]
        c_tm1 = self.c_t0.to(X.device).expand(X.size(0), -1) if H_tm1 is None else H_tm1[1]
        m_tm1 = self.m_t0.to(X.device).expand(X.size(0), -1, -1) if H_tm1 is None else H_tm1[2]
        R_tm1 = [x_.to(X.device).expand(X.size(0), -1) for x_ in self.R_t0] \
            if H_tm1 is None else H_tm1[3:3 + cfg.num_read_heads]
        A_tm1 = [torch.softmax(x_.to(X.device), 1).expand(X.size(0), -1) for x_ in self.A_t0] \
            if H_tm1 is None else H_tm1[-(cfg.num_read_heads + cfg.num_write_heads):]

        h = torch.cat(R_tm1, dim=CHANNEL_DIM)
        h = self.line_in(h)
        h = torch.cat((X, h), dim=CHANNEL_DIM)
        h_t, (_, c_t) = self.controller(h, (h_tm1, c_tm1))

        R_ou = torch.split(self.r_heads(h_t),
                           cfg.memory_size + 1 + 1 + (2 * cfg.conv_shift_range + 1) + 1,
                           dim=CHANNEL_DIM)
        W_ou = torch.split(self.w_heads(h_t),
                           cfg.memory_size + 1 + 1 + (2 * cfg.conv_shift_range + 1) + 1 + 2 * cfg.memory_size,
                           dim=CHANNEL_DIM)

        A_t = [None] * (cfg.num_read_heads + cfg.num_write_heads)
        for ii, head in enumerate(list(R_ou) + list(W_ou)):
            A_t[ii] = addressing(
                k=torch.tanh(head[:, :cfg.memory_size]),
                beta=F.softplus(head[:, cfg.memory_size]),
                g=torch.sigmoid(head[:, cfg.memory_size + 1]),
                s=torch.softmax(head[:, cfg.memory_size + 2:cfg.memory_size + 2 + 2 * cfg.conv_shift_range + 1], -1),
                gamma=F.softplus(head[:, cfg.memory_size + 2 + 2 * cfg.conv_shift_range + 1]),
                m_tm1=m_tm1,
                a_tm1=A_tm1[ii])

        # Reading

        r_attn = A_t[:cfg.num_read_heads]
        R_t = [None] * cfg.num_read_heads
        for ii in range(cfg.num_read_heads):
            R_t[ii] = torch.sum(r_attn[ii].unsqueeze(dim=2) * m_tm1, dim=1)

        # Writing

        w_attn = A_t[cfg.num_read_heads:]
        w_del = [w[:, -2 * cfg.memory_size:-1 * cfg.memory_size] for w in W_ou]
        w_add = [w[:, -1 * cfg.memory_size:] for w in W_ou]
        m_t = m_tm1
        for ii in range(cfg.num_write_heads):
            w = w_attn[ii].unsqueeze(dim=2)
            del_vec = torch.sigmoid(w_del[ii]).unsqueeze(dim=1)
            add_vec = torch.tanh(w_add[ii]).unsqueeze(dim=1)

            m_t = m_t * (1. - torch.matmul(w, del_vec)) + torch.matmul(w, add_vec)
        # m_t = torch.sigmoid(m_t*10-5)-torch.sigmoid(m_t*-10-5)

        y_t = self.line_ou(torch.cat([h_t] + R_t, dim=CHANNEL_DIM)).clamp(-cfg.clip_value, cfg.clip_value)

        nan(y_t)
        return y_t, [h_t, c_t, m_t] + R_t + A_t


class NTMModel(nn.Module):
    def __init__(self):
        super(NTMModel, self).__init__()

        self.line_prep = nn.Linear(cfg.input_size, cfg.num_units)
        self.ntm_cell = NTMCell()
        self.line_post = nn.Linear(cfg.num_units, cfg.output_size)

    def forward(self, input, mask=None, return_sequence=False):
        seq, h = [None] * input.size(LENGTH_DIM), [None] * (input.size(LENGTH_DIM) + 1)
        for ii in range(input.size(LENGTH_DIM)):
            if mask is None or torch.sum(mask[:, ii:]) > 0:
                x = self.line_prep(input[:, :, ii])
                x, h[ii + 1] = self.ntm_cell(x, h[ii])
                seq[ii] = torch.sigmoid(self.line_post(x))
            else:
                seq[ii] = torch.zeros_like(seq[ii - 1])
                h[ii] = h[ii - 1]
        x = torch.stack(seq, dim=LENGTH_DIM)

        if return_sequence:
            return x, h[1:]
        else:
            return x

    @property
    def device(self):
        return self.line_prep.bias.device
