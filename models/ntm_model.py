import torch
from torch import nn, Tensor, FloatTensor
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
    def __init__(self, in_size=(cfg.num_bits_per_vector + 2), ou_size=(cfg.num_bits_per_vector)):
        super(NTMCell, self).__init__()

        # Initialize memory
        self.H_t0, self.C_t0, self.m_t0, self.R_t0, self.A_t0 = (None,) * 5
        self._init_t0()

        self.line_prep = nn.Linear(cfg.memory_size * cfg.num_read_heads, cfg.num_units)
        self.controllers = nn.ModuleList([LSTMCell(in_size=cfg.num_units * 2) for _ in range(cfg.num_layers)])
        self.r_heads = nn.Linear(cfg.num_units,
                                 cfg.num_read_heads * (cfg.memory_size + 1 + 1 + (2 * cfg.conv_shift_range + 1) + 1))
        self.w_heads = nn.Linear(cfg.num_units,
                                 cfg.num_write_heads * (cfg.memory_size + 1 + 1 + (2 * cfg.conv_shift_range + 1) + 1 +
                                                        2 * cfg.memory_size))
        self.line_ou = nn.Linear(cfg.num_units + cfg.num_read_heads * cfg.memory_size, cfg.num_units)

    def _init_t0(self):
        if cfg.init_mode == 'constant':
            self.m_t0 = nn.init.constant(torch.empty(1, cfg.num_memory_locations, cfg.memory_size), 1e-6)
            self.H_t0 = [nn.init.constant(torch.empty(1, cfg.num_units), 1e-6) for _ in range(cfg.num_layers)]
            self.C_t0 = [nn.init.constant(torch.empty(1, cfg.num_units), 1e-6) for _ in range(cfg.num_layers)]
            self.R_t0 = [nn.init.constant(torch.empty(1, cfg.memory_size), 1e-6) for _ in range(cfg.num_read_heads)]
            self.A_t0 = [torch.cat((
                FloatTensor([[100.]]), nn.init.constant(torch.empty(1, cfg.num_memory_locations - 1), 1e-6)), dim=1) for
                _ in
                range(cfg.num_read_heads + cfg.num_write_heads)]
        elif cfg.init_mode == 'random':
            self.m_t0 = nn.init.xavier_normal(torch.empty(1, cfg.num_memory_locations, cfg.memory_size))
            self.H_t0 = [nn.init.xavier_normal(torch.empty(1, cfg.num_units)) for _ in range(cfg.num_layers)]
            self.C_t0 = [nn.init.xavier_normal(torch.empty(1, cfg.num_units)) for _ in range(cfg.num_layers)]
            self.R_t0 = [nn.init.xavier_normal(torch.empty(1, cfg.memory_size)) for _ in
                         range(cfg.num_read_heads)]
            self.A_t0 = [nn.init.xavier_normal(torch.empty(1, cfg.num_memory_locations)) for _ in
                         range(cfg.num_read_heads + cfg.num_write_heads)]
        elif cfg.init_mode == 'learned':
            self.m_t0 = Parameter(nn.init.xavier_uniform(torch.empty(1, cfg.num_memory_locations, cfg.memory_size)))
            self.H_t0 = ParameterList(
                [Parameter(nn.init.xavier_uniform(torch.empty(1, cfg.num_units))) for _ in range(cfg.num_layers)])
            self.C_t0 = ParameterList(
                [Parameter(nn.init.xavier_uniform(torch.empty(1, cfg.num_units))) for _ in range(cfg.num_layers)])
            self.R_t0 = ParameterList(
                [Parameter(nn.init.xavier_uniform(torch.empty(1, cfg.memory_size))) for _ in range(cfg.num_read_heads)])
            self.A_t0 = ParameterList(
                [Parameter(nn.init.xavier_uniform(torch.empty(1, cfg.num_memory_locations)))
                 for _ in range(cfg.num_read_heads + cfg.num_write_heads)])
        else:
            raise ValueError(
                f"Invalid config option for \"init_mode\": {cfg.init_mode} (\"constant\", \"random\", or \"learned\" expected)")

    def forward(self, X, Persistent_tm1=None):
        if Persistent_tm1 is None:
            bs, device = X.size(0), X.device
            H_tm1 = [h.to(device).expand(bs, -1) for h in self.H_t0]
            C_tm1 = [c.to(device).expand(bs, -1) for c in self.C_t0]
            m_tm1 = self.m_t0.to(device).expand(bs, -1, -1)
            R_tm1 = [F.tanh(r).to(device).expand(bs, -1) for r in self.R_t0]
            A_tm1 = [F.softmax(a).to(device).expand(bs, -1) for a in self.A_t0]
        else:
            H_tm1 = Persistent_tm1['H']
            C_tm1 = Persistent_tm1['C']
            m_tm1 = Persistent_tm1['m']
            R_tm1 = Persistent_tm1['R']
            A_tm1 = Persistent_tm1['A']

        h_t = torch.cat(R_tm1, dim=CHANNEL_DIM)
        h_t = self.line_prep(h_t)

        H_t, C_t = [], []
        for ii in range(cfg.num_layers):
            h_t = torch.cat((X, h_t), dim=CHANNEL_DIM)
            h_t, (_, c_t) = self.controllers[ii](h_t, (H_tm1[ii], C_tm1[ii]))
            H_t.append(h_t)
            C_t.append(c_t)

        h_t = h_t.clamp(-cfg.clip_value, cfg.clip_value)

        R_ou = torch.split(self.r_heads(h_t),
                           cfg.memory_size + 1 + 1 + (2 * cfg.conv_shift_range + 1) + 1,
                           dim=CHANNEL_DIM)
        W_ou = torch.split(self.w_heads(h_t),
                           cfg.memory_size + 1 + 1 + (2 * cfg.conv_shift_range + 1) + 1 + 2 * cfg.memory_size,
                           dim=CHANNEL_DIM)

        A_t = [None] * (cfg.num_read_heads + cfg.num_write_heads)
        s_t_debug = []
        for ii, head in enumerate(list(R_ou) + list(W_ou)):
            A_t[ii] = addressing(
                k=torch.tanh(head[:, :cfg.memory_size]),
                beta=F.softplus(head[:, cfg.memory_size]),
                g=torch.sigmoid(head[:, cfg.memory_size + 1]),
                s=torch.softmax(head[:, cfg.memory_size + 2:cfg.memory_size + 2 + 2 * cfg.conv_shift_range + 1], -1),
                gamma=F.softplus(head[:, cfg.memory_size + 2 + 2 * cfg.conv_shift_range + 1]),
                m_tm1=m_tm1,
                a_tm1=A_tm1[ii])
            s_t_debug.append(
                torch.softmax(head[:, cfg.memory_size + 2:cfg.memory_size + 2 + 2 * cfg.conv_shift_range + 1], -1))

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

        y_t = self.line_ou(torch.cat([h_t] + R_t, dim=CHANNEL_DIM)).clamp(-cfg.clip_value, cfg.clip_value)

        nan(y_t)
        return y_t, {'H': H_t, 'C': C_t, 'm': m_t, 'R': R_t, 'A': A_t, 's_debug': s_t_debug}


class NTMModel(nn.Module):
    def __init__(self):
        super(NTMModel, self).__init__()

        self.line_prep = nn.Linear(cfg.input_size, cfg.num_units)
        self.ntm_cell = NTMCell()
        self.line_post = nn.Linear(cfg.num_units, cfg.output_size)

    def forward(self, input, mask=None, return_sequence=False):
        seq, h = [None] * input.size(LENGTH_DIM), [None] * (input.size(LENGTH_DIM) + 1)
        for ii in range(input.size(LENGTH_DIM)):
            if mask is None or torch.sum(mask[:, :, ii:]) > 0:
                x = self.line_prep(input[:, :, ii])
                # x = input[:, :, ii]
                x, h[ii + 1] = self.ntm_cell(x, h[ii])
                seq[ii] = torch.sigmoid(self.line_post(x))
                # seq[ii] = torch.sigmoid(x)
            else:
                seq[ii] = torch.zeros_like(seq[ii - 1])
                h[ii + 1] = h[ii]
        x = torch.stack(seq, dim=LENGTH_DIM)

        if return_sequence:
            return x, h[1:]
        else:
            return x

    @property
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # return self.ntm_cell.controllers[0].device
