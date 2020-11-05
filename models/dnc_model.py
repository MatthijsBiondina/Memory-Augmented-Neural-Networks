from typing import List, Dict, Union, Any

import torch
from torch import nn, sigmoid, Tensor, softmax, ByteTensor, IntTensor, LongTensor, prod, index_select, arange
from torch.nn import ParameterList, Parameter
from torch.nn.functional import softplus, relu, elu

import utils.tools as tools
from utils.tools import nan
import utils.config as cfg


class LSTMCell(nn.Module):
    def __init__(self, in_size=cfg.num_units):
        super(LSTMCell, self).__init__()

        self.line_f = nn.Linear(in_size, cfg.num_units)
        self.line_i = nn.Linear(in_size, cfg.num_units)
        self.line_u = nn.Linear(in_size, cfg.num_units)
        self.line_o = nn.Linear(in_size, cfg.num_units)

    def forward(self, X_t, s_tm1=None):
        i_t = sigmoid(self.line_i(X_t))
        f_t = sigmoid(self.line_f(X_t))
        s_t = f_t * s_tm1 + i_t * torch.tanh(self.line_u(X_t))
        o_t = sigmoid(self.line_o(X_t))
        h_t = o_t * torch.tanh(s_t)

        return h_t, s_t


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()

        chi_size = cfg.input_size + cfg.num_read_heads * cfg.memory_size
        rnn_cells = [LSTMCell(chi_size + cfg.num_units)]
        for _ in range(1, cfg.num_layers):
            rnn_cells.append(LSTMCell(chi_size + 2 * cfg.num_units))
        self.rnn_cells = nn.ModuleList(rnn_cells)

        self.line_v = nn.Linear(cfg.num_units * cfg.num_layers, cfg.output_size)
        self.line_xi = nn.Linear(cfg.num_units * cfg.num_layers,
                                 cfg.memory_size * cfg.num_read_heads + 3 * cfg.memory_size + 5 * cfg.num_read_heads + 3)
        # self.line_y = nn.Linear(cfg.memory_size * cfg.num_read_heads, cfg.output_size)

    def forward(self, chi_t, H_tm1, S_tm1):
        H_t, S_t = [torch.empty(0)] * cfg.num_layers, [torch.empty(0)] * cfg.num_layers
        H_t[0], S_t[0] = self.rnn_cells[0](torch.cat((chi_t, H_tm1[0]), dim=1), S_tm1[0])
        for l in range(1, cfg.num_layers):
            H_t[l], S_t[l] = self.rnn_cells[l](torch.cat((chi_t, H_tm1[l], H_t[l - 1]), dim=1))

        v_t = self.line_v(torch.cat(H_t, dim=1))
        xi_t = self.line_xi(torch.cat(H_t, dim=1))

        return v_t, xi_t, H_t, S_t


class DNCCell(nn.Module):

    def __init__(self):
        super(DNCCell, self).__init__()

        self.R_t0 = None
        self.M_t0 = None
        self.__init_learnable_parameters()

        self.controller = Controller()
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
        self.line_read = nn.Linear(cfg.num_read_heads * cfg.memory_size, cfg.memory_size)

    def forward(self, x_t: Tensor, TM1: Dict[str, Union[Tensor, List[Tensor]]] = None):
        TM1 = TM1 if TM1 is not None else self.__init_t0(x_t.size(0))

        # pass through controller
        R_tm1, H_tm1, S_tm1 = TM1['R'], TM1['H'], TM1['S']
        v_t, xi_t, H_t, S_t = self.controller(torch.cat([x_t] + R_tm1, dim=1), H_tm1=H_tm1, S_tm1=S_tm1)

        # extract interface parameters
        K_r_t, Beta_r_t, k_w_t, beta_w_t, e_t, v_t, F_t, g_a_t, g_w_t, PI_t = self._interface_parameters(xi_t)

        """Memory addressing"""
        # Dynamic memory allocation
        W_r_tm1 = TM1['W_r']
        psi_t = self._memory_retention_vector(F_t, W_r_tm1)

        # usage vector
        u_tm1, w_w_tm1 = TM1['u'], TM1['w_w']
        u_t = (u_tm1 + w_w_tm1 - u_tm1 * w_w_tm1) * psi_t

        # allocation weighting
        a_t = self._allocation_weightings(u_t)

        # write weighting
        M_tm1 = TM1['M']
        c_w_t = self._content_based_addressing(M_tm1, k_w_t, beta_w_t)
        w_w_t = g_w_t * (g_a_t * a_t + (1 - g_a_t) * c_w_t)

        # Write to memory
        M_t = M_tm1 * (1 - torch.bmm(w_w_t.unsqueeze(dim=2), e_t.unsqueeze(dim=1))) \
              + torch.bmm(w_w_t.unsqueeze(dim=2), v_t.unsqueeze(dim=1))

        # Temporal memory linkage
        p_tm1, L_tm1 = TM1['p'], TM1['L']
        p_t = (1 - w_w_t.sum(dim=1, keepdims=True)) * p_tm1 + w_w_t  # precedence weighting
        L_t = self._link_matrix(L_tm1, w_w_t, p_tm1)

        # Forward and backward weightings
        W_r_t = [None] * cfg.num_read_heads
        for ii in range(cfg.num_read_heads):
            fwd_i_t = torch.bmm(L_t, W_r_tm1[ii].unsqueeze(dim=2)).squeeze(dim=2)
            bwd_i_t = torch.bmm(L_t.transpose(1, 2), W_r_tm1[ii].unsqueeze(dim=2)).squeeze(dim=2)
            c_r_i_t = self._content_based_addressing(M_t, K_r_t[ii], Beta_r_t[ii])
            W_r_t[ii] = PI_t[ii][:, 0] * bwd_i_t + PI_t[ii][:, 1] * c_r_i_t + PI_t[ii][:, 2] * fwd_i_t

        # Read from memory
        R_t = [None] * cfg.num_read_heads
        for ii in range(cfg.num_read_heads):
            R_t[ii] = torch.bmm(M_t.transpose(1, 2), W_r_t[ii].unsqueeze(dim=2)).squeeze(dim=2)

        # Output
        y_t = v_t + self.line_read(torch.cat(R_t, dim=1))
        nan(y_t)
        return y_t, {'H': H_t, 'L': L_t, 'M': M_t, 'R': R_t, 'S': S_t, 'W_r': W_r_t, 'p': p_t, 'u': u_t, 'w_w': w_w_t}

    def _link_matrix(self, L_tm1: Tensor, w: Tensor, p: Tensor):
        w_i = w.unsqueeze(dim=2)
        w_j = w.unsqueeze(dim=1)
        p_j = p.unsqueeze(dim=1)

        L_t = (1 - w_i - w_j) * L_tm1 + w_i * p_j
        return L_t * (1 - torch.eye(L_t.size(1)).unsqueeze(0).to(self.device))

    def _allocation_weightings(self, u: Tensor):
        # with torch.no_grad():
        #     phi = [None] * u.size(0)
        #     for ii in range(u.size(0)):
        #         phi[ii] = sorted(list(range(u.size(1))), key=lambda x: u[ii][x].item())
        #     phi = LongTensor(phi).to(self.device)
        #
        # phi_a = phi
        # _, phi_b = torch.sort(u, 1)

        _, phi = torch.sort(u, 1)
        bii = arange(u.size(0))
        a = torch.zeros_like(u)
        products = torch.ones(u.size(0), u.size(1) + 1).to(self.device)
        for ii in range(u.size(1)):
            a = a.index_put((bii, phi[:, ii]), (1 - u[bii, phi[:, ii]]) * products[:, ii])
            products = products.index_put((bii, torch.full(bii.size(), ii + 1, dtype=torch.int64)),
                                          products[:, ii] * u[bii, phi[:, ii]])
        return a

    def _memory_retention_vector(self, F: List[Tensor], W: List[Tensor]) -> Tensor:
        psi = torch.ones_like(W[0])
        for f, w in zip(F, W):
            psi *= (1 - w * f)
        return psi

    def _content_based_addressing(self, M: Tensor, k: Tensor, beta: Tensor) -> Tensor:
        k = k.unsqueeze(dim=1).expand_as(M)
        similarity = self.cosine_similarity(k, M)
        attention = softmax(similarity * beta, dim=1)
        return attention

    def _interface_parameters(self, xi):
        # R read keys
        K_r = [None] * cfg.num_read_heads
        for ii in range(cfg.num_read_heads):
            K_r[ii] = xi[:, ii * cfg.memory_size:(ii + 1) * cfg.memory_size]
        xi = xi[:, (ii + 1) * cfg.memory_size:]

        # R read strengths
        Beta_r = [None] * cfg.num_read_heads
        for ii in range(cfg.num_read_heads):
            Beta_r[ii] = 1 + softplus(xi[:, ii]).unsqueeze(dim=1)  # oneplus function
        xi = xi[:, (ii + 1):]

        # 1 write key
        k_w = xi[:, :cfg.memory_size]
        xi = xi[:, cfg.memory_size:]

        # 1 write strength
        beta_w = 1 + softplus(xi[:, 0]).unsqueeze(dim=1)
        xi = xi[:, 1:]

        # 1 erase vector
        e = sigmoid(xi[:, :cfg.memory_size])
        xi = xi[:, cfg.memory_size:]

        # 1 write vector
        v = xi[:, :cfg.memory_size]
        xi = xi[:, cfg.memory_size:]

        # R free gates
        F = [None] * cfg.num_read_heads
        for ii in range(cfg.num_read_heads):
            F[ii] = sigmoid(xi[:, ii]).unsqueeze(dim=1)
        xi = xi[:, (ii + 1):]

        # 1 allocation gate and 1 write gate
        g_a = sigmoid(xi[:, 0]).unsqueeze(dim=1)
        g_w = sigmoid(xi[:, 1]).unsqueeze(dim=1)
        xi = xi[:, 2:]

        # R read modes
        PI = [None] * cfg.num_read_heads
        for ii in range(cfg.num_read_heads):
            PI[ii] = softmax(xi[:, ii * 3:(ii + 1) * 3], dim=1).unsqueeze(dim=2)

        return K_r, Beta_r, k_w, beta_w, e, v, F, g_a, g_w, PI

    def __init_learnable_parameters(self):
        if cfg.init_mode == 'learned':
            self.R_t0 = ParameterList(
                [Parameter(nn.init.xavier_uniform(torch.empty(1, cfg.memory_size)).to(self.device))
                 for _ in range(cfg.num_read_heads)])
            self.M_t0 = Parameter(
                nn.init.xavier_uniform(torch.empty(1, cfg.num_memory_locations, cfg.memory_size)).to(self.device))
        else:
            self.R_t0 = [nn.init.xavier_uniform(torch.empty(1, cfg.memory_size)).to(self.device) * 1e-3
                         for _ in range(cfg.num_read_heads)]
            self.M_t0 = nn.init.xavier_uniform(torch.empty(1, cfg.num_memory_locations, cfg.memory_size)).to(
                self.device) * 1e-3

    def __init_t0(self, bs):
        R_t0 = [r.expand(bs, -1) for r in self.R_t0]
        H_t0 = [torch.zeros(bs, cfg.num_units) for _ in range(cfg.num_layers)]
        S_t0 = [torch.zeros(bs, cfg.num_units) for _ in range(cfg.num_layers)]
        M_t0 = self.M_t0.expand(bs, -1, -1)
        u_t0 = torch.zeros(bs, cfg.num_memory_locations)
        W_r_t0 = [torch.zeros(bs, cfg.num_memory_locations)]
        w_w_t0 = torch.zeros(bs, cfg.num_memory_locations)
        p_t0 = torch.zeros(bs, cfg.num_memory_locations)
        L_t0 = torch.zeros(bs, cfg.num_memory_locations, cfg.num_memory_locations)

        """DEBUG"""
        # u_t0 = sigmoid(nn.init.xavier_uniform(torch.empty_like(u_t0)))
        # W_r_t0 = [softmax(nn.init.xavier_uniform(w), dim=1) for w in W_r_t0]
        # w_w_t0 = softmax(nn.init.xavier_uniform(w_w_t0), dim=1)
        # p_t0 = softmax(nn.init.xavier_uniform(p_t0), dim=1)
        # L_t0 = sigmoid(nn.init.xavier_uniform(L_t0))

        initials = {'R': R_t0, 'H': H_t0, 'S': S_t0, 'M': M_t0, 'u': u_t0, 'W_r': W_r_t0, 'w_w': w_w_t0,
                    'p': p_t0, 'L': L_t0}
        for key in initials:
            if isinstance(initials[key], Tensor):
                initials[key] = initials[key].to(self.device)
            elif isinstance(initials[key], list):
                for ii in range(len(initials[key])):
                    initials[key][ii] = initials[key][ii].to(self.device)
            else:
                raise TypeError(f"Expected types \"Tensor\" or \"list\", but got \"{type(initials)}\"")
        return initials

    @property
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MDNModule(nn.Module):
    def __init__(self):
        super(MDNModule, self).__init__()

        self.line_h = nn.Linear(cfg.num_units, cfg.num_units)
        self.line_alpha = nn.Linear(cfg.num_units, cfg.mdn_mixing_units)
        self.line_mu = nn.Linear(cfg.num_units, cfg.mdn_mixing_units)
        self.line_std = nn.Linear(cfg.num_units, cfg.mdn_mixing_units)

    def forward(self, X):
        h = relu(self.line_h(X))
        alpha = softmax(self.line_alpha(h), dim=1)
        mu = self.line_mu(h)
        std = elu(self.line_std(h)) + 1

        return torch.stack((alpha, mu, std), dim=2)


class DNCModel(nn.Module):
    def __init__(self):
        super(DNCModel, self).__init__()

        self.dnc_cell = DNCCell()

        if cfg.task in ('copy', 'copy_repeat', 'associative_recall'):
            self.line_post = nn.Linear(cfg.memory_size, cfg.output_size)
        if cfg.task is 'stocks':
            self.line_post = nn.Linear(cfg.memory_size, cfg.num_units)
            self.mdn = MDNModule()

    def forward(self, X, mask=None, return_sequence=False):
        y, h = [None] * X.size(2), [None] * (X.size(2) + 1)
        for ii in range(X.size(2)):
            if cfg.task is "stocks" or mask is None or torch.sum(mask[:, :, ii:]) > 0:
                y[ii], h[ii + 1] = self.dnc_cell(X[:, :, ii], h[ii])
                y[ii] = self.line_post(y[ii])
                if cfg.task in ('copy', 'copy_repeat', 'associative_recall'):
                    y[ii] = cfg.output_func(y[ii])
                elif cfg.task in ('stocks',):
                    y[ii] = self.mdn(y[ii])
            else:
                y[ii], h[ii + 1] = torch.zeros_like(y[ii - 1]), h[ii]
        y = torch.stack(y, dim=2)
        nan(y)

        if return_sequence:
            return y, h[1:]
        else:
            return y

    @property
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
