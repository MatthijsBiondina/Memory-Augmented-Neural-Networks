import sys
from math import ceil

import torch
from torch import optim, Tensor, nn, autograd, FloatTensor
from tqdm import tqdm
from torchsummary import summary
import utils.config as cfg
from utils.board import BokehBoard
from utils.tools import poem, pyout, nan
from utils.utils import plots, plot_ntm


class TrainLoop:
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.device = model.device
        self.loss_obj = nn.MSELoss(reduction='none')
        if cfg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        elif cfg.optimizer == 'RMSProp':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=cfg.learning_rate, momentum=0.9)
        self.board = BokehBoard()

    def save(self, path="./res/model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="./res/model.pth"):
        self.model.load_state_dict(torch.load(path))

    def summary(self):
        data_example = self.data.dataset[0]
        # plots(data_example)
        # summary(self.model, (self.data.dataset[0][0].size()[0], 1), depth=99, verbose=1,
        #         col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"))
        # sys.exit(0)

    def fit(self):
        # torch.autograd.set_detect_anomaly(True)
        epochs = int(ceil(cfg.num_train_steps / len(self.data.dataset)))
        pbar = tqdm(range(epochs), desc=poem("NaN"))
        self.data.dataset.curriculum_point = 1
        for epoch in pbar:
            self.data.dataset.curriculum = cfg.curriculum

            rloss, bloss = 0., 0.
            for X, T, M in self.data:
                X, T, M = X.to(self.device), T.to(self.device), M.to(self.device)

                self.model.zero_grad()

                Y, H = self.model(X, mask=M, return_sequence=True)

                loss = self.loss_func(T, Y, M)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                rloss += loss.item() * X.size(0)
                bloss += self.bit_loss(T, Y, M).item() * X.size(0)

            pbar.desc = poem(rloss / len(self.data.dataset))

            self.board.update_plots(epoch,
                                    loss_data=(rloss / len(self.data.dataset), None),
                                    vis_data=(X, Y, H),
                                    bit_data=bloss / len(self.data.dataset))
            if cfg.curriculum == "prediction_gain":
                if rloss / len(self.data.dataset) < 0.01:
                    self.data.dataset.curriculum_point += 1
                self.data.dataset.curriculum_point = min(self.data.dataset.curriculum_point, cfg.max_seq_len)
            else:
                self.data.dataset.curriculum_point = min(cfg.max_seq_len, int(epoch / epochs * cfg.max_seq_len + 1))

    def evaluate(self):
        self.data.dataset.curriculum = "none"
        with torch.no_grad():
            rloss = 0.
            for X, T, M in self.data:
                X, T, M = X.to(self.device), T.to(self.device), M.to(self.device)
                Y = self.model(X)
                loss = self.loss_func(T, Y, M)
                rloss += loss.item() * X.size(0)
        pyout(f"Test score: {rloss / len(self.data.dataset)}")
        # plots((X[0], T[0], Y[0]))

    def visualize(self):
        self.data.dataset.curriculum = "none"
        with torch.no_grad():
            for X, T, M in self.data:
                X, T, M = X[:1].to(self.device), T[:1].to(self.device), M[:1].to(self.device)
                Y, H = self.model(X, return_sequence=True)

                return (X.detach().cpu().numpy()[0], Y.detach().cpu().numpy()[0], H)

    def loss_func(self, y_true: Tensor, y_pred: Tensor, mask: Tensor = None, train=True):
        nan(y_pred)
        loss = self.loss_obj(y_pred.clamp(0, 1), y_true)
        loss = loss if mask is None else loss * mask
        # loss = torch.mean(loss) if mask is None else torch.sum(loss) / torch.sum(mask)
        loss = (torch.mean(loss, dim=(1, 2)) if mask is None
                else torch.sum(loss, dim=(1, 2)) / torch.sum(mask, dim=(1, 2)))
        if train:
            with torch.no_grad():
                mu, std = loss.mean(), loss.std()
                outliers = torch.logical_or(loss < mu - std, loss > mu + std).type(FloatTensor).to(loss.device)

            loss = (1. - outliers) * loss + outliers * mu
        loss = torch.mean(loss)
        return loss

    def bit_loss(self, y_true: Tensor, y_pred: Tensor, mask: Tensor = None):
        y_bit_true, y_bit_pred = (y_true > 0.5) + 0., (y_pred > 0.5) + 0.
        loss = nn.L1Loss(reduction='none')(y_bit_true, y_bit_pred)
        loss = loss if mask is None else loss * mask
        loss = torch.sum(loss, dim=(1, 2))
        loss = torch.mean(loss, dim=0)
        return loss
