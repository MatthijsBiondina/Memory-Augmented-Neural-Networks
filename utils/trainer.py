from math import ceil

import torch
from torch import optim, Tensor, nn, autograd
from tqdm import tqdm
from torchsummary import summary
import utils.config as cfg
from utils.tools import poem, pyout, nan
from utils.utils import plots, plot_ntm


class TrainLoop:
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.device = model.device
        self.loss_obj = nn.BCELoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.metrics = {'episodes': [], 'loss': []}

    def save(self, path="./res/model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="./res/model.pth"):
        self.model.load_state_dict(torch.load(path))

    def summary(self):
        data_example = self.data.dataset[0]
        plots(data_example)
        summary(self.model, self.data.dataset[0][0].size())

    def fit(self):
        epochs = int(ceil(cfg.num_train_steps / len(self.data.dataset)))
        pbar = tqdm(range(epochs), desc=poem("NaN"))
        for epoch in pbar:
            self.data.dataset.curriculum_point = min(cfg.max_seq_len, int(epoch / epochs * cfg.max_seq_len + 1))
            rloss = 0.
            for X, T, M in self.data:
                X, T, M = X.to(self.device), T.to(self.device), M.to(self.device)

                self.model.zero_grad()

                Y = self.model(X, mask=M)

                loss = self.loss_func(T, Y, M)
                loss.backward()
                self.optimizer.step()
                rloss += loss.item() * X.size(0)

            pbar.desc = poem(rloss / len(self.data.dataset))

    def evaluate(self):
        self.data.dataset.curriculum = "none"
        # self.data.dataset.max_seq_len = 100
        with torch.no_grad():
            rloss = 0.
            for X, T, M in self.data:
                X, T, M = X.to(self.device), T.to(self.device), M.to(self.device)
                Y = self.model(X)
                loss = self.loss_func(T, Y, M)
                rloss += loss.item() * X.size(0)
        pyout(f"Test score: {rloss / len(self.data.dataset)}")
        plots((X[0], T[0], Y[0] * M[0]))

    def visualize(self):
        self.data.dataset.curriculum = "none"
        # self.data.dataset.max_seq_len = 100
        with torch.no_grad():
            for X, T, M in self.data:
                X, T, M = X[:1].to(self.device), T[:1].to(self.device), M[:1].to(self.device)
                Y, H = self.model(X, return_sequence=True)
                plots((X[0], T[0], Y[0] * M[0]))
                plot_ntm(X, Y, H)
                break

    def loss_func(self, y_true: Tensor, y_pred: Tensor, mask: Tensor = None):
        nan(y_pred)
        loss = self.loss_obj(y_pred.clamp(0, 1), y_true)
        loss = loss if mask is None else loss * mask
        loss = torch.mean(loss) if mask is None else torch.sum(loss) / torch.sum(mask)
        return loss
