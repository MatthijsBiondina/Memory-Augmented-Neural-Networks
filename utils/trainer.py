from math import ceil
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils.config as cfg
from utils.loss_functions import Loss
from utils.board import BokehBoard
from utils.tools import poem, pyout
from utils import tools

class TrainLoop:
    def __init__(self, data, model, eval_data=None):
        self.data = data
        self.eval_data: DataLoader = eval_data if eval_data is not None else data
        self.model = model
        self.device = model.device
        self.loss_func = Loss()
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

            train_loss = 0.
            for ii, (X, T, M) in enumerate(self.data):
                X, T, M = X.to(self.device), T.to(self.device), M.to(self.device)

                self.model.zero_grad()

                Y, H = self.model(X, mask=M, return_sequence=True)

                loss = self.loss_func(T, Y, M)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                train_loss += loss.item() * X.size(0)

                pbar.desc = poem(train_loss / (ii + 1))
                pbar.update(0)
                if ii % cfg.steps_per_eval == 0:
                    with torch.no_grad():
                        for X, T, M in self.eval_data:
                            X, T, M = X.to(self.device), T.to(self.device), M.to(self.device)
                            Y, H = self.model(X, mask=M, return_sequence=True)
                            loss = self.loss_func(T, Y, M)
                            break

                    self.board.update_plots(
                        epoch,
                        loss_data=(train_loss / (cfg.steps_per_eval * cfg.batch_size), loss.item()),
                        vis_data=(X, Y, H),
                        steps=epoch * len(self.data.dataset) + ii * cfg.batch_size,
                        target=T,
                        mask=M)

                    train_loss = 0.

            if cfg.curriculum == "prediction_gain":
                if train_loss / len(self.data.dataset) < 0.01:
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

    def visualize(self):
        self.data.dataset.curriculum = "none"
        with torch.no_grad():
            for X, T, M in self.data:
                X, T, M = X[:1].to(self.device), T[:1].to(self.device), M[:1].to(self.device)
                Y, H = self.model(X, return_sequence=True)

                return (X.detach().cpu().numpy()[0], Y.detach().cpu().numpy()[0], H)

    # def loss_func(self, y_true: Tensor, y_pred: Tensor, mask: Tensor = None, train=True):
    #     nan(y_pred)
    #     if cfg.task in ('copy', 'copy_repeat', 'associative_recall'):
    #         loss = self.loss_obj(y_pred.clamp(0, 1), y_true)
    #         loss = loss if mask is None else loss * mask
    #         # loss = torch.mean(loss) if mask is None else torch.sum(loss) / torch.sum(mask)
    #         loss = (torch.mean(loss, dim=(1, 2)) if mask is None
    #                 else torch.sum(loss, dim=(1, 2)) / torch.sum(mask, dim=(1, 2)))
    #         if train:
    #             with torch.no_grad():
    #                 mu, std = loss.mean(), loss.std()
    #                 outliers = torch.logical_or(loss < mu - std, loss > mu + std).type(FloatTensor).to(loss.device)
    #
    #             loss = (1. - outliers) * loss + outliers * mu
    #         loss = torch.mean(loss)
    #         return loss
    #     elif cfg.task == 'stocks':
    #         mu, std = y_pred[:, 0].unsqueeze(dim=1), y_pred[:, 1].unsqueeze(dim=1)
    #         prob = 1 / torch.sqrt(2 * pi * torch.square(torch.ones_like(std)) + 1e-6) \
    #                * torch.exp(-torch.square(y_true - mu) / (2 * torch.square(torch.ones_like(std)) + 1e-6))
    #         loss = -prob * mask
    #         loss = (torch.mean(loss, dim=(1, 2)) if mask is None
    #                 else torch.sum(loss, dim=(1, 2)) / torch.sum(mask, dim=(1, 2)))
    #         loss = torch.mean(loss)
    #
    #         return loss

    # def bit_loss(self, y_true: Tensor, y_pred: Tensor, mask: Tensor = None):
    #     y_bit_true, y_bit_pred = (y_true > 0.5) + 0., (y_pred > 0.5) + 0.
    #     loss = nn.L1Loss(reduction='none')(y_bit_true, y_bit_pred)
    #     loss = loss if mask is None else loss * mask
    #     loss = torch.sum(loss, dim=(1, 2))
    #     loss = torch.mean(loss, dim=0)
    #     return loss
