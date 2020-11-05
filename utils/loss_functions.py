import torch
from torch import Tensor, nn, mean, FloatTensor
import torch.distributions as D
from torch.nn.functional import mse_loss

from utils import config as cfg
from utils.tools import nan
from utils import tools


class Loss:
    def __init__(self):
        self.loss_function = None
        if cfg.task in ('copy', 'copy_repeat', 'associative_recall'):
            self.loss_function = self.bitwise_mse_loss
        elif cfg.task in ('stocks',):
            self.loss_function = self.mdn_loss

    def __call__(self, *args, **kwargs):
        _ = (nan(y) for y in args)
        return self.loss_function(*args, **kwargs)

    def bitwise_mse_loss(self, y_true: Tensor, y_pred: Tensor, mask: Tensor = None, train=True):
        loss = mse_loss(y_pred.clamp(0, 1), y_true, reduction='none')
        loss = loss if mask is None else loss * mask
        if mask is None:
            loss = mean(loss, dim=(1, 2))
        else:
            loss = torch.sum(loss, dim=(1, 2)) / torch.sum(mask, dim=(1, 2))
        if train:
            with torch.no_grad():
                mu, std = loss.mean(), loss.std()
                outliers = torch.logical_or(loss < mu - std, loss > mu + std).type(FloatTensor).to(loss.device)

            loss = (1. - outliers) * loss + outliers * mu
        loss = torch.mean(loss)
        return loss

    def mdn_loss(self, y_true: Tensor, y_pred: Tensor, mask: Tensor = None, train=True):
        alpha = y_pred[:, :, :, 0].transpose(1, 2).reshape(-1, cfg.mdn_mixing_units)
        mu = y_pred[:, :, :, 1].transpose(1, 2).reshape(-1, cfg.mdn_mixing_units)
        std = y_pred[:, :, :, 2].transpose(1, 2).reshape(-1, cfg.mdn_mixing_units) + 1e-6
        gmm = D.MixtureSameFamily(D.Categorical(alpha), D.Normal(mu, std))

        nll = - gmm.log_prob(y_true.view(-1))
        nan(nll)
        if mask is None:
            return torch.mean(nll)
        else:
            return torch.sum(nll * mask.view(-1)) / torch.sum(mask)
