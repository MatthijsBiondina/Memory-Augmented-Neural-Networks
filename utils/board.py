import os
import matplotlib.cm as cm
import matplotlib as mpl
import torch
from bokeh.plotting import output_file, figure, save
from bokeh.layouts import gridplot
from bokeh.models import LinearColorMapper
from bokeh.palettes import BuPu
import holoviews as hv
import numpy as np
from utils import tools
import json

hv.extension('bokeh')

from utils.tools import makedir
import utils.config as cfg


def not_none(vlist, klist=None):
    if klist is None:
        return [x for x in vlist if x is not None]
    else:
        return [x for x, k in zip(vlist, klist) if k is not None]


class BokehBoard:
    def __init__(self, fname=cfg.experiment_name):
        makedir(os.path.join('.', 'res', 'ou', fname), delete=True)
        self.folder = os.path.join('.', 'res', 'ou', fname)
        self.fpath = os.path.join('.', 'res', 'ou', fname, 'loss.html')
        self.metrics = {'episode': [], 'train_loss': [], 'test_loss': [], 'bit_loss': []}

    def save_metrics(self):
        with open(os.path.join(self.folder, 'metrics.json'), 'w+') as f:
            try:
                json.dump(self.metrics, f)
            except TypeError:
                pass
        try:
            items = [item for item in dir(cfg) if not item.startswith('__')]
            with open(os.path.join(self.folder, 'config.txt'), 'w') as f:
                pad_len = max([len(x) for x in items])
                for item in items:
                    value = eval(f"cfg.{item}")
                    f.write(f"{item}{' ' * (pad_len - len(item))} := {value}\n")
        except:
            pass

    def update_plots(self, epoch, loss_data, vis_data, bit_data=None, steps=None):
        if steps is None:
            fig_loss = self.loss_plot((epoch + 1) * cfg.batch_size, *loss_data)
        else:
            fig_loss = self.loss_plot(steps, *loss_data)
        fig_io, ii = self.io_plot(*vis_data[:2])
        fig_st = self.stock_plot(*vis_data[:2], ii)
        fig_mem = self.mem_plot(vis_data[2], ii)
        # fig_db = self.db_plot(vis_data[2],ii)
        # fig_bit = self.bit_plot((epoch + 1) * cfg.batch_size, bit_data)

        fig = gridplot([[fig_loss, fig_io], [fig_st, fig_mem]])

        self.save_metrics()
        if epoch % cfg.steps_per_eval == 0 or steps is not None:
            output_file(self.fpath, title=self.fpath.split('/')[-2])
            save(fig)
        return

    def loss_plot(self, episode, train_loss, test_loss):
        self.metrics['episode'].append(episode)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['test_loss'].append(test_loss)

        fig = figure(width=950, height=500, title="Train Loss", x_axis_label='episodes', y_axis_label='loss')
        fig.line(not_none(self.metrics['episode'], self.metrics['train_loss']),
                 not_none(self.metrics['train_loss']), legend_label="Train loss",
                 line_color="royalblue", line_width=3, line_alpha=0.75)
        fig.line(not_none(self.metrics['episode'], self.metrics['test_loss']),
                 not_none(self.metrics['test_loss']), legend_label="Test loss",
                 line_color="orchid", line_width=3, line_alpha=0.75)
        return fig

    def stock_plot(self, X, Y, ii):
        fig = figure(width=950, height=500, title="Prediction", x_axis_label='time', y_axis_label='quote')
        x = X[ii][0][1:cfg.max_seq_len * cfg.history_multiplier + 1].cpu().numpy()
        for jj in range(1, x.shape[0]):
            x[jj] += x[jj - 1]

        mu = Y[ii][0][cfg.max_seq_len * cfg.history_multiplier + 2:].cpu().numpy()
        std = Y[ii][1][cfg.max_seq_len * cfg.history_multiplier + 2:].cpu().numpy()
        while np.sum(mu[-1]) == 0:
            mu = mu[:-1]
            std = std[:-1]
        mu += x[-1]

        fig.line(np.arange(0, x.shape[0]),
                 x, line_color="royalblue", line_width=3, line_alpha=0.75)
        fig.line(np.arange(x.shape[0], x.shape[0] + mu.shape[0]),
                 mu, line_color="orchid", line_width=3, line_alpha=0.75)
        fig.varea(np.arange(x.shape[0], x.shape[0] + mu.shape[0]),
                  mu - std, mu + std, color="orchid", alpha=0.25)
        return fig

    def bit_plot(self, episode, bit_loss):
        self.metrics['bit_loss'].append(bit_loss)

        fig = figure(width=950, height=500, title="Bit Loss", x_axis_label='episodes', y_axis_label='bits per sequence')
        fig.line(not_none(self.metrics['episode'], self.metrics['bit_loss']),
                 not_none(self.metrics['bit_loss']), legend_label="Train loss",
                 line_color="royalblue", line_width=3, line_alpha=0.75)
        return fig

    def io_plot(self, X, Y):
        # get index of longest sequence for plotting
        ii = torch.argmax(torch.argmax(X[:, -1, :], dim=1)).detach().cpu().item()

        X = X.detach().clamp(-1,1).cpu().numpy()[ii]
        Y = Y.detach().clamp(-1,1).cpu().numpy()[ii]

        data = np.concatenate((X, Y), axis=0)[::-1].T
        data = [(i, j, data[i, j]) for i in range(data.shape[0]) for j in range(data.shape[1])]
        hmap = hv.HeatMap(data)
        hmap.opts(cmap='BuPu', width=950, height=500, xlabel='t', ylabel='bit')
        hmap = hv.render(hmap)
        return hmap, ii

    def mem_plot(self, H, ii):
        while H[-1] is None:
            H.pop(-1)
        A = [np.zeros((len(H) + 1, cfg.num_memory_locations))
             for _ in range(cfg.num_read_heads + cfg.num_write_heads)]
        for t in range(len(H)):
            try:
                for a_plt, a_h in zip(A, H[t]['A'][::-1]):
                    a_h = a_h.detach().cpu().numpy()[ii]
                    a_plt[t, :] = a_h
            except:
                pass
            try:
                A[0][t, :] = H[t]['w_w'][ii].detach().cpu().numpy()
                for a_plt, a_h in zip(A[1:], H[t]['W_r']):
                    a_plt[t, :] = a_h.detach().cpu().numpy()[ii]
            except:
                pass

        for a_plt in A:
            a_plt[-1, :] = np.ones(cfg.num_memory_locations)
        data = np.concatenate(A, axis=0)
        data = [(i, j, data[i, j]) for i in range(data.shape[0]) for j in range(data.shape[1])]
        hmap = hv.HeatMap(data)
        hmap.opts(cmap='BuPu', width=950, height=500, xlabel='t', ylabel='memory location')
        hmap = hv.render(hmap)
        return hmap

    def db_plot(self, H, ii):
        # return None
        while H[-1] is None:
            H.pop(-1)
        C = [np.zeros((len(H) + 1, cfg.conv_shift_range * 2 + 1))
             for _ in range(cfg.num_read_heads + cfg.num_write_heads)]
        for t in range(len(H)):
            for c_plt, c_h in zip(C, H[t]['s_debug'][::-1]):
                c_h = c_h.detach().cpu().numpy()[ii]
                c_plt[t, :] = c_h[::-1]
        for c_plt in C:
            c_plt[-1, :] = np.ones(cfg.conv_shift_range * 2 + 1)
        data = np.concatenate(C, axis=0)
        data = [(i, j, data[i, j]) for i in range(data.shape[0]) for j in range(data.shape[1])]
        hmap = hv.HeatMap(data)
        hmap.opts(cmap='BuPu', width=950, height=500, xlabel='t', ylabel='memory location')
        hmap = hv.render(hmap)
        return hmap
