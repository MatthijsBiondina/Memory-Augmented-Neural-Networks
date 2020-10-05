import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import utils.config as cfg

BATCH_DIM, CHANNEL_DIM, LENGTH_DIM = 0, 1, 2


def plots(X):
    sns.set(style='white', palette='muted', color_codes=True)

    f, axs = plt.subplots(len(X), 1, figsize=(7, 7))
    axs = axs if len(X) > 1 else [axs]

    sns.despine(left=True)

    for ii, x_ in enumerate(X):
        x = x_.cpu().numpy()
        sns.heatmap(x, vmin=0, vmax=1, ax=axs[ii])

    plt.show()


def activation_plot(x):
    x = x.cpu().numpy()
    f, ax = plt.subplots(1, 1, figsize=(x.shape[1] / 5, x.shape[0] / 5))
    sns.heatmap(x, vmin=-1, vmax=1, ax=ax, cbar=False, xticklabels=False, yticklabels=False)

    return f


def plot_ntm(X, Y, H):
    a = np.zeros((cfg.num_read_heads + cfg.num_write_heads, len(H), cfg.num_memory_locations))
    head = cfg.num_read_heads + cfg.num_write_heads

    for t in range(len(H)):
        for h_ii in range(cfg.num_read_heads + cfg.num_write_heads):
            a[h_ii, t] = H[t][-h_ii - 1].cpu().numpy()

    f, axs = plt.subplots(1, a.shape[0], figsize=(head * a.shape[1] / 5, a.shape[2] / 5))
    for ii, ax in enumerate(axs):
        sns.heatmap(a[ii].T, vmin=0, vmax=1, ax=ax, cbar=False, xticklabels=False, yticklabels=False)

    plt.show()

    return