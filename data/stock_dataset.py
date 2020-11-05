import torch
from sklearn import preprocessing
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import utils.tools as tools
from utils.tools import listdir, poem
import utils.config as cfg
from utils.utils import sample_seq_len


class StockDataset(Dataset):
    def __init__(self, seq_len: int = cfg.max_seq_len, mode: str = 'train'):
        self.mode = mode
        self.curriculum = cfg.curriculum
        self.curriculum_point = cfg.max_seq_len
        self.prices, self.data, self.indices = self._index_stocks()
        tools.pyout(f"dataset size: {len(self)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_len = sample_seq_len(self.curriculum, self.curriculum_point)
        ii, t0 = self.indices[idx]

        x_sos = torch.FloatTensor([0] * cfg.num_bits_per_vector + [1, 0]).unsqueeze(1)
        x_seq = torch.cat(
            (torch.FloatTensor(self.data[ii][:, t0:t0 + cfg.max_seq_len * cfg.history_multiplier]),
             torch.zeros(2, cfg.max_seq_len * cfg.history_multiplier)), dim=0)
        x_eos = torch.FloatTensor([0] * cfg.num_bits_per_vector + [0, 1]).unsqueeze(1)
        x_out = torch.zeros(cfg.input_size, cfg.max_seq_len)
        x = torch.cat((x_sos, x_seq, x_eos, x_out), dim=1)

        y = [0] * (cfg.max_seq_len * cfg.history_multiplier + 2)
        for t in range(t0 + cfg.max_seq_len * cfg.history_multiplier,
                       t0 + cfg.max_seq_len * cfg.history_multiplier + seq_len):
            y.append(y[-1] + self.data[ii][0, t])
        y += [0] * (cfg.max_seq_len - seq_len)
        y = torch.FloatTensor(y).unsqueeze(0)
        m = torch.cat((torch.zeros(1, cfg.max_seq_len * cfg.history_multiplier + 2),
                       torch.ones(1, seq_len),
                       torch.zeros(1, cfg.max_seq_len - seq_len)), dim=1)

        return x, y, m

    def _index_stocks(self):
        prices, dataset, indices = [], [], []

        scaler = preprocessing.StandardScaler()
        CLIP = 1.5

        for ii, symbol in enumerate(tqdm(listdir("./res/in/stocks"),
                                         desc=poem("Indexing Stocks"), leave=False)):
            try:
                dataframe = pd.read_csv(symbol)
                prices.append(dataframe[['Close']].to_numpy().T)
                df = dataframe[['Close', 'Open', 'Low', 'High', 'Volume']]
                df = df.diff().dropna()

                df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
                df = df.clip(np.quantile(df, .25, axis=0) - CLIP,
                             np.quantile(df, .75, axis=0) + CLIP)
                df = 2 * (df - df.min()) / (df.max() - df.min()) - 1

                df = df.to_numpy().T
                dataset.append(df)

                if df.shape[1] <= 200 + cfg.max_seq_len * 3:
                    continue
                if self.mode == 'train':
                    for jj in range(df.shape[1] - 200 - cfg.max_seq_len * (cfg.history_multiplier + 1), ):
                        indices.append((ii, jj))
                elif self.mode == 'eval':
                    for jj in range(df.shape[1] - 200 - cfg.max_seq_len * (cfg.history_multiplier + 1),
                                    df.shape[1] - 100 - cfg.max_seq_len * (cfg.history_multiplier + 1)):
                        indices.append((ii, jj))
                elif self.mode == 'test':
                    for jj in range(df.shape[1] - 100 - cfg.max_seq_len * (cfg.history_multiplier + 1),
                                    df.shape[1] - cfg.max_seq_len * (cfg.history_multiplier + 1)):
                        indices.append((ii, jj))
            except ValueError as e:
                tools.pyout(f"Invalid symbol: {symbol.split('/')[-1]}")
            # break

        return prices, dataset, indices
