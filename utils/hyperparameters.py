import json

from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from utils import tools
import os
import seaborn as sns
from utils.tools import listdir, str_clean
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class Hyper:
    def __init__(self, root='./res'):
        self.folders = list(self._index(root))
        self.scores = {}
        self.times = {}
        self.params = self._read_config()
        self.params = self._remove_nonunique(self.params)
        self.params['score'] = list(self._read_score())

        self.df = pd.DataFrame.from_dict(self.params)

    def plot_scores(self,hue='learning_rate'):
        cmap = cm.cool
        fig, ax = plt.subplots(figsize=(10.80, 6.40))
        # fig.set_facecolor('black')
        ax.set_facecolor('black')
        divider  = make_axes_locatable(ax)

        cr_min = np.log10(min(self.params[hue][ii] for ii in self.scores.keys()))
        cr_max = np.log10(max(self.params[hue][ii] for ii in self.scores.keys()))
        for key in tqdm(self.scores):
            color = (np.log10(self.params[hue][key]) - cr_min) / (cr_max-cr_min)
            color = cmap(color)[:3] + (0.5,)
            plt.plot(self.times[key], self.scores[key], color=color,linewidth=2)


        plt.title(hue)

        plt.xlabel('train step')
        plt.ylabel('loss (mse)')

        cax = divider.append_axes('right', size='5%', pad=0.05)
        norm = matplotlib.colors.Normalize(
            vmin=10**cr_min, vmax=10**cr_max)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=(norm.vmin, norm.vmax))
        plt.xlabel(hue)


        plt.savefig(f"./res/{hue}.pdf")

    def heatmaps(self):
        for key in self.params:
            try:
                if key is not "score":
                    sns.jointplot(x=np.log10(self.df[key]),
                                  y=self.df['score'],
                                  kind='scatter')
                    plt.savefig(f'./res/{key}_scatter.pdf')
            except:
                pass

    def _remove_nonunique(self, Din: dict):
        Dou = {}
        for key in Din:
            if not Din[key].count(Din[key][0]) == len(Din[key]):
                Dou[key] = Din[key]
        return Dou

    def _index(self, root):
        for folder in listdir(root):
            if os.path.isfile(os.path.join(folder, 'loss.html')):
                yield folder

    def _read_config(self):
        params = {}
        with open(os.path.join(self.folders[0], 'config.txt')) as f:
            for line in f:
                line = str_clean(line, ' ', '\n').split(':=')
                if line[1] in ('True', 'False'):
                    params[line[0]] = [bool(line[1])]
                elif line[1].isdigit():
                    params[line[0]] = [int(line[1])]
                else:
                    try:
                        params[line[0]] = [float(line[1])]
                    except ValueError:
                        params[line[0]] = [str(line[1])]
        for folder in self.folders[1:]:
            with open(os.path.join(folder, 'config.txt')) as f:
                for line in f:
                    line = str_clean(line, ' ', '\n').split(':=')
                    params[line[0]].append(type(params[line[0]][-1])(line[1]))
        return params

    def _read_score(self):
        scores = []
        for ii, folder in enumerate(self.folders):
            with open(os.path.join(folder, 'metrics.json')) as f:
                M = json.load(f)
                scores.append(M['train_loss'][-1])

                self.scores[ii] = M['train_loss']
                self.times[ii] = M['episode']

        return scores
