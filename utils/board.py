import os

from utils.tools import makedir


class BokehBoard:
    def __init__(self, fname="default"):
        makedir(os.path.join('.', 'res', fname), delete=True)
        self.metrics = {'episode': [], 'train_loss': [], 'test_loss': []}

    def update_loss_plot(self, episode, train_loss, test_loss):
        self.metrics['episode'].append(episode)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['test_loss'].append(test_loss)

        fig = figure(width=720)