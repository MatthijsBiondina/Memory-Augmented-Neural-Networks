from torch.utils.data import DataLoader

from data.associative_recall_dataset import AssociativeRecallDataset
from data.copy_dataset import CopyDataset
import torch
import utils.config as cfg
from data.copy_repeat_dataset import CopyRepeatDataset
from data.stock_dataset import StockDataset
from models.dnc_model import DNCModel
from models.lstm_model import LSTMModel
from models.ntm_model import NTMModel
from models.transformer_model import TransformerModel
from utils.trainer import TrainLoop

from utils.utils import plots

# torch.autograd.set_detect_anomaly(True)

# cfg.num_bits_per_vector = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if cfg.task == 'copy':
    data = CopyDataset()
elif cfg.task == 'copy_repeat':
    data = CopyRepeatDataset()
elif cfg.task == 'associative_recall':
    data = AssociativeRecallDataset()
elif cfg.task == 'stocks':
    data = StockDataset()

dataset = DataLoader(data, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

if cfg.mann == 'lstm':
    model = LSTMModel().to(device)
elif cfg.mann == 'ntm':
    model = NTMModel().to(device)
elif cfg.mann == 'dnc':
    model = DNCModel().to(device)
elif cfg.mann == 'transformer':
    model = TransformerModel().to(device)

# print(cfg.batch_size)

train_loop = TrainLoop(dataset, model)
train_loop.summary()

# train_loop.load(path = f"./res/{cfg.task}_model.pth")
train_loop.fit()
train_loop.evaluate()
train_loop.visualize()
# train_loop.save(path=f"./res/{cfg.task}_model.pth")

state_dict = model.state_dict()
pass
