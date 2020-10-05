from torch.utils.data import DataLoader

from data.copy_dataset import CopyDataset
import torch
import utils.config as cfg
from data.copy_repeat_dataset import CopyRepeatDataset
from models.lstm_model import LSTMModel
from models.ntm_model import NTMModel
from utils.trainer import TrainLoop

from utils.utils import plots

# torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if cfg.task == 'copy':
    dataset = DataLoader(CopyDataset(), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
elif cfg.task == 'copy_repeat':
    dataset = DataLoader(CopyRepeatDataset(), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

# model = LSTMModel().to(device)
model = NTMModel().to(device)
train_loop = TrainLoop(dataset, model)
train_loop.summary()

# train_loop.load(path = f"./res/{cfg.task}_model.pth")
train_loop.fit()
train_loop.evaluate()
train_loop.visualize()
train_loop.save(path = f"./res/{cfg.task}_model.pth")

state_dict = model.state_dict()
pass
