import time
import sys
from datetime import datetime
from utils import tools

import torch
from torch.nn.functional import softplus, sigmoid

args = len(sys.argv) > 1

mann: str = 'dnc'  # lstm | ntm | dnc
num_layers: int = 1
num_units: int = 100
num_memory_locations: int = 32 if not args else int(sys.argv[3])
memory_size: int = 20
num_read_heads: int = 1
num_write_heads: int = 1
conv_shift_range: int = 1  # only necessary for ntm
clip_value: int = 20  # Maximum absolute value of controller and outputs
init_mode: str = 'random'  # learned | constant | random

optimizer: str = 'Adam'  # RMSProp | Adam
learning_rate: float = 3e-3 if not args else float(sys.argv[1])
max_grad_norm: float = 10.
num_train_steps: int = 31250000
batch_size: int = 64 if not args else int(sys.argv[2])
eval_batch_size: int = 1

curriculum: str = 'uniform'  # none | uniform | naive | look_back | look_back_and_forward | prediction_gain
pad_to_max_seq_len: bool = False

verbose: bool = True  # if true prints lots of feedback
steps_per_eval: int = 10

task: str = 'stocks'  # copy | copy_repeat | associative_recall | stocks
experiment_name: str = f"{mann}_{task}_{batch_size}_{num_memory_locations}_{learning_rate}" if args else "debug"

num_bits_per_vector = None
input_size = None
max_seq_len = None
output_size = None
output_func = None
max_repeats = None
max_items = None
history_multiplier = None
mdn_mixing_units = None
if task is "copy":  # task specific parameters
    num_bits_per_vector: int = 8
    input_size = num_bits_per_vector + 2
    max_seq_len: int = 20
    output_size = num_bits_per_vector
    output_func = torch.nn.Sigmoid()
elif task is "copy_repeat":
    num_bits_per_vector = 8
    input_size = num_bits_per_vector + 2
    output_size = num_bits_per_vector + 1
    max_seq_len: int = 20
    max_repeats = 10
    output_func = torch.nn.Sigmoid()
elif task is "associative_recall":
    num_bits_per_vector = 6
    input_size = num_bits_per_vector + 2
    output_size = num_bits_per_vector
    output_func = torch.nn.Sigmoid()
    max_items: int = 6
elif task is "stocks":
    num_bits_per_vector = 5
    input_size = num_bits_per_vector + 2
    output_size = 2
    max_seq_len: int = 5
    history_multiplier: int = 12
    mdn_mixing_units = 5


    def output_func(y):
        mu = y[:, 0]
        s2 = 2 * sigmoid(y[:, 1])
        return torch.stack((mu, s2), dim=1)
