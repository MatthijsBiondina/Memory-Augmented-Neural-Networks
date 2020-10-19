import time
import sys
from datetime import datetime

args = len(sys.argv) > 1

mann: str = 'ntm' # lstm | ntm | dnc
num_layers: int = 1
num_units: int = 100
num_memory_locations: int = 32 if not args else int(sys.argv[3])
memory_size: int = 20
num_read_heads: int = 1
num_write_heads: int = 1
conv_shift_range: int = 1  # only necessary for ntm
clip_value: int = 20  # Maximum absolute value of controller and outputs
init_mode: str = 'learned'  # learned | constant | random

optimizer: str = 'Adam'  # RMSProp | Adam
learning_rate: float = 0.01 if not args else float(sys.argv[1])
max_grad_norm: float = 10.
num_train_steps: int = 31250 * 2
batch_size: int = 128 if not args else int(sys.argv[2])
eval_batch_size: int = 1

curriculum: str = 'uniform'  # none | uniform | naive | look_back | look_back_and_forward | prediction_gain
pad_to_max_seq_len: bool = False

task: str = 'associative_recall'
num_bits_per_vector: int = 8
max_seq_len: int = 20
max_repeats: int = 10
max_items: int = 6  # maximum number of items for associative recall task

verbose: bool = True  # if true prints lots of feedback
experiment_name: str = f"{mann}_{learning_rate}_{batch_size}_{num_memory_locations}" if args else "debug"
steps_per_eval: int = 10

input_size = num_bits_per_vector + 2
output_size = num_bits_per_vector if task in ('copy','associative_recall') else num_bits_per_vector + 1
