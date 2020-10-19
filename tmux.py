import random
import subprocess


def start_session(ii, cuda=''):
    cmd = f"tmux new-session -d -s hypersearch_{ii} 'source ~/venv/bin/activate && cd ~/MANN && CUDA_VISIBLE_DEVICES={cuda} python test.py'"
    subprocess.run([cmd], shell=True, check=True)


for ii in range(5):
    start_session(ii, 1)
for ii in range(5, 15):
    start_session(ii, 2)
