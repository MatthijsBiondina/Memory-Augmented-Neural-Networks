import random
import subprocess
import time

def start_session(ii, cuda=''):
    cmd = f"tmux new-session -d -s {ii}_hypersearch_{ii} 'source ~/venv/bin/activate && cd ~/MANN && CUDA_VISIBLE_DEVICES={cuda} python test.py'"
    # cmd = f"tmux new-session -d -s {ii}_hypersearch_{ii} 'source ~/venv/bin/activate'"
    print(cmd)
    subprocess.run([cmd], shell=True, check=True)
    time.sleep(1)

for ii in range(5):
    start_session(ii, 2)
for ii in range(5, 15):
    start_session(ii, 1)
for ii in range(15,25):
    start_session(ii, 0)

time.sleep(10)