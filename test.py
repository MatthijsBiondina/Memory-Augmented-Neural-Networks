import random
import subprocess

while True:
    lr = 0.1**(random.uniform(1.5,3))
    bs = int(2**random.randint(6, 6))
    ms = 32

    subprocess.run([f"python main.py {lr} {bs} {ms}"], shell=True, check=True)
