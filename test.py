import random
import subprocess

while True:
    lr = eval(f"1e-{random.uniform(2, 6)}")
    bs = int(eval(f"2**{random.uniform(4, 8)}"))
    ms = 33

    subprocess.run([f"python main.py {lr} {bs} {ms}"], shell=True, check=True)
