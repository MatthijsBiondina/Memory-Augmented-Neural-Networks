import random
import subprocess

while True:
    lr = eval(f"1**(-{random.uniform(2, 6)})")
    bs = int(eval(f"2**{random.uniform(4, 8)}"))
    ms = 32

    subprocess.run([f"python main.py {lr} {bs} {ms}"], shell=True, check=True)
