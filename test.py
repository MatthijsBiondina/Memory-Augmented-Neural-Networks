import random
import subprocess

while True:
    lr = eval(f"1e-{random.randint(2, 6)}")
    bs = random.randint(8, 512)
    ms = random.randint(20,64)

    subprocess.run([f"python main.py {lr} {bs} {ms}"], shell=True, check=True)
