import random
import subprocess

while True:
    lr = eval(f"1e-{random.randint(2, 6)}")
    bs = random.randint(8, 512)
    ms = 32

    subprocess.run(["python", "main.py", str(lr), str(bs), str(ms)], shell=True, check=True)
