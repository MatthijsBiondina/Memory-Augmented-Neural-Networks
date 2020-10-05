import torch
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    t = torch.FloatTensor(t)
    t = torch.sigmoid(t*10-5)-torch.sigmoid(t*-10-5)

    return t.numpy()

x = np.arange(-1.,1.,0.05)

plt.plot(x,f(x))
plt.show()