import numpy as np

def LeakyRelu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)