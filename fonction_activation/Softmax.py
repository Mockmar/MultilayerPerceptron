import numpy as np

def softmax(x):
    """Compute the softmax of vector x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)