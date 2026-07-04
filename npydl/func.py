import numpy as np
import math


# random seed
def fix_seed(x=1557):
    np.random.seed(x)


# init dist
def xavier(fin, fout):
    scale = 1 / max(1., (fin + fout) / 2.)
    limit = math.sqrt(3.0 * scale) # where limit is sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fin, fout))


# func
def sigmoid(x):
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = np.exp(x - np.max(x)) # overflow proof
    return x / np.sum(x)


# loss func
def sum_squares_loss(y, tgt): # L2
    return np.sum((y-tgt)**2) * 0.5


def cross_entropy_error(y, tgt):
    y, tgt = (y.reshape(1, y.size), tgt.reshape(1, tgt.size)) if y.ndim == 1 else (tgt, y)
    return -np.sum(tgt * np.log(y + 1e-9)) / y.shape[0]
