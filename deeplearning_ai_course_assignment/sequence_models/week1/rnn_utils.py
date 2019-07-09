# coding: utf-8

import numpy as np


def softmax(x):
    if len(x.shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))   # 防止exp后溢出
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    return x


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    """
    A = 1 / (1 + np.exp(-Z))

    return A