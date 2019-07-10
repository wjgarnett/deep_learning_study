#coding: utf-8

import numpy as np
from rnn_utils import *


def rnn_cell_forward(xt, a_prev, parameters):
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    a_next = np.tanh(np.matmul(Wax, xt) + np.matmul(Waa, a_prev) + ba)
    print(a_next.shape)
    yt_pred = softmax(np.matmul(Wya, a_next) + by)
    print(yt_pred.shape)
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """
    Args:
        x: Input data, shape (n_x, m, T_x), n_x代表单个时间步的特征维度，m为样本数，Tx代表总步数
        a0: hidden state的初始值, shape (n_a, m)
        parameters: dict type
            权重矩阵: Waa-(n_a, n_a), Wax-(n_a, n_x), Wya-(n_y, n_a)
            偏置向量：ba-(n_a, 1), by-(n_y, 1)  相加时会broadcasting

    Returns:
        a: hidden state of every time step, shape (n_a, m, T_x)
        y_pred: predictions of every time step, shape (n_y, m, Y_x)
        caches: backward pass需要用到的一些值
    """
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape

    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    a_next = a0
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)

    return a, y_pred, caches


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Args:
        xt: input data at time step t, shape (n_x, m)
        a_prev: hidden state at time step t-1, shape (n_a, m)
        c_prev: memory state at time step t-1, shape (n_a, m)
        parameters:
            Wf, bf -- forget gate, shape (n_a, n_a+n_x), (n_a, 1)
            Wi, bi -- update gate
            Wc, bc --
            Wo, bo -- output gate
            Wy, by -- shape (n_y, n_a), (n_y, 1)

    Returns:
    """
    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wy = parameters['Wy']
    by = parameters['by']

    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # 连接
    concat = np.zeros(((n_a+n_x), m))
    concat[: n_a] = a_prev
    concat[n_a: ] = xt

    ft = sigmoid(np.matmul(Wf, concat) + bf)
    it = sigmoid(np.matmul(Wi, concat) + bi)
    cct = np.tanh(np.matmul(Wc, concat) + bc)
    c_next = ft*c_prev + it * cct
    ot = sigmoid(np.matmul(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    yt_pred = softmax(np.matmul(Wy, a_next) + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape

    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    a_next = a0
    c_next = np.zeros((n_a, m))

    for t in range(T_x):
        a_next, c_next, yt, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        a[:, :, t] = a_next
        y[:, :, t] = yt
        c[:, :, t] = c_next
        caches.append(cache)

    caches = (caches, x)

    return a, y, c, caches








if __name__ == '__main__':
    # np.random.seed(1)
    # xt = np.random.randn(3, 10)
    # a_prev = np.random.randn(5, 10)
    # Waa = np.random.randn(5, 5)
    # Wax = np.random.randn(5, 3)
    # Wya = np.random.randn(2, 5)
    # ba = np.random.randn(5, 1)
    # by = np.random.randn(2, 1)
    # parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
    #
    # a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
    # print("a_next[4] = ", a_next[4])
    # print("a_next.shape = ", a_next.shape)
    # print("yt_pred[1] =", yt_pred[1])
    # print("yt_pred.shape = ", yt_pred.shape)

    # np.random.seed(1)
    # x = np.random.randn(3, 10, 4)
    # a0 = np.random.randn(5, 10)
    # Waa = np.random.randn(5, 5)
    # Wax = np.random.randn(5, 3)
    # Wya = np.random.randn(2, 5)
    # ba = np.random.randn(5, 1)
    # by = np.random.randn(2, 1)
    # parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
    #
    # a, y_pred, caches = rnn_forward(x, a0, parameters)
    # print("a[4][1] = ", a[4][1])
    # print("a.shape = ", a.shape)
    # print("y_pred[1][3] =", y_pred[1][3])
    # print("y_pred.shape = ", y_pred.shape)
    # print("caches[1][1][3] =", caches[1][1][3])
    # print("len(caches) = ", len(caches))

    # np.random.seed(1)
    # xt = np.random.randn(3, 10)
    # a_prev = np.random.randn(5, 10)
    # c_prev = np.random.randn(5, 10)
    # Wf = np.random.randn(5, 5 + 3)
    # bf = np.random.randn(5, 1)
    # Wi = np.random.randn(5, 5 + 3)
    # bi = np.random.randn(5, 1)
    # Wo = np.random.randn(5, 5 + 3)
    # bo = np.random.randn(5, 1)
    # Wc = np.random.randn(5, 5 + 3)
    # bc = np.random.randn(5, 1)
    # Wy = np.random.randn(2, 5)
    # by = np.random.randn(2, 1)
    #
    # parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
    #
    # a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
    # print("a_next[4] = ", a_next[4])
    # print("a_next.shape = ", c_next.shape)
    # print("c_next[2] = ", c_next[2])
    # print("c_next.shape = ", c_next.shape)
    # print("yt[1] =", yt[1])
    # print("yt.shape = ", yt.shape)
    # print("cache[1][3] =", cache[1][3])
    # print("len(cache) = ", len(cache))

    np.random.seed(1)
    x = np.random.randn(3, 10, 7)
    a0 = np.random.randn(5, 10)
    Wf = np.random.randn(5, 5 + 3)
    bf = np.random.randn(5, 1)
    Wi = np.random.randn(5, 5 + 3)
    bi = np.random.randn(5, 1)
    Wo = np.random.randn(5, 5 + 3)
    bo = np.random.randn(5, 1)
    Wc = np.random.randn(5, 5 + 3)
    bc = np.random.randn(5, 1)
    Wy = np.random.randn(2, 5)
    by = np.random.randn(2, 1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a, y, c, caches = lstm_forward(x, a0, parameters)
    print("a[4][3][6] = ", a[4][3][6])
    print("a.shape = ", a.shape)
    print("y[1][4][3] =", y[1][4][3])
    print("y.shape = ", y.shape)
    print("caches[1][1[1]] =", caches[1][1][1])
    print("c[1][2][1]", c[1][2][1])
    print("len(caches) = ", len(caches))

