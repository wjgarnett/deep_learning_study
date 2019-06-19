# coding: utf-8

import numpy as np


# convolution functions

# helper_function

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0) # heigh和width方向做填充

    return X_pad


def conv_single_ste(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = float(b) + Z

    return Z


def conv_forward(A_prev, W, b, hyparams):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f, f, n_C_prev, n_C) = W.shape

    stride = hyparams['stride']
    pad = hyparams['pad']

    n_H = int((n_H_prev - f + 2*pad)/stride + 1)
    n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

    Z = np.zeros([m, n_H, n_W, n_C])

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    verti_start = h * stride
                    verti_end = h*stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f
                    a_slice_prev = a_prev_pad[verti_start: verti_end, horiz_start: horiz_end]
                    Z[i, w, h, c] = conv_single_ste(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    assert (Z.shape == (m, n_H, n_W, n_C))

    cache = (A_prev, W, b, hyparams)

    return Z, cache


def conv_backward(dZ, cache):
    (A_prev, W, b, hyparams) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hyparams['stride']
    pad = hyparams['pad']

    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # padding
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    verti_start = h*stride
                    verti_end = verti_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[verti_start: verti_end, horiz_start: horiz_end, :]

                    # print('----', dZ[i, h, w, c])
                    # print(W[:, :, :, c].shape)
                    db[:, :, :, c] += dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]  # 这里是在对卷积前向结果矩阵中的每一个元素求偏导
                    da_prev_pad[verti_start:verti_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c] # 同上

        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad: -pad, :]   #  切片操作会copy,此处是将结果copy到原始的array中

    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db


def pool_forward(A_prev, hyparams, mode='max'):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hyparams['f']
    stride = hyparams['stride']

    n_H = int(1+(n_H_prev-f)/stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros([m, n_H, n_W, n_C])

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    verti_start = h*stride
                    verti_end = verti_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_prev_slice = A_prev[i, verti_start: verti_end, horiz_start: horiz_end, c]
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hyparams)

    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache


def create_mask_from_window(x):
    mask = (x == np.max(x))

    return mask


def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    a = average*np.ones([n_H, n_W])

    return a


def pooling_backward(dA, cache, mode='max'):
    (A_prev, hyparams) = cache

    stride = hyparams['stride']
    f = hyparams['f']

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i, :, :, :]
        for h in range(n_H_prev-f+1):
            for w in range(n_W_prev-f+1):
                for c in range(n_C_prev):
                    verti_start = h * stride
                    verti_end = verti_start +f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # bugs
                    if mode == 'max':
                        a_prev_slice = a_prev[verti_start: verti_end, horiz_start: horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        print('---mask',mask.shape)
                        print(f)
                        print(dA[i, verti_start: verti_end, horiz_start: horiz_end, c].shape)
                        dA_prev[i, verti_start: verti_end, horiz_start: horiz_end, c] += np.multiply(mask, \
                                                    dA[i, verti_start: verti_end, horiz_start: horiz_end, c])
                    elif mode == 'average':
                        da = np.mean(dA[i, verti_start: verti_end, horiz_start: horiz_end, c])
                        shape = (f, f)
                        dA_prev[i, verti_start: verti_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    assert (dA_prev.shape == A_prev.shape)

    return dA_prev


if __name__ == '__main__':
    # np.random.seed(1)
    # A_prev = np.random.randn(3, 7, 7, 3)
    # W = np.random.randn(3, 3, 3, 8)
    # b = np.random.randn(1, 1, 1, 8)
    # hparameters = {"pad": 2,
    #                "stride": 2}
    #
    # Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    # print("Z's mean =", np.mean(Z))
    # print("Z[3,2,1] =", Z[0, 2, 1])
    # print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
    # print(Z.shape)

    # np.random.seed(1)
    # A_prev = np.random.randn(2, 4, 4, 3)
    # hparameters = {"stride": 2, "f": 3}
    #
    # A, cache = pool_forward(A_prev, hparameters)
    # print("mode = max")
    # print("A =", A)
    # print()
    # A, cache = pool_forward(A_prev, hparameters, mode="average")
    # print("mode = average")
    # print("A =", A)

    # np.random.seed(1)
    # A_prev = np.random.randn(10, 4, 4, 3)
    # W = np.random.randn(2, 2, 3, 8)
    # b = np.random.randn(1, 1, 1, 8)
    # hparameters = {"pad": 2, "stride": 2}
    # Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    # print('Z.shape is ', Z.shape)
    # np.random.seed(1)
    # dA, dW, db = conv_backward(Z, cache_conv)
    # print("dA_mean =", np.mean(dA))
    # print("dW_mean =", np.mean(dW))
    # print("db_mean =", np.mean(db))

    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride": 1, "f": 2}
    A, cache = pool_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pooling_backward(dA, cache, mode="max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])
    print()
    dA_prev = pooling_backward(dA, cache, mode="average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])