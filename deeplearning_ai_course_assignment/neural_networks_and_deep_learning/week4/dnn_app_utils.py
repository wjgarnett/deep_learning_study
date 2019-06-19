# coding: utf-8
import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ


def relu(Z):
    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)
    cache = Z

    return A, cache


def relu_backward(dA, cache):
    Z = cache

    dZ = np.array(dA)
    dZ[Z<=0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def initialize_parameters(n_x, n_h, n_y):
    '''
    initialize one hidden layer neural network parameters
    :param n_x: size of input layer
    :param n_h: size of hidden layer
    :param n_y: size of output layer
    :return: python dict type ...
    '''
    np.random.seed(1)

    W1 = np.random.rand(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }

    return parameters


def initialize_parameters_deep(layer_dims):
    '''
    :param layer_dims: python array(list), contain the dimensions of each layer in our neural network
    :return:
    '''
    np.random.seed(1)
    L = len(layer_dims)

    parameters = {}
    for i in range(1, L):
        parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) / np.sqrt(layer_dims[i-1]) # ?
        parameters['b'+str(i)] = np.zeros((layer_dims[i], 1))
        assert (parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i-1]))
        assert (parameters['b' + str(i)].shape == (layer_dims[i], 1))

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def compute_cost(AL, Y):
    '''
    compute cross-entropy cost
    :param AL:
    :param Y:
    :return:
    '''
    m = Y.shape[1]

    cost = (-1.0/m) * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T))
    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    A_pre, W, b = cache
    m = A_pre.shape[1]

    dW = (1.0/m) * np.dot(dZ, A_pre.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_pre = np.dot(W.T, dZ)

    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    assert (dA_pre.shape == A_pre.shape)

    return dA_pre, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_forward(X, parameters):
    A = X
    L = len(parameters) // 2    # num of layers

    # 正向传播
    caches = []
    for l in range(1, L):   # 前L-1层正向传播
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid') # 最后一层正向传播
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)

    # 反向传播
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))    # 后向传播初始化
    cache = caches[L-1]
    dAL_prev, dW_L, db_L = linear_activation_backward(dAL, cache, 'sigmoid')
    grads['dA'+str(L)], grads['dW'+str(L)], grads['db'+str(L)] = dAL_prev, dW_L, db_L
    for l in reversed(range(L-1)):
        cache = caches[l]
        dA_prev, dW_tmp, db_tmp = linear_activation_backward(grads['dA'+str(l+2)], cache, 'relu')
        grads['dA' + str(l+1)], grads['dW' + str(l+1)], grads['db' + str(l+1)] = dA_prev, dW_tmp, db_tmp

    return grads


def update_parameters(parameters, grads, learning_rate):
    for i in range(len(parameters)//2):
        parameters['W' + str(i+1)] = parameters['W' + str(i+1)] - grads['dW' + str(i+1)]*learning_rate
        parameters['b' + str(i + 1)] = parameters['b' + str(i + 1)] - grads['db' + str(i + 1)] * learning_rate

    return parameters


def predict(X, Y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probs, caches = L_model_forward(X, parameters)
    probs[probs<0.5] = 0
    probs[probs>=0.5] = 1
    p = probs

    print('accuracy: ', np.sum(p == Y) / float(m))

    return p














