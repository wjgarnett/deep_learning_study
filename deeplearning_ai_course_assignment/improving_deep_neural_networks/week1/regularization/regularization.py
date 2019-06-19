# coding: utf-8
from regularization_utils import *
import numpy as np
import logging # for debug
logging.basicConfig(format='%(levelname)s: %(message)s\t%(pathname)s[line:%(lineno)d]',
                    level=logging.ERROR) # DEBUG ERROR INFO


def dropout(A, keep_prob):
    np.random.seed(2)
    D = np.random.rand(A.shape[0], A.shape[1])
    D = D < keep_prob
    A = np.multiply(A, D) / keep_prob

    return A, D


def L_model_forward_with_dropout(X, parameters, keep_prob):
    np.random.seed(2)
    L = len(parameters) // 2
    A = X
    caches = []
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        A, D = dropout(A, keep_prob)
        cache = (cache[0], cache[1], (D,))
        logging.info('cache size: {}'.format(len(cache)))
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')  # 最后一层正向传播
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = Y.shape[1]
    cross_entropy_cost = compute_cost(AL, Y)

    L = len(parameters) // 2
    L2_regularization_cost = 0
    for i in range(L):
        L2_regularization_cost += (lambd/(2*m)) * np.sum(np.square(parameters['W'+str(i+1)]))

    cost = cross_entropy_cost + L2_regularization_cost

    return cost


def L_model_backward_with_regularization(AL, Y, caches, lambd):
    m = Y.shape[1]
    grads = {}
    L = len(caches)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    cache = caches[L-1]
    logging.debug('cache: {}'.format(len(cache)))
    A, W, b = cache[0]
    Z = cache[1]
    dAL_prev, dW_L, db_L = linear_activation_backward(dAL, cache, 'sigmoid')
    dW_L += (1.0*lambd / m)*W   # 加入正则化的dW_L
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = dAL_prev, dW_L, db_L
    for l in reversed(range(L-1)):
        cache = caches[l]
        A, W, b= cache[0]
        Z = cache[1]
        dA_prev, dW_tmp, db_tmp = linear_activation_backward(grads['dA' + str(l + 2)], cache, 'relu')
        logging.debug('W.shape: {}'.format(W.shape))
        logging.debug('dW_L.shape: {}'.format(dW_L.shape))
        dW_tmp += (1.0 * lambd / m) * W  # 加入正则化的dW_L
        grads['dA' + str(l + 1)], grads['dW' + str(l + 1)], grads['db' + str(l + 1)] = dA_prev, dW_tmp, db_tmp

    return grads


def L_model_backward_with_dropout(AL, Y, caches, keep_prob):
    m = Y.shape[1]
    grads = {}
    L = len(caches)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # 后向传播初始化
    cache = caches[L-1]
    dAL_prev, dW_L, db_L = linear_activation_backward(dAL, cache, 'sigmoid') # 该层无dropout
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = dAL_prev, dW_L, db_L
    logging.debug('dAL.shape: {}'.format(dAL.shape))
    logging.debug('dAL_prev.shape: {}'.format(dAL_prev.shape))
    for l in reversed(range(L-1)):
        cache = caches[l]
        Dl = cache[2][0]
        logging.debug('Dl.shape: {}'.format(Dl.shape))
        dAL_prev = grads['dA'+str(l+2)]
        dAL_prev = np.multiply(dAL_prev, Dl) / keep_prob
        logging.debug('dAL_prev.shape: {}'.format(dAL_prev.shape))
        logging.debug('---------------\n')
        # linear_activation_backward 返回值中的dA_prev需做处理
        dA_prev, dW_tmp, db_tmp = linear_activation_backward(dAL_prev, (cache[0], cache[1]), 'relu')
        grads['dA' + str(l + 1)], grads['dW' + str(l + 1)], grads['db' + str(l + 1)] = dA_prev, dW_tmp, db_tmp

    return grads


def model(X, Y, learning_rate=0.05, num_iterations=30000, print_cost=True, lambd=0, keep_prob=1):
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    # 网络参数初始化
    parameters = initialize_parameters(layers_dims)
    costs = []
    for i in range(num_iterations):
        # forward
        if keep_prob == 1:
            A3, caches = L_model_forward(X, parameters)
        elif keep_prob < 1:
            A3, caches = L_model_forward_with_dropout(X, parameters, keep_prob)

        # cost
        if lambd == 0:
            cost = compute_cost(A3, Y)
        else:
            cost = compute_cost_with_regularization(A3, Y, parameters, lambd)

        # backward
        assert(lambd == 0 or keep_prob == 1)
        if lambd == 0 and keep_prob == 1:
            grads = L_model_backward(A3, Y, caches)
        elif lambd != 0:
            grads = L_model_backward_with_regularization(A3, Y, caches, lambd)
        elif keep_prob < 1:
            grads = L_model_backward_with_dropout(A3, Y, caches, keep_prob)

        update_parameters(parameters, grads, learning_rate)


        if print_cost and i % 1000 == 0:
            print('cost after iteration {}： {}'.format(i, cost))
            costs.append(cost)

        # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    parameters = model(train_X, train_Y)
    pred_train = predict(train_X, train_Y, parameters)
    pred_test = predict(test_X, test_Y, parameters)