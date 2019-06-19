# coding: utf-8

import numpy as np
import math
import matplotlib.pyplot as plt

from optimization_methods_test_case import *
from opt_utils import *


# rcParams用于自定义图形的各种属性
plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'  # 设置颜色style


def update_parameters_with_gd(parameters, grads, learning_rate):
    # 获取网络层数
    L = len(parameters) // 2

    # update parameters
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - grads['dW'+str(l+1)]*learning_rate
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - grads['db' + str(l + 1)] * learning_rate

    return parameters


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # shuffle
    permutation = list(np.random.permutation(m))
    # print('permutation:', permutation)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape(1, m)

    num_complete_mini_batches = math.floor(m / mini_batch_size)
    for i in range(num_complete_mini_batches):
        mini_bath_X = shuffled_X[:, i*mini_batch_size : (i+1)*mini_batch_size]
        mini_bath_Y = shuffled_Y[:, i * mini_batch_size: (i + 1) * mini_batch_size]
        mini_batches.append((mini_bath_X, mini_bath_Y))

    if m % mini_batch_size:
        mini_bath_X = shuffled_X[:, num_complete_mini_batches*mini_batch_size : ]
        mini_bath_Y = shuffled_Y[:, num_complete_mini_batches * mini_batch_size:]
        mini_batches.append((mini_bath_X, mini_bath_Y))

    return mini_batches


def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v['dW'+ str(l+1)] = np.zeros((parameters['W' + str(l+1)].shape[0], parameters['W' + str(l+1)].shape[1]))
        v['db' + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape[0], parameters['b' + str(l + 1)].shape[1]))

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        v['dW'+str(l+1)] = beta * v['dW'+str(l+1)] + (1-beta) * grads['dW'+str(l+1)]
        v['db' + str(l + 1)] = beta * v['db' + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]

        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*v['dW'+str(l+1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v['db' + str(l + 1)]

    return parameters, v    # 注意要返回parameters & v


def initialzie_adam(parameters):
    L = len(parameters) // 2
    v = {}  # momentum
    s = {}  # RMSProp

    for l in range(L):
        v['dW'+str(l+1)] = np.zeros((parameters['W'+str(l+1)].shape[0], parameters['W'+str(l+1)].shape[1]))
        v['db' + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape[0], parameters['b' + str(l + 1)].shape[1]))
        s['dW' + str(l + 1)] = np.zeros((parameters['W' + str(l + 1)].shape[0], parameters['W' + str(l + 1)].shape[1]))
        s['db' + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape[0], parameters['b' + str(l + 1)].shape[1]))

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corr = {}
    s_corr = {}

    for l in range(L):

        v['dW'+str(l+1)] = beta1*v['dW'+str(l+1)] + (1-beta1)*grads['dW'+str(l+1)]
        v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
        # correction
        v_corr['dW'+str(l+1)] = v['dW'+str(l+1)] / (1-pow(beta1, t))
        v_corr['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - pow(beta1, t))

        s['dW'+str(l+1)] = beta2*s['dW'+str(l+1)] + (1-beta2)*np.power(grads['dW'+str(l+1)], 2)
        s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
        # correction
        s_corr['dW'+str(l+1)] = s['dW'+str(l+1)]/(1-pow(beta2, t))
        s_corr['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - pow(beta2, t))

        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*np.divide(v_corr['dW'+str(l+1)], \
                                                                                      np.sqrt(s_corr['dW'+str(l+1)]+epsilon))
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * np.divide(
            v_corr['db' + str(l + 1)], \
            np.sqrt(s_corr['db' + str(l + 1)] + epsilon))

        return parameters, v, s


def model(X, Y, layers_dims, optimizer, learning_rate=0.007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, \
          epsilon=1e-8, num_epoch=1000, print_cost=True):
    L = len(layers_dims)
    costs = []
    t = 0;
    seed = 10;
    parameters = initialize_parameters(layers_dims)

    if optimizer == 'gd':
        pass
    elif optimizer == 'momentum':
        v = initialize_velocity(parameters)
    elif optimizer == 'adam':
        v, s = initialzie_adam(parameters)

    for i in range(num_epoch):
        seed += 1
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)

        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            A3, caches = forward_propagation(mini_batch_X, parameters)
            cost = compute_cost(A3, mini_batch_Y)
            grads = backward_propagation(mini_batch_X, mini_batch_Y, caches)
            if optimizer == 'gd':
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == 'momentum':
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == 'adam':
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, beta1, beta2, learning_rate, epsilon)


        if print_cost and i % 100 == 0:
            print('cost after iterations {}: {}'.format(i, cost))

    costs.append(cost)
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters



if __name__ == '__main__':
    # parameters, grads, learning_rate = update_parameters_with_gd_test_case()
    # parameters = update_parameters_with_gd(parameters, grads, learning_rate)
    # print(parameters)

    # X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
    # mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
    # print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    # print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    # print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    # print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    # print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
    # print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    # print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

    # parameters = initialize_velocity_test_case()
    # v = initialize_velocity(parameters)
    # print(v)

    # parameters, grads, v = update_parameters_with_momentum_test_case()
    #
    # parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))
    # print("v[\"dW1\"] = " + str(v["dW1"]))
    # print("v[\"db1\"] = " + str(v["db1"]))
    # print("v[\"dW2\"] = " + str(v["dW2"]))
    # print("v[\"db2\"] = " + str(v["db2"]))

    # parameters = initialize_adam_test_case()
    # v, s = initialzie_adam(parameters)
    # print(v)
    # print(s)

    # parameters, grads, v, s = update_parameters_with_adam_test_case()
    # parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)
    #
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))
    # print("v[\"dW1\"] = " + str(v["dW1"]))
    # print("v[\"db1\"] = " + str(v["db1"]))
    # print("v[\"dW2\"] = " + str(v["dW2"]))
    # print("v[\"db2\"] = " + str(v["db2"]))
    # print("s[\"dW1\"] = " + str(s["dW1"]))
    # print("s[\"db1\"] = " + str(s["db1"]))
    # print("s[\"dW2\"] = " + str(s["dW2"]))
    # print("s[\"db2\"] = " + str(s["db2"]))

    train_X, train_Y = load_dataset()
    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer="adam")

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

