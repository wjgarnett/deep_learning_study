# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from initialization_utils import *


def initialize_parameters_zeros(layers_dims):
    np.random.seed(1)
    L = len(layers_dims)

    parameters = {}
    for i in range(1, L):
        parameters['W' + str(i)] = np.zeros((layers_dims[i], layers_dims[i-1]))
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))

    return parameters


def initialize_parameters_random(layers_dims):
    np.random.seed(1)
    L = len(layers_dims)

    parameters = {}
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1]) * 0.5
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))

    return parameters


def initialize_parameters_he(layers_dims):
    np.random.seed(1)
    L = len(layers_dims)

    parameters = {}
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1]) * np.sqrt(2/layers_dims[i-1])
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))

    return parameters


def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization='he'):
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dims)

    costs = []
    grads = {}
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        costs.append(cost)
        if print_cost and i % 1000 == 0:
            print('cost after iteration {}: {}'.format(i, cost))

    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per thousands)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_dataset()
    parameters = model(train_X, train_Y, initialization='random')   # 'zero' 'random' 'he'
    pred_train = predict(train_X, train_Y, parameters)
    pred_test = predict(test_X, test_Y, parameters)