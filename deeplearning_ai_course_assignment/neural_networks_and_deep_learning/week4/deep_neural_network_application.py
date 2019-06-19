# coding: utf-8

from neural_networks_and_deep_learning.week2.utils import load_dataset
import matplotlib.pyplot as plt
# from dnn_app_utils import *
from dnn_utils import *
import numpy as np


def pre_process(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes):
    # 图像展开为列向量
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten / 255.0
    test_set_x = test_set_x_flatten / 255.0

    return train_set_x, test_set_x


# def two_layer_model(X, Y, layers_dims, learning_rate = 0.005, num_iterations = 3000, print_cost = False):
#     np.random.seed(1)
#
#     n_x, n_h, n_y = layers_dims
#     parameters = initialize_parameters(nx, n_h, n_y)
#
#     W1 = parameters['W1']
#     b1 = parameters['b1']
#     W2 = parameters['W2']
#     b2 = parameters['b2']
#
#     grads = {}
#     costs = []
#     for i in range(num_iterations):
#         # forward
#         A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
#         A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
#         # compute cost
#         cost = compute_cost(A2, Y)
#         # backward
#         dA2 = -(np.divide(Y, A2) - np.divide(1-Y, 1-A2))
#         dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
#         dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
#
#         grads['dW1'] = dW1
#         grads['db1'] = db1
#         grads['dW2'] = dW2
#         grads['db2'] = db2
#
#         parameters = update_parameters(parameters, grads, learning_rate)
#
#         W1 = parameters['W1']
#         b1 = parameters['b1']
#         W2 = parameters['W2']
#         b2 = parameters['b2']
#
#         costs.append(cost)
#         if print_cost and i % 100 == 0:
#             print('cost after iteration {}: {}'.format(i, cost))
#
#     # plot
#     plt.plot(costs)
#     plt.xlabel('iteration per hundreds')
#     plt.ylabel('cost')
#     plt.title('learning_rate = ', learning_rate)
#     plt.show()
#
#     return parameters


def L_layers_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=2000, print_cost=False):
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        # AL, caches = L_layers_model_forward(X, parameters)
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        # grads = L_layers_model_backward(AL, Y, caches)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        costs.append(cost)
        if print_cost and i % 100 == 0:
            print('cost after iteration {}: {}'.format(i, cost))

    plt.plot(costs)
    plt.xlabel('iterations per 100s')
    plt.ylabel('cost')
    plt.title('learning_rate = ' + str(learning_rate))
    plt.show()

    return parameters


def print_mislabeled_images(classes, X, y, p):
    a = p+y
    mislabeled_indices = np.asarray(np.where(a == 1))
    print(mislabeled_indices.shape)

    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i+1)
        plt.imshow(X[:, index].reshape(64, 64, 3))

    plt.show()


if __name__ == '__main__':
    # load raw data
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    # pre_prcosee
    train_set_x, test_set_x = pre_process(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes)

    layers_dims = [12288, 64, 16, 4, 1]
    parameters = L_layers_model(train_set_x, train_set_y, layers_dims, num_iterations=2500, print_cost=True)

    pred_train = predict(train_set_x, train_set_y, parameters)
    pred_test = predict(test_set_x, test_set_y, parameters)

    print_mislabeled_images(classes, test_set_x, test_set_y, pred_test)
