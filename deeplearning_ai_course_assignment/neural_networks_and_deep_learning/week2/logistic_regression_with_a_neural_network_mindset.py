# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from utils import load_dataset

def pre_process(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes):
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    # print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    # print("train_set_y shape: " + str(train_set_y.shape))
    # print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    # print("test_set_y shape: " + str(test_set_y.shape))
    # print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

    train_set_x = train_set_x_flatten / 255.0
    test_set_x = test_set_x_flatten / 255.0

    return train_set_x, test_set_x


def sigmoid(z):
    '''
    computer the sigmoid of z
    :param z:
    :return:
    '''
    s = 1.0 / (1+np.exp(-z))

    return s


def initialize_with_zeros(dims):
    w = np.zeros((dims, 1))
    b = 0.0

    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    # print('m: ', m)

    # forward propagation
    A = sigmoid(np.dot(w.T, X) +b)
    # cost = -np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) / m
    cost=-np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m
    cost = np.squeeze(cost)

    # backward propagation
    dw = np.dot(X, (A-Y).T)/m
    db = np.sum(A-Y)/m

    grads = {'dw': dw,
             'db': db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0 :
            print('cost after iteration %i: %f' % (i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_pred = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X)+b)
    # print('A: ', A)
    Y_pred = np.round(A)
    # print('Y_pred: ', Y_pred)

    return  Y_pred


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params['w']
    b = params['b']

    Y_pred_test = predict(w, b, X_test)
    Y_pred_train = predict(w, b, X_train)

    print('train accuracy: {} %'.format( 100 * (1.0 - np.mean(np.abs(Y_train - Y_pred_train))) ))
    print('test accuracy: {} %'.format(100 * (1.0 - np.mean(np.abs(Y_test - Y_pred_test)))))

    d = {'costs': costs,
         'w': w,
         'b': b,
         'learning_rate': learning_rate,
         'num_iterations': num_iterations,
         'Y_pred_test': Y_pred_test,
         'Y_pred_train': Y_pred_train}

    return d


if __name__ == '__main__':
    # loading data
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    train_set_x, test_set_x = pre_process(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes)

    d = model(train_set_x, train_set_y, test_set_x, test_set_y)

    # plot learning curve with costs
    costs = np.squeeze(d['costs'])
    print(costs)
    plt.plot(costs, color='#054E9F')
    plt.ylabel('cost')
    plt.xlabel('iterations per hundreds')
    plt.title('learning rate = ' + str(d['learning_rate']))
    plt.show()

    # index = 10
    # num_px = train_set_x_orig.shape[1]
    # plt.imshow(train_set_x[:, index].reshape((num_px, num_px, 3)))
    # print('d["Y_pred_test"]: ', d["Y_pred_test"][0, index])
    #
    # print("y = " + str(train_set_x[0, index]) + ", you predicted that it is a \"" +
    #       classes[int(d["Y_pred_train"][0, index])].decode("utf-8") + "\" picture.")
    # plt.show()




