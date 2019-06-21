# coding: utf-8

import h5py
import numpy as np


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def random_mini_batches(X, Y, minibatch_size, seed):
    np.random.seed(seed)
    m = X.shape[0]
    minibatches = []

    # shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = int(m/minibatch_size)
    for k in range(num_complete_minibatches):
        minibatch_X = shuffled_X[k*minibatch_size : (k+1)*minibatch_size, :, :, :]
        minibatch_Y = shuffled_Y[k * minibatch_size: (k + 1) * minibatch_size, :]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)

    if m % minibatch_size != 0:
        minibatch_X = shuffled_X[num_complete_minibatches*minibatch_size : m, :, :, :]
        minibatch_Y = shuffled_Y[num_complete_minibatches*minibatch_size : m, :]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)

    return minibatches