# coding: utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_tutorial_utils import *

def run_example1():
    y_hat = tf.constant(36, name='y_hat')
    y = tf.constant(39, name='y')

    loss = tf.Variable((y - y_hat)**2, name='loss')

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        print(session.run(loss))


def linear_function_example():
    np.random.seed(1)

    X = tf.constant(np.random.randn(3, 1), name='X')
    W = tf.constant(np.random.randn(4, 3), name='W')
    b = tf.constant(np.random.randn(4, 1), name='b')
    Y = tf.add(tf.matmul(W, X), b)

    with tf.Session() as sess:
        result = sess.run(Y)
        print('result: {}'.format(result))


def cost_example(logits, labels):
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    with tf.Session() as sess:
        cost = sess.run(cost, feed_dict={z:logits, y:labels})   # 使用实参logits替换z...

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0005, num_epochs=1500, minibatch_size=32, print_cost=True):
    tf.reset_default_graph()
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    seed = 3
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))  # tf.argmax返回最大值所在索引

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters





if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # # show img
    # plt.imshow(X_train_orig[5])
    # plt.show()

    # preprocess
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    X_train = X_train_flatten / 255.0
    X_test = X_test_flatten / 255.0
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    # print("number of training examples = " + str(X_train.shape[1]))
    # print("number of test examples = " + str(X_test.shape[1]))
    # print("X_train shape: " + str(X_train.shape))
    # print("Y_train shape: " + str(Y_train.shape))
    # print("X_test shape: " + str(X_test.shape))
    # print("Y_test shape: " + str(Y_test.shape))

    parameters = model(X_train, Y_train, X_test, Y_test)
