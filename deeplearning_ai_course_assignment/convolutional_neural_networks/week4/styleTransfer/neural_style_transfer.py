# coding: utf-8

from neural_style_transfer_utils import *
import cv2
import numpy as np
import tensorflow as tf

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image



def compute_content_cost(a_C, a_G):

    m, n_H, n_W, n_C = list(a_C.shape)

    J_content = tf.reduce_sum(tf.square(a_C - a_G) / (4*n_H*n_W*n_C))   # 4???

    return J_content


def gram_matrix(A):
    """ compute gram matrix (style matrix)

    Args:
        A: matrix of shape (n_C, n_H*n_W)

    Returns:
        GA: gram matrix of shape (n_C, n_C)
    """

    GA = tf.matmul(A, tf.transpose(A))

    return GA


def compute_layer_style_cost(a_S, a_G):

    m, n_H, n_W, n_C = list(a_G.shape)

    a_S = tf.reshape(a_S, [n_C, n_H*n_W])
    a_G = tf.reshape(a_G, [n_C, n_H * n_W])

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.square(GS - GG)) / (4*(int(n_C*n_H*n_W))**2)

    return J_style_layer


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(model, style_layers=STYLE_LAYERS):

    J_style = 0.0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)

        a_G = out   # ???

        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):

    J = alpha * J_content + beta * J_style

    return J


def model_nn(sess, input_image, num_iterations=2000):

    sess.run(tf.global_variables_initializer())

    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):
        sess.run(train_step)

        generated_image = sess.run(model['input'])

        if i % 50 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            save_image("output/" + str(i) + ".jpg", generated_image)

    save_image('output/generated_image.jpg', generated_image)

    return generated_image












if __name__ == '__main__':
    tf.reset_default_graph()

    with tf.Session() as test:
        np.random.seed(3)
        J_content = np.random.randn()
        J_style = np.random.randn()
        J = total_cost(J_content, J_style)
        print("J = " + str(J))
    # model = load_vgg_model("imagenet-vgg-verydeep-19.mat")
    # print(model)
    #
    # ### do forward test
    # # img = cv2.imread('images/style.jpg')
    # # img = cv2.resize(img, (400, 300))
    # # img = np.expand_dims(img, axis=0)
    # # with tf.Session() as sess:
    # #     sess.run(tf.global_variables_initializer())
    # #     model['input'].assign(img)
    # #     result = sess.run(model['conv4_2'])
    # #     print(result)
    #
    # img = cv2.imread('images/content.jpg')
    # cv2.imshow("content", img)
    # cv2.waitKey()

    # tf.reset_default_graph()
    #
    # with tf.Session() as test:
    #     tf.set_random_seed(1)
    #     a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    #     a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    #     J_content = compute_content_cost(a_C, a_G)
    #     print("J_content = " + str(J_content.eval()))

    # tf.reset_default_graph()
    #
    # with tf.Session() as test:
    #     tf.set_random_seed(1)
    #     A = tf.random_normal([3, 2 * 1], mean=1, stddev=4)
    #     GA = gram_matrix(A)
    #
    #     print("GA = " + str(GA.eval()))

    # tf.reset_default_graph()
    #
    # with tf.Session() as test:
    #     tf.set_random_seed(1)
    #     a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    #     a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    #     J_style_layer = compute_layer_style_cost(a_S, a_G)
    #
    #     print("J_style_layer = " + str(J_style_layer.eval()))

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    content_image = scipy.misc.imread("images/content.jpg")
    content_image = reshape_and_normalize_image(content_image)

    style_image = scipy.misc.imread('images/style.jpg')
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image)
    # imshow(generated_image[0])
    # plt.show()


    model = load_vgg_model("imagenet-vgg-verydeep-19.mat")

    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)

    a_G = out   # a_G在model_nn中计算

    J_content = compute_content_cost(a_C, a_G)

    sess.run(model['input'].assign(style_image))
    J_style = compute_style_cost(model, STYLE_LAYERS)
    J = total_cost(J_content, J_style)

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)

    model_nn(sess, generated_image)