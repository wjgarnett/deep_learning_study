# coding: utf-8

import tensorflow as tf
from inception_blocks_v2 import *
from keras import backend as K
K.set_image_data_format('channels_first')


def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.square(anchor - positive)
    neg_dist = tf.square(anchor - negative)

    basic_loss = tf.reduce_sum(pos_dist - neg_dist, axis=1) + alpha

    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


if __name__ == '__main__':
    # FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    # print('total params: ', FRmodel.count_params())

    with tf.Session() as test:
        tf.set_random_seed(1)
        y_true = (None, None, None)
        y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
                  tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
                  tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
        loss = triplet_loss(y_true, y_pred)
        print("loss = " + str(loss.eval()))