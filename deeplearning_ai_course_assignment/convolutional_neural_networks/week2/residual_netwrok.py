# coding: utf-8

import h5py
import numpy as np
from keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, ZeroPadding2D, MaxPooling2D, \
    AveragePooling2D, Flatten, Dense
from keras.models import Model
from keras.initializers import glorot_uniform
import tensorflow as tf
import keras.backend as K

# from Ipython.display import SVG
# from keras.utils import plot_model


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

def identity_block(X, f, filters, stage, block):
    '''
    三层的恒等残差块,输入输出层具有相同的维度情形使用
    三层实现：第一层、第三层通常分别为1x1升维、降维操作，以降低计算量和参数数量
    :param X:
    :param f:
    :param filters: 主路径filters数目
    :param stage:
    :param block:
    :return:
    '''
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X

    # 1st layer
    X = Conv2D(filters = F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base+'2a', \
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    # 2nd layer
    X = Conv2D(filters = F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base+'2b', \
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # 3rd layer
    X = Conv2D(filters = F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base+'2c', \
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    '''
    输入输出维度不同时，shortcut做卷积变换维度
    :param X:
    :param f:
    :param filters:
    :param stage:
    :param block:
    :param s:
    :return:
    '''
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    # stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2),name = 'avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


if __name__ == '__main__':
    # tf.reset_default_graph()
    #
    # np.random.seed(1)
    # A_prev = tf.placeholder('float', [3, 4, 4, 6])
    # X = np.random.randn(3, 4, 4, 6)
    # A = indentity_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     out = sess.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    #     print("out = " + str(out[0][1][1][0]))

    # tf.reset_default_graph()
    # np.random.seed(1)
    # A_prev = tf.placeholder("float", [3, 4, 4, 6])
    # X = np.random.randn(3, 4, 4, 6)
    # A = convolutional_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
    # with tf.Session() as test:
    #     test.run(tf.global_variables_initializer())
    #     out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    #     print("out = " + str(out[0][1][1][0]))


    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    model = ResNet50(input_shape=(64, 64, 3), classes=6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, batch_size=32)
    preds = model.evaluate(X_test, Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
    # plot_model(model, to_file='model.png')
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))

