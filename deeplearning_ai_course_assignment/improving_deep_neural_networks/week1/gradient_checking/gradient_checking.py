# coding: utf-8
import numpy as np
from gradient_checking_utils import *

import logging
logging.basicConfig(format='%(levelname)s: %(message)s\t%(pathname)s[line:%(lineno)d]',
                    level=logging.ERROR) # DEBUG ERROR


def forward_propagation_n(X, Y, parameters):
    AL, caches = L_model_forward(X, parameters)
    cost = compute_cost(AL, Y)

    return AL, cost, caches


def backward_propagation_n(AL, Y, caches):
    grads = L_model_backward(AL, Y, caches)

    return grads


def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    parameters_vector, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_vector.shape[0]
    J_puls = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    grad_approx = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        theta_plus = np.copy(parameters_vector)
        theta_plus[i][0] = theta_plus[i][0] + epsilon
        AL,J_puls[i], _1 = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))

        theta_minus = np.copy(parameters_vector)
        theta_minus[i][0] = theta_minus[i][0] - epsilon
        AL, J_minus[i], _1 = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))

        grad_approx[i] = (J_puls[i] - J_minus[i]) / (2*epsilon)

    print(grad_approx.shape)
    print(grad.shape)
    print('gradients: ', gradients)
    numerator = np.linalg.norm(grad_approx - grad)
    denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approx)
    difference = numerator / denominator

    if difference < 1e-7:
        print('backward propagation works well !!!')
    else:
        print('backward propagation may be wrong !!!')

    return difference


if __name__ == '__main__':
    X, Y, parameters = gradient_check_n_test_case()

    AL, cost, caches = forward_propagation_n(X, Y, parameters)
    gradients = backward_propagation_n(AL, Y, caches)
    difference = gradient_check_n(parameters, gradients, X, Y)
    print('difference = {}'.format(difference))
