import numpy as np


def min_distance_to_cut_loss(y_true, y_pred):
    loss_array = np.abs(
        y_true.numpy() - np.arange(y_pred.shape[1]).reshape((-1, 1))
    ).min(axis=1)
    loss_array = loss_array**2

    return y_pred * loss_array


def inv_exp_distance_to_cut_loss(y_true, y_pred, lbda=0.5):
    loss_array = np.abs(
        y_true.numpy() - np.arange(y_pred.shape[1]).reshape((-1, 1))
    ).min(axis=1)
    loss_array = np.exp(-lbda * loss_array)

    return (y_pred - loss_array) ** 2
