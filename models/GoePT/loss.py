import numpy as np
from numpy.typing import ArrayLike

eps = 1e-6


def cross_entropy_loss(y_pred: ArrayLike, y_true: ArrayLike) -> np.ndarray:
    """
    Compute cross entropy loss between true 1-hot encoded vector and softmax output of a predictor.
    """
    # Make sure to not have log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # Compute cross entropy loss
    # TODO: label smoothing??
    y_pred = np.exp(y_pred)
    softmax_sum = y_pred.sum(axis=1).reshape((y_pred.shape[0], 1))
    y_pred = y_pred / softmax_sum
    loss = -np.log((y_pred * y_true).sum(axis=1))
    loss = np.average(loss)
    # or:
    # loss = np.sum(np.log(y_pred * y_true))
    # raise NotImplementedError("Implement the Cross-Entropy loss")
    return loss
