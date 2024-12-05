import numpy as np
import cupy as cp
from numpy.typing import ArrayLike

eps = 1e-6


def cross_entropy_loss(y_pred: ArrayLike, y_true: ArrayLike) -> cp.ndarray:
    """
    Compute cross entropy loss between true 1-hot encoded vector and softmax output of a predictor.
    """
    # Make sure to not have log(0)
    y_pred = cp.clip(y_pred, eps, 1 - eps)
    # Compute cross entropy loss
    # TODO: label smoothing??
    y_pred = cp.exp(y_pred)
    softmax_sum = y_pred.sum(axis=1).reshape((y_pred.shape[0], 1))
    y_pred = y_pred / softmax_sum
    loss = -cp.log((y_pred * y_true).sum(axis=1))
    loss = cp.average(loss)
    # or:
    # loss = cp.sum(cp.log(y_pred * y_true))
    # raise NotImplementedError("Implement the Cross-Entropy loss")
    return loss
