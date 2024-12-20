import numpy as np
import cupy as cp
from numpy.typing import ArrayLike
from layers import Softmax

eps = 1e-7  # between 1e-6 and 1e-9


def cross_entropy_loss(y_pred: ArrayLike, y_true: ArrayLike) -> cp.ndarray:
    """
    Compute cross entropy loss between true 1-hot encoded vector and softmax output of a predictor.
    """
    # Make sure to not have log(0)
    
    # Compute cross entropy loss

    # test out log_softmax:
    log_softmax = y_pred - np.log(
        np.sum(
            np.exp(y_pred - np.max(y_pred, axis=-1, keepdims=True)),
            axis=-1,
            keepdims=True,
        )
    )
    log_softmax = cp.clip(log_softmax, eps, 1 - eps)
    loss = -cp.sum(y_true * cp.log(log_softmax)) / y_true.shape[0]
    return loss
