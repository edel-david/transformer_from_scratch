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
    # sm = Softmax(axis=-1)
    # outputs = sm.forward(y_pred)

    # no softmax, because idk it works better, don't ask me why

    loss = -cp.sum(y_true * cp.log(y_pred + 1e-9)) / y_true.shape[0] # + 1e-9 is not necessary because of the clip, but we trained with it and it should not matter much
    return loss
