from types import NoneType
from typing import Union

import numpy as np
import cupy as cp
import base64
import zlib
import wandb
global step
step= 0

def log(name,vector,step_arg=None):
    global step
    if step_arg is not None:
        step = step_arg
    wandb.log({f"{name}_mean": vector.mean().item()},  step=step)
    wandb.log({f"{name}_var":vector.var().item()},step=step)

def log_one(name,value,step_arg=None):
    global step
    if step_arg is not None:
        step = step_arg
    wandb.log({f"{name}": value},  step=step)
    

def compress_numpy_array(array: Union[cp.ndarray, None]) -> dict:

    if not isinstance(type(array), NoneType):
        compressed_bytes = zlib.compress(array.tobytes())
        array_base64 = base64.b64encode(compressed_bytes).decode("utf-8")

        return {"data": array_base64, "dtype": str(array.dtype), "shape": array.shape}

    else:
        return {"data": None, "dtype": None, "shape": None}


def decompress_numpy_array(data_dict: dict) -> Union[cp.ndarray, None]:

    if not isinstance(type(data_dict["data"]), NoneType):
        compressed_bytes = base64.b64decode(data_dict["data"])
        array_bytes = zlib.decompress(compressed_bytes)
        return cp.frombuffer(array_bytes, dtype=data_dict["dtype"]).reshape(
            data_dict["shape"]
        )
