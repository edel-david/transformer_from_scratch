from types import NoneType
from typing import Union

import numpy as np
import base64
import zlib


def compress_numpy_array(array: Union[np.ndarray, None]) -> dict:

    if not isinstance(type(array), NoneType):
        compressed_bytes = zlib.compress(array.tobytes())
        array_base64 = base64.b64encode(compressed_bytes).decode("utf-8")

        return {"data": array_base64, "dtype": str(array.dtype), "shape": array.shape}

    else:
        return {"data": None, "dtype": None, "shape": None}


def decompress_numpy_array(data_dict: dict) -> Union[np.ndarray, None]:

    if not isinstance(type(data_dict["data"]), NoneType):
        compressed_bytes = base64.b64decode(data_dict["data"])
        array_bytes = zlib.decompress(compressed_bytes)
        return np.frombuffer(array_bytes, dtype=data_dict["dtype"]).reshape(
            data_dict["shape"]
        )
