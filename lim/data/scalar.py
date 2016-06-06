from ..math.scalar import isint

import numpy as np

def cast(v, dtype=None):
    if dtype is not None:
        return dtype(v)

    if isint(v):
        return int(v)
    return v

def npy2py_type(npy_type):
    int_types = [
        np.int_,
        np.intc,
        np.intp,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64]

    float_types = [
        np.float_,
        np.float16,
        np.float32,
        np.float64]

    bytes_types = [
        np.str_,
        np.string_]

    if npy_type in int_types:
        return int
    if npy_type in float_types:
        return float
    if npy_type in bytes_types:
        return bytes

    if npy_type.char in ['S', 'a']:
        return bytes

    raise TypeError
