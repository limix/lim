import numpy as np


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

    if hasattr(npy_type, 'char'):
        if npy_type.char in ['S', 'a']:
            return bytes
        raise TypeError

    return npy_type


def npy2py_cast(npy_value):
    type_ = npy2py_type(type(npy_value))
    return type_(npy_value)


def asarray(*args, **kwargs):
    from . import NPyMatrix
    return NPyMatrix(np.asarray(*args, **kwargs))
