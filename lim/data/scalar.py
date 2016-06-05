from ..math.scalar import isint

def cast(v, dtype=None):
    if dtype is not None:
        return dtype(v)

    if isint(v):
        return int(v)
    return v
