# def read_mslice(filepath, mslice, shape):
#     from ...data import Slice
#
#     l = mslice[0]
#     r = mslice[1]
#
#     if isinstance(l, int) and isinstance(r, int):
#         return read_item(filepath, l, r, shape)
#
#     if isinstance(l, int) and isinstance(r, Slice):
#         return read_row(filepath, l, r)

def _normalize(G):
    from numpy import nan

    G = G.astype(float)
    G[G == 3] = nan
    return G

def _read_row_slice(filepath, r, start, stop, step, shape):
    from . import bed_ffi
    from numpy import empty
    from numpy import nan

    fp = bed_ffi.ffi.new("char[]", filepath)
    G = empty((stop - start)//step, dtype=int)
    pointer = bed_ffi.ffi.cast("long*", G.ctypes.data)

    bed_ffi.lib.bed_read_row_slice(fp, shape[0], shape[1], r,
                                   start, stop, step, pointer)

    return G

def read_row_slice(filepath, r, start, stop, step, shape):
    from . import bed_ffi

    major = _bed_major(filepath)

    if major == 'm':
        shape = (shape[1], shape[0])
        G = _read_col_slice(filepath, r, start, stop, step, shape)
    elif major == 's':
        G = _read_row_slice(filepath, r, start, stop, step, shape)

    return _normalize(G)

def _read_col_slice(filepath, c, start, stop, step, shape):
    from . import bed_ffi
    from numpy import empty
    from numpy import nan

    fp = bed_ffi.ffi.new("char[]", filepath)
    G = empty((stop - start)//step, dtype=int)
    pointer = bed_ffi.ffi.cast("long*", G.ctypes.data)

    bed_ffi.lib.bed_read_col_slice(fp, shape[0], shape[1], c,
                                   start, stop, step, pointer)

    return G

def read_col_slice(filepath, c, start, stop, step, shape):
    from . import bed_ffi

    major = _bed_major(filepath)

    if major == 'm':
        shape = (shape[1], shape[0])
        G = _read_row_slice(filepath, c, start, stop, step, shape)
    elif major == 's':
        G = _read_col_slice(filepath, c, start, stop, step, shape)

    return _normalize(G)

def read_item(filepath, r, c, shape):
    from . import bed_ffi
    from numpy import nan

    major = _bed_major(filepath)
    if major == 'm':
        r, c = c, r
        shape = (shape[1], shape[0])

    fp = bed_ffi.ffi.new("char[]", filepath)
    v = bed_ffi.lib.bed_read_item(fp, shape[0], shape[1], r, c)

    if v == 3:
        v = nan

    return float(v)

def _read(filepath, shape):
    from . import bed_ffi
    from numpy import empty
    from numpy import nan

    fp = bed_ffi.ffi.new("char[]", filepath)
    G = empty(shape, dtype=int)
    pointer = bed_ffi.ffi.cast("long*", G.ctypes.data)

    bed_ffi.lib.bed_read(fp, shape[0], shape[1], pointer)

    return G

def read(filepath, shape):

    major = _bed_major(filepath)
    if major == 'm':
        shape = (shape[1], shape[0])
        G = _read(filepath, shape)
        G = G.T
    elif major == 's':
        G = _read(filepath, shape)

    return _normalize(G)

def read_mslice(filepath, shape, mslice):
    # ESSA PORRA NAO FOI IMPLEMENTADA AINDA
    # import pytest; pytest.set_trace()
    from . import bed_ffi
    from numpy import empty
    from numpy import nan

    fp = bed_ffi.ffi.new("char[]", filepath)
    nrows = (rslice.stop - rslice.start) // rslice.step
    ncols = (cslice.stop - cslice.start) // cslice.step
    G = empty((nrows, ncols), dtype=int)
    pointer = bed_ffi.ffi.cast("long*", G.ctypes.data)

    bed_ffi.lib.bed_read_slice(fp, shape[0], shape[1], c,
                                   start, stop, step, pointer)

    # bed_read_slice(char* filepath, int nrows, int ncols,
    #            int r_start, int r_stop, int r_step,
    #            int c_start, int c_stop, int c_step,
    #            long* matrix)
    return G

def _bed_major(filepath):
    from . import bed_ffi

    fp = bed_ffi.ffi.new("char[]", filepath)
    major = bed_ffi.lib.bed_major(fp)
    if major == 0:
        return 's' # samples first
    elif major == 1:
        return 'm' # markers first
    raise ValueError("Doesn't look like a valid BED file format.")

def _read_slice(filepath, rslice, cslice, shape):
    from . import bed_ffi
    from numpy import empty
    from numpy import nan

    fp = bed_ffi.ffi.new("char[]", filepath)
    nrows = (rslice.stop - rslice.start) // rslice.step
    ncols = (cslice.stop - cslice.start) // cslice.step
    G = empty((nrows, ncols), dtype=int)
    pointer = bed_ffi.ffi.cast("long*", G.ctypes.data)

    bed_ffi.lib.bed_read_slice(fp, shape[0], shape[1], c,
                                   start, stop, step, pointer)

    return G
