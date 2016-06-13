import h5py as h5

from numpy import asarray
from numpy import arange
from numpy import atleast_2d

from ..data import Table
from ..data import Column
from ..util.type import npy2py_type

def reader(filepath, itempath, dtype=None):

    with h5.File(filepath, 'r') as f:

        if dtype is None:
            dtype = npy2py_type(f[itempath].dtype)

        arr = asarray(f[itempath], dtype=dtype)
        arr = atleast_2d(arr)

    index = arange(arr.shape[0], dtype=int)

    table = Table()
    for i in range(arr.shape[1]):
        column_name = 'column_name_%d' % i
        c = Column(column_name, index, arr[:,i])
        table.add(c)

    table.index_name = 'index_name'
    return table
