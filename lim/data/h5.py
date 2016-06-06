import h5py as h5

from numpy import asarray

from .array import ArrayViewInterface
from .scalar import npy2py_type

class H5Path(ArrayViewInterface):
    def __init__(self, filepath, itempath, dtype=None):
        super(H5Path, self).__init__()
        self._filepath = filepath
        self._itempath = itempath
        if dtype is None:
            with h5.File(self._filepath, 'r') as f:
                harr = f[self._itempath]
                dtype = harr[(0,) * harr.ndim].dtype
                self._dtype = npy2py_type(dtype)
        else:
            self._dtype = dtype

    def item(self, *args):
        with h5.File(self._filepath, 'r') as f:
            return self._dtype(f[self._itempath][args])

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        with h5.File(self._filepath, 'r') as f:
            return f[self._itempath].shape

    @property
    def ndim(self):
        return len(self.shape)

    def _create_array(self, lslice):
        with h5.File(self._filepath, 'r') as f:
            return f[self._itempath][lslice].astype(self._dtype)
