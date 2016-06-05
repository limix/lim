import h5py as h5

from .array import ArrayViewInterface
from .scalar import cast

class H5Path(ArrayViewInterface):
    def __init__(self, filepath, itempath, dtype=None):
        super(H5Path, self).__init__()
        self._filepath = filepath
        self._itempath = itempath
        self._dtype = dtype
        with h5.File(self._filepath, 'r') as f:
            harr = f[self._itempath]
            self._dtype = type(cast(harr[(0,) * harr.ndim].item(0)))

    def item(self, *args):
        with h5.File(self._filepath, 'r') as f:
            return cast(f[self._itempath][args], self._dtype)

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
