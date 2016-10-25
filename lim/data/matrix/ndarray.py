from numpy import atleast_2d
from numpy import arange

from bidict import bidict

from .interface import MatrixInterface


class NPyMatrix(MatrixInterface):

    def __init__(self, arr, sample_id=None, marker_id=None):
        super(NPyMatrix, self).__init__()
        self._arr = atleast_2d(arr)

        if sample_id is None:
            self._sample_id = arange(self._arr.shape[0], dtype=int)
        else:
            self._sample_id = sample_id

        if marker_id is None:
            self._marker_id = arange(self._arr.shape[1], dtype=int)
        else:
            self._marker_id = marker_id

        self._sample_map = None
        self._marker_map = None
        self._update_map()

    def _update_map(self):
        self._sample_map = bidict(zip(self._sample_id,
                                      arange(self.shape[0], dtype=int)))

        self._marker_map = bidict(zip(self._marker_id,
                                      arange(self.shape[1], dtype=int)))

    def item(self, *args):
        return self._arr.item(*args)

    def __getitem__(self, args):
        return self._arr.__getitem__(args)

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    def __repr__(self):
        return self._arr.__repr__()

    def __str__(self):
        return self._arr.__str__()

    def __array__(self, *args, **kwargs):
        return self._arr.__array__(*args, **kwargs)

    @property
    def sample_id(self):
        return self._sample_id

    @property
    def marker_id(self):
        return self._marker_id
