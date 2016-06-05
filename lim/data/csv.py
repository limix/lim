from __future__ import absolute_import

import csv

from numpy import asarray

from .slice import check_index
from .array import ArrayViewInterface
from .scalar import cast

class CSVPath(ArrayViewInterface):
    def __init__(self, filepath, dtype=float):
        super(CSVPath, self).__init__()
        self._filepath = filepath
        self._dtype = dtype

    def item(self, *args):
        if len(args) == 1:
            r, c = args[0], 0
        else:
            r, c = args[0], args[1]

        i = 0
        with open(self._filepath) as csvfile:
            reader = csv.reader(csvfile)
            row = next(reader)
            while i < r:
                row = next(reader)
                i += 1
            return cast(row[c], self._dtype)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        with open(self._filepath) as csvfile:
            nc = len(next(csv.reader(csvfile)))
            nr = sum(1 for row in csv.reader(csvfile)) + 1
        return nr, nc

    @property
    def ndim(self):
        return len(self.shape)

    def _create_array(self, lslice):
        if self.ndim == 1:
            assert len(lslice) == 1
            return self._create_1d_array(lslice)
        if self.ndim == 2:
            assert len(lslice) == 2
            return self._create_2d_array(lslice[0], lslice[1])
        assert False

    def __array__(self):
        return self._create_array((slice(0, None, 1),)*self.ndim)

    def _create_1d_array(self, slice_):
        pass

    def _create_2d_array(self, row_slice, col_slice):
        rs = row_slice
        cs = col_slice
        arr = []
        with open(self._filepath) as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                if check_index(i, rs) == -1:
                    break
                elif check_index(i, rs) == 1:
                    n = len(row)
                    srow = [row[j] for j in range(n) if check_index(j, cs) == 1]
                    arr.append(srow)
                i += 1
        return asarray(arr).astype(self._dtype)
