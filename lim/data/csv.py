from __future__ import absolute_import

import csv

from numpy import asarray

from .slice import check_index
from .array import ArrayViewInterface

class CSVPath(ArrayViewInterface):
    def __init__(self, filepath, dtype=float):
        super(CSVPath, self).__init__()
        self._filepath = filepath
        self._dtype = dtype

    def _is_col_layout(self):
        with open(self._filepath) as csvfile:
            row = next(csv.reader(csvfile))
            return len(row) == 1

    def item(self, *args):
        if len(args) == 1:
            if self._is_col_layout():
                return self._item_1d_col(args[0])
            return self._item_1d_row(args[0])
        return self._item_2d(args[0], args[1])

    def _item_2d(self, r, c):
        i = 0
        with open(self._filepath) as csvfile:
            reader = csv.reader(csvfile)
            row = next(reader)
            while i < r:
                row = next(reader)
                i += 1
            return self._dtype(row[c])

    def _item_1d_row(self, i):
        with open(self._filepath) as csvfile:
            row = next(csv.reader(csvfile))
            return self._dtype(row[i])

    def _item_1d_col(self, i):
        with open(self._filepath) as csvfile:
            reader = csv.reader(csvfile)
            row = next(reader)
            j = 0
            while j < i:
                row = next(reader)
                j += 1
            return self._dtype(row[0])

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        with open(self._filepath) as csvfile:
            nc = len(next(csv.reader(csvfile)))
            nr = sum(1 for row in csv.reader(csvfile)) + 1
        if nr == 1:
            return (nc,)
        if nc == 1:
            return (nr,)
        return nr, nc

    @property
    def ndim(self):
        return len(self.shape)

    def _create_array(self, lslice):
        if self.ndim == 1:
            assert len(lslice) == 1
            if self._is_col_layout():
                return self._create_1d_array_col(lslice[0])
            return self._create_1d_array_row(lslice[0])
        if self.ndim == 2:
            assert len(lslice) == 2
            return self._create_2d_array(lslice[0], lslice[1])
        assert False

    def _create_1d_array_row(self, slice_):
        with open(self._filepath) as csvfile:
            reader = csv.reader(csvfile)
            row = next(reader)
            n = len(row)
            srow = [row[j] for j in range(n) if check_index(j, slice_) == 1]
            return asarray(srow).astype(self._dtype)

    def _create_1d_array_col(self, slice_):
        with open(self._filepath) as csvfile:
            reader = csv.reader(csvfile)
            srows = []
            for (i, row) in enumerate(reader):
                if check_index(i, slice_) == 1:
                    srows.append(row[0])
            return asarray(srows).astype(self._dtype)

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
