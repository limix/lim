from __future__ import absolute_import

from pandas import read_csv

from ..data.array import ArrayViewInterface

def reader(filepath, dtype=float, row_header=False, col_header=False):
    return CSVPath(filepath, dtype=dtype, row_header=row_header,
                   col_header=col_header)

class CSVPath(ArrayViewInterface):
    def __init__(self, filepath, dtype=float,
                 row_header=False, col_header=False):
        super(CSVPath, self).__init__()
        self._filepath = filepath

        header = 0 if col_header else None
        index_col = 0 if row_header else None

        data = read_csv(filepath, header=header, index_col=index_col,
                              dtype=dtype)

        self.set_axis_name(0, data.index.name)
        self.set_axis_name(1, data.columns.name)

        self.set_axis_values(0, data.index.values)
        self.set_axis_values(1, data.columns.values)

        self._X = data.as_matrix()
        self._dtype = dtype

    def item(self, *args):
        return self._X[args]

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._X.shape

    @property
    def ndim(self):
        return len(self.shape)

    def _create_array(self, lslice):
        return self._X[lslice[0],lslice[1]]
