from __future__ import absolute_import

from pandas import read_csv

from ..data import Table
from ..data import Column

def reader(filepath, dtype=float, row_header=False, col_header=False):
    return CSVPath(filepath, dtype=dtype, row_header=row_header,
                   col_header=col_header)

class CSVPath(object):
    def __init__(self, filepath, dtype=float,
                 row_header=False, col_header=False):
        self._filepath = filepath

        header = 0 if col_header else None
        index_col = 0 if row_header else None

        data = read_csv(filepath, header=header, index_col=index_col)
        data = data.astype(dtype)

        table = Table()
        for (i, cn) in enumerate(data):
            if not col_header or data[cn].name is None:
                data[cn].name = 'column_name_%d' % i
            c = Column(data[cn].name, data.index.values, data[cn].values)
            table.add(c)

        if data.index.name is None:
            table.index_name = 'index_name'
        else:
            table.index_name = data.index.name


    #     self.set_axis_name(0, data.index.name)
    #     self.set_axis_name(1, data.columns.name)
    #
    #     self.set_axis_values(0, data.index.values)
    #     self.set_axis_values(1, data.columns.values)
    #
    #     self._X = data.as_matrix()
    #     self._dtype = dtype
    #
    # def item(self, *args):
    #     return self._X[args]
    #
    # @property
    # def dtype(self):
    #     return self._dtype
    #
    # @property
    # def shape(self):
    #     return self._X.shape
    #
    # @property
    # def ndim(self):
    #     return len(self.shape)
    #
    # def _create_array(self, lslice):
    #     return self._X[lslice[0],lslice[1]]
