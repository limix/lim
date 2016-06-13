from __future__ import absolute_import

from pandas import read_csv

from ..data import Table
from ..data import Column

def reader(filepath, dtype=float, row_header=False, col_header=False):

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

    return table
