from pandas import DataFrame

from .column import Column

class Table(object):
    def __init__(self):
        self._df = DataFrame()

    def add(self, c):
        for (i, v) in iter(c.items()):
            self._df.set_value(i, c.name, v)

    @property
    def index_name(self):
        return self._df.index.name

    @index_name.setter
    def index_name(self, v):
        self._df.index.name = v

    @property
    def index(self):
        return self._df.index.values

    @property
    def columns(self):
        return self._df.columns.values

    def set_index_value(self, old_val, new_val):
        self._df.index[old_val] = new_val

    def __getitem__(self, colname):
        return Column(colname, self.index, self._df[colname].values)

    def __getattr__(self, colname):
        return Column(colname, self.index, self._df[colname].values)

    @property
    def shape(self):
        return self._df.shape

    @property
    def ndim(self):
        return 2

    @property
    def dtypes(self):
        return self._df.dtypes

    def as_matrix(self):
        return self._df.as_matrix()
