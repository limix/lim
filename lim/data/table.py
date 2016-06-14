from numpy import where

from pandas import DataFrame

from .column import Column
from ..util.type import npy2py_type

class Table(object):
    def __init__(self, df=None):
        self._df = DataFrame() if df is None else df

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
    def index_values(self):
        return self._df.index.values

    @index_values.setter
    def index_values(self, values):
        self._df.index = values

    @property
    def columns(self):
        return self._df.columns.values

    def set_index_value(self, old_val, new_val):
        otype = npy2py_type(type(old_val))
        ntype = npy2py_type(type(new_val))

        index_name = self._df.index.name
        values = self._df.index.values
        i = where(values == old_val)[0][0]

        if otype != ntype:
            values = values.astype(ntype)

        values[i] = new_val

        self._df.index = values

        self._df.index.name = index_name

    def __getitem__(self, colname):
        return Column(colname, self.index_values, self._df[colname].values)

    def __getattr__(self, colname):
        return Column(colname, self.index_values, self._df[colname].values)

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

    def __repr__(self):
        return repr(self._df)

    def __str__(self):
        return bytes(self._df)
